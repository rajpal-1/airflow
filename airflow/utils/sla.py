# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import absolute_import

import logging

from six import string_types
from sqlalchemy import or_

import airflow.models  # pylint: disable=cyclic-import
from airflow.utils import asciiart
from airflow.utils.email import send_email
from airflow.utils.session import provide_session
from airflow.utils.state import State

log = logging.getLogger(__name__)


def yield_uncreated_runs(dag, last_scheduled_run, ts):
    """
    Yield new DagRuns that haven't been created yet. This functionality is
    important to SLA misses because it is possible for the scheduler to fall
    so far behind that it cannot create a DAGRun when it is supposed to (like
    if it is offline, or if there are strict concurrency limits). We need to
    understand and alert on what DAGRuns *should* have been created by this
    point in time.

    Uncreated DagRuns are found by following a dag's schedule until the next
    execution is greater than ts, a timestamp that is very close to the current
    moment.
    """

    # TODO: A lot of this logic is duplicated from the scheduler. It would
    # be better to have one function that yields upcoming DAG runs in a
    # consistent way that is usable for both use cases.

    # Start by assuming that there is no next run.
    next_run_date = None

    # The first DAGRun has not been created yet.
    if not last_scheduled_run:
        task_start_dates = [t.start_date for t in dag.tasks]
        if task_start_dates:
            next_run_date = dag.normalize_schedule(min(task_start_dates))
    # The DagRun is @once and has already happened.
    elif dag.schedule_interval == '@once':
        return
    # Start from the next "normal" run.
    else:
        next_run_date = dag.following_schedule(last_scheduled_run.execution_date)

    while True:
        # There should be a next execution.
        if not next_run_date:
            return

        # The next execution shouldn't be in the future.
        if next_run_date > ts:
            return

        # The next execution shouldn't be beyond the DAG's end date.
        # n.b. - tasks have their own end dates checked later
        if next_run_date and dag.end_date and next_run_date > dag.end_date:
            return

        # Calculate the end of this execution period.
        if dag.schedule_interval == '@once':
            period_end = next_run_date
        else:
            period_end = dag.following_schedule(next_run_date)

        # The next execution shouldn't still be mid-period.
        if period_end > ts:
            return

        # We've passed every filter; this is a valid future DagRun that
        # presumably hasn't been scheduled due to concurrency limits.
        # Create and yield a fake DAGRun, which won't exist in the db yet.
        next_run = airflow.models.DagRun(
            dag_id=dag.dag_id,
            run_id='manual__' + next_run_date.isoformat(),
            execution_date=next_run_date,
            start_date=ts,
            state=State.NONE,
            external_trigger=False,
        )
        next_run.dag = dag
        yield next_run

        # Examine the next date.
        next_run_date = dag.following_schedule(next_run_date)


def yield_uncreated_tis(dag_run, ts):
    """
    Given an unscheduled `DagRun`, yield any unscheduled TIs that will exist
    for it in the future, respecting the end date of the DAG and task. See note
    above for why this is important for SLA notifications.

    Uncreated TIs are found by following a dag's schedule until the next execution
    is greater than ts, a timestamp that is very close to the current moment.
    """
    for task in dag_run.dag.tasks:
        end_dates = []
        if dag_run.dag.end_date:
            end_dates.append(dag_run.dag.end_date)
        if task.end_date:
            end_dates.append(task.end_date)

        # Create TIs if there is no end date, or it hasn't happened yet.
        if not end_dates or ts < min(end_dates):
            yield airflow.models.TaskInstance(task, dag_run.execution_date)


def get_sla_misses(ti, session):
    """
    Get all SLA misses that match a particular TaskInstance. There may be
    several matches if the Task has several independent SLAs.
    """
    SM = airflow.models.SlaMiss
    return session.query(SM).filter(
        SM.dag_id == ti.dag_id,
        SM.task_id == ti.task_id,
        SM.execution_date == ti.execution_date
    ).all()


def create_sla_misses(ti, timestamp, session):
    """
    Determine whether a TaskInstance has missed any SLAs as of a provided
    timestamp. If it has, create `SlaMiss` objects in the provided session.
    Note that one TaskInstance can have multiple SLA miss objects: for example,
    it can both start late and run longer than expected.
    """
    # Skipped task instances will never trigger SLAs because they
    # were intentionally not scheduled. Though, it's still a valid and
    # interesting SLA miss if a task that's *going* to be skipped today is
    # late! That could mean that an upstream task is hanging.
    if ti.state == State.SKIPPED:
        return

    log.debug("Calculating SLA misses for %s as of %s", ti, timestamp)

    SM = airflow.models.SlaMiss

    # Get existing SLA misses for this task instance.
    ti_misses = {sm.sla_type: sm for sm in get_sla_misses(ti, session)}

    # Calculate SLA misses that don't already exist. Wrapping exceptions here
    # is important so that an exception in one type of SLA doesn't
    # prevent other task SLAs from getting triggered.

    # SLA Miss for Expected Duration
    # n.b. - this one can't be calculated until the ti has started!
    if SM.TASK_DURATION_EXCEEDED not in ti_misses \
            and ti.task.expected_duration and ti.start_date:
        try:
            if ti.state in State.finished():
                duration = ti.end_date - ti.start_date
            else:
                # Use the current time, if the task is still running.
                duration = timestamp - ti.start_date

            if duration > ti.task.expected_duration:
                log.debug("Task instance %s's duration of %s > its expected "
                          "duration of %s. Creating duration exceeded SLA miss.",
                          ti, duration, ti.task.expected_duration)
                session.merge(SM(
                    task_id=ti.task_id,
                    dag_id=ti.dag_id,
                    execution_date=ti.execution_date,
                    sla_type=SM.TASK_DURATION_EXCEEDED,
                    timestamp=timestamp))
            else:
                log.debug("Task instance %s's duration of %s <= its expected "
                          "duration of %s, SLA not yet missed.",
                          ti, duration, ti.task.expected_duration)
        except Exception:  # pylint: disable=broad-except
            log.exception(
                "Failed to calculate expected duration SLA miss for "
                "task instance %s",
                ti
            )

    # SLA Miss for Expected Start
    if SM.TASK_LATE_START not in ti_misses and ti.task.expected_start:
        try:
            # If a TI's exc date is 01-01-2018, we expect it to start by the next
            # execution date (01-02-2018) plus a delta of expected_start.
            expected_start = ti.task.dag.following_schedule(ti.execution_date)
            expected_start += ti.task.expected_start

            # The case where we have started the ti, but late
            if ti.start_date and ti.start_date > expected_start:
                log.debug("Task instance %s's actual start %s > its expected "
                          "start of %s. Creating late start SLA miss.",
                          ti, ti.start_date, expected_start)
                session.merge(SM(
                    task_id=ti.task_id,
                    dag_id=ti.dag_id,
                    execution_date=ti.execution_date,
                    sla_type=SM.TASK_LATE_START,
                    timestamp=timestamp))

            # The case where we haven't even started the ti yet
            elif timestamp > expected_start:
                log.debug("Task instance %s has not started by its expected "
                          "start of %s. Creating late start SLA miss.",
                          ti, expected_start)
                session.merge(SM(
                    task_id=ti.task_id,
                    dag_id=ti.dag_id,
                    execution_date=ti.execution_date,
                    sla_type=SM.TASK_LATE_START,
                    timestamp=timestamp))
            else:
                log.debug("Task instance %s's expected start of %s hasn't "
                          "happened yet, SLA not yet missed.",
                          ti, expected_start)
        except Exception:  # pylint: disable=broad-except
            log.exception(
                "Failed to calculate expected start SLA miss for "
                "task instance %s",
                ti
            )

    # SLA Miss for Expected Finish
    if SM.TASK_LATE_FINISH not in ti_misses and ti.task.expected_finish:
        try:
            # If a TI's exc date is 01-01-2018, we expect it to finish by the next
            # execution date (01-02-2018) plus a delta of expected_finish.
            expected_finish = ti.task.dag.following_schedule(ti.execution_date)
            expected_finish += ti.task.expected_finish

            if ti.end_date and ti.end_date > expected_finish:
                log.debug("Task instance %s's actual finish %s > its expected "
                          "finish of %s. Creating late finish SLA miss.",
                          ti, ti.end_date, expected_finish)
                session.merge(SM(
                    task_id=ti.task_id,
                    dag_id=ti.dag_id,
                    execution_date=ti.execution_date,
                    sla_type=SM.TASK_LATE_FINISH,
                    timestamp=timestamp))

            elif timestamp > expected_finish:
                log.debug("Task instance %s has not finished by its expected "
                          "finish of %s. Creating late finish SLA miss.",
                          ti, expected_finish)
                session.merge(SM(
                    task_id=ti.task_id,
                    dag_id=ti.dag_id,
                    execution_date=ti.execution_date,
                    sla_type=SM.TASK_LATE_FINISH,
                    timestamp=timestamp))
            else:
                log.debug("Task instance %s's expected finish of %s hasn't "
                          "happened yet, SLA not yet missed.",
                          ti, expected_finish)
        except Exception:  # pylint: disable=broad-except
            log.exception(
                "Failed to calculate expected finish SLA miss for "
                "task instance %s",
                ti
            )


def send_sla_miss_email(context):
    """
    Send an SLA miss email. This is the default SLA miss callback.
    """
    sla_miss = context["sla_miss"]

    if sla_miss.sla_type == sla_miss.TASK_DURATION_EXCEEDED:
        email_function = send_task_duration_exceeded_email
    elif sla_miss.sla_type == sla_miss.TASK_LATE_START:
        email_function = send_task_late_start_email
    elif sla_miss.sla_type == sla_miss.TASK_LATE_FINISH:
        email_function = send_task_late_finish_email
    else:
        log.warning("Received unexpected SLA Miss type: %s", sla_miss.sla_type)
        return

    email_to, email_subject, email_body = email_function(context)
    send_email(email_to, email_subject, email_body)


def describe_task_instance(ti):
    """
    Return a string representation of the task instance.
    """
    return "{dag_id}.{task_id} [{exc_date}]".format(
        dag_id=ti.dag_id,
        task_id=ti.task_id,
        exc_date=ti.execution_date
    )


def get_sla_miss_subject(miss_type, ti):
    """
    Return a consistent subject line for SLA miss emails.
    """
    return "[airflow] [SLA] {miss_type} on {task_instance}".format(
        miss_type=miss_type,
        task_instance=describe_task_instance(ti)
    )


def get_subscribers(task_instances):
    """
    Return a list of unique emails from a list of task instances.
    """
    def _yield_subscribers(task_instances):
        for ti in task_instances:
            email = ti.task.email
            if isinstance(email, string_types):
                yield email
            elif email:
                for e in email:
                    yield e

    unique_emails = list(set(_yield_subscribers(task_instances)))
    log.debug("Found subscribers: %s", ", ".join(unique_emails))
    return unique_emails


@provide_session
def get_impacted_downstream_task_instances(task_instance, session=None):
    """
    Given a task instance that has had an SLA miss, return any
    downstream task instances that may have been impacted too. In this case, we
    mean any tasks that may now themselves be delayed due to the initial delay,
    even if the downstream tasks themselves do not have SLAs set.
    """
    dag = task_instance.task.dag

    TI = airflow.models.TaskInstance
    downstream_tasks = task_instance.task.get_flat_relatives(upstream=False)

    log.debug("Downstream task IDs of %s: %s", task_instance.task_id,
              ", ".join(t.task_id for t in downstream_tasks))

    # The intent is to capture states that indicate that work was never started
    # on a task, presumably because this task not achieving its SLA prevented
    # the downstream task from ever successfully starting.

    # It is possible for an upstream task to cause a downstream task to *fail*,
    # like if it never produces a required artifact. But in a well-behaved DAG
    # where dependencies are encoded properly, this shouldn't happen.
    blocked_states = (
        State.UPSTREAM_FAILED,
        State.SCHEDULED,
        State.QUEUED,
    )

    qry = (
        session
        .query(TI)
        .filter(TI.dag_id == dag.dag_id)
        .filter(TI.task_id.in_(list(t.task_id for t in downstream_tasks)))
        .filter(TI.execution_date == task_instance.execution_date)
        # State.NONE is actually a None, which is a null, which breaks
        # comparisons without writing them like this.
        .filter(
            or_(TI.state == None, TI.state.in_(blocked_states))  # noqa E711 pylint: disable=C0121
        )
        .order_by(TI.task_id)
    )

    # Make sure to set Task on each returned TI.
    impacted_downstreams = qry.all()
    for ti in impacted_downstreams:
        ti.task = dag.get_task(ti.task_id)

    log.debug("Impacted downstream task instances: %s", impacted_downstreams)
    return impacted_downstreams


def send_task_duration_exceeded_email(context):
    """
    This helper function is the default implementation for a duration SLA
    miss callback. It sends an email to all subscribers explaining that the
    task is taking too long, which downstream tasks may be impacted, and
    where to go for further information.

    Note that if the task instance hasn't been created yet (such as scheduler
    concurrency limits), this function may have a "dummy" task instance in its
    context. In that case, it will not exist in the db or have a full set of
    attributes.
    """

    ti = context["ti"]
    target_time = ti.task.expected_duration
    impacted_downstreams = get_impacted_downstream_task_instances(ti)

    # Get the subscribers for this task instance plus any downstreams
    email_to = get_subscribers([ti] + impacted_downstreams)
    email_subject = get_sla_miss_subject("Exceeded duration", ti)
    email_body = (
        "<pre><code>{task_string}</pre></code> missed an SLA: duration "
        "exceeded <pre><code>{target_time}</pre></code>.\n\n"

        "View Task Details: {ti_url}\n\n"

        "This may be impacting the following downstream tasks:\n"
        "<pre><code>{impacted_downstreams}\n{art}</pre></code>".format(
            task_string=describe_task_instance(ti),
            target_time=target_time,
            impacted_downstreams="\n".join(
                describe_task_instance(d) for d in impacted_downstreams),
            ti_url=ti.details_url,
            art=asciiart.snail)
    )

    return email_to, email_subject, email_body


def send_task_late_start_email(context):
    """
    This helper function is the default implementation for a late finish SLA
    miss callback. It sends an email to all subscribers explaining that the
    task hasn't started on time, which downstream tasks may be impacted, and
    where to go for further information.

    Note that if the task instance hasn't been created yet (such as scheduler
    concurrency limits), this function may have a "dummy" task instance in its
    context. In that case, it will not exist in the db or have a full set of
    attributes.
    """
    ti = context["ti"]

    target_time = ti.execution_date + ti.task.expected_start
    impacted_downstreams = get_impacted_downstream_task_instances(ti)

    # Get the subscribers for this task instance plus any downstreams
    email_to = get_subscribers([ti] + impacted_downstreams)
    email_subject = get_sla_miss_subject("Late start", ti)
    email_body = (
        "<pre><code>{task_string}</pre></code> missed an SLA: did not start "
        "by <pre><code>{target_time}</pre></code>.\n\n"

        "View Task Details: {ti_url}\n\n"

        "This may be impacting the following downstream tasks:\n"
        "<pre><code>{impacted_downstreams}\n{art}</pre></code>".format(
            task_string=describe_task_instance(ti),
            target_time=target_time,
            impacted_downstreams="\n".join(
                describe_task_instance(d) for d in impacted_downstreams),
            ti_url=ti.details_url,
            art=asciiart.snail)
    )

    return email_to, email_subject, email_body


def send_task_late_finish_email(context):
    """
    This helper function is the default implementation for a late finish SLA
    miss callback. It sends an email to all subscribers explaining that the
    task hasn't finished on time, which downstream tasks may be impacted, and
    where to go for further information.

    Note that if the task instance hasn't been created yet (such as scheduler
    concurrency limits), this function may have a "dummy" task instance in its
    context. In that case, it will not exist in the db or have a full set of
    attributes.
    """
    ti = context["ti"]
    target_time = ti.execution_date + ti.task.expected_finish
    impacted_downstreams = get_impacted_downstream_task_instances(ti)

    # Get the subscribers for this task instance plus any downstreams
    email_to = get_subscribers([ti] + impacted_downstreams)
    email_subject = get_sla_miss_subject("Late finish", ti)
    email_body = (
        "<pre><code>{task_string}</pre></code> missed an SLA: did not finish "
        "by <pre><code>{target_time}</pre></code>.\n\n"

        "View Task Details: {ti_url}\n\n"

        "This may be impacting the following downstream tasks:\n"
        "<pre><code>{impacted_downstreams}\n{art}</pre></code>".format(
            task_string=describe_task_instance(ti),
            target_time=target_time,
            impacted_downstreams="\n".join(
                describe_task_instance(d) for d in impacted_downstreams),
            ti_url=ti.details_url,
            art=asciiart.snail)
    )

    return email_to, email_subject, email_body
