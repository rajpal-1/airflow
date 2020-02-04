#
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
#
import multiprocessing
import os
import time
import unittest
from datetime import timedelta

import pytest
from mock import patch

from airflow import settings
from airflow.exceptions import AirflowException
from airflow.executors.sequential_executor import SequentialExecutor
from airflow.jobs.local_task_job import LocalTaskJob
from airflow.models.dag import DAG
from airflow.models.dagbag import DagBag
from airflow.models.taskinstance import SimpleTaskInstance, TaskInstance
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils import timezone
from airflow.utils.net import get_hostname
from airflow.utils.session import create_session
from airflow.utils.state import State
from tests.test_utils.db import clear_db_runs
from tests.test_utils.mock_executor import MockExecutor

DEFAULT_DATE = timezone.datetime(2016, 1, 1)
TEST_DAG_FOLDER = os.environ['AIRFLOW__CORE__DAGS_FOLDER']


class TestLocalTaskJob(unittest.TestCase):
    def setUp(self):
        clear_db_runs()
        patcher = patch('airflow.jobs.base_job.sleep')
        self.addCleanup(patcher.stop)
        self.mock_base_job_sleep = patcher.start()

    def create_ti_with_job(self, dag_id, task_id, state, session):
        dag = DAG(
            dag_id,
            start_date=DEFAULT_DATE
        )
        with dag:
            DummyOperator(task_id=task_id)

        dag.clear()
        dr = dag.create_dagrun(run_id='test',
                               state=State.SUCCESS,
                               execution_date=DEFAULT_DATE,
                               start_date=DEFAULT_DATE)
        ti = dr.get_task_instance(task_id=task_id)
        job = LocalTaskJob(task_instance=ti,
                           executor=SequentialExecutor())
        ti.state = state
        job.state = state
        session.merge(ti)
        session.add(job)
        session.commit()
        ti.job_id = job.id
        session.merge(ti)
        session.commit()
        return (ti, job)

    def test_localtaskjob_essential_attr(self):
        """
        Check whether essential attributes
        of LocalTaskJob can be assigned with
        proper values without intervention
        """
        dag = DAG(
            'test_localtaskjob_essential_attr',
            start_date=DEFAULT_DATE,
            default_args={'owner': 'owner1'})

        with dag:
            op1 = DummyOperator(task_id='op1')

        dag.clear()
        dr = dag.create_dagrun(run_id="test",
                               state=State.SUCCESS,
                               execution_date=DEFAULT_DATE,
                               start_date=DEFAULT_DATE)
        ti = dr.get_task_instance(task_id=op1.task_id)

        job1 = LocalTaskJob(task_instance=ti,
                            ignore_ti_state=True,
                            executor=SequentialExecutor())

        essential_attr = ["dag_id", "job_type", "start_date", "hostname"]

        check_result_1 = [hasattr(job1, attr) for attr in essential_attr]
        self.assertTrue(all(check_result_1))

        check_result_2 = [getattr(job1, attr) is not None for attr in essential_attr]
        self.assertTrue(all(check_result_2))

    @patch('os.getpid')
    def test_localtaskjob_heartbeat(self, mock_pid):
        session = settings.Session()
        dag = DAG(
            'test_localtaskjob_heartbeat',
            start_date=DEFAULT_DATE,
            default_args={'owner': 'owner1'})

        with dag:
            op1 = DummyOperator(task_id='op1')

        dag.clear()
        dr = dag.create_dagrun(run_id="test",
                               state=State.SUCCESS,
                               execution_date=DEFAULT_DATE,
                               start_date=DEFAULT_DATE,
                               session=session)
        ti = dr.get_task_instance(task_id=op1.task_id, session=session)
        ti.state = State.RUNNING
        ti.hostname = "blablabla"
        session.commit()

        job1 = LocalTaskJob(task_instance=ti,
                            ignore_ti_state=True,
                            executor=SequentialExecutor())
        self.assertRaises(AirflowException, job1.heartbeat_callback)

        mock_pid.return_value = 1
        ti.state = State.RUNNING
        ti.hostname = get_hostname()
        ti.pid = 1
        session.merge(ti)
        session.commit()

        job1.heartbeat_callback(session=None)

        mock_pid.return_value = 2
        self.assertRaises(AirflowException, job1.heartbeat_callback)

    @patch('os.getpid')
    def test_heartbeat_failed_fast(self, mock_getpid):
        """
        Test that task heartbeat will sleep when it fails fast
        """
        mock_getpid.return_value = 1
        self.mock_base_job_sleep.side_effect = time.sleep

        with create_session() as session:
            dagbag = DagBag(
                dag_folder=TEST_DAG_FOLDER,
                include_examples=False,
            )
            dag_id = 'test_heartbeat_failed_fast'
            task_id = 'test_heartbeat_failed_fast_op'
            dag = dagbag.get_dag(dag_id)
            task = dag.get_task(task_id)

            dag.create_dagrun(run_id="test_heartbeat_failed_fast_run",
                              state=State.RUNNING,
                              execution_date=DEFAULT_DATE,
                              start_date=DEFAULT_DATE,
                              session=session)
            ti = TaskInstance(task=task, execution_date=DEFAULT_DATE)
            ti.refresh_from_db()
            ti.state = State.RUNNING
            ti.hostname = get_hostname()
            ti.pid = 1
            session.commit()

            job = LocalTaskJob(task_instance=ti, executor=MockExecutor(do_update=False))
            job.heartrate = 2
            heartbeat_records = []
            job.heartbeat_callback = lambda session: heartbeat_records.append(job.get_heartbeat())
            job._execute()
            self.assertGreater(len(heartbeat_records), 2)
            for i in range(1, len(heartbeat_records)):
                time1 = heartbeat_records[i - 1]
                time2 = heartbeat_records[i]
                # Assert that difference small enough
                delta = (time2 - time1).total_seconds()
                self.assertAlmostEqual(delta, job.heartrate, delta=0.05)

    @pytest.mark.xfail(condition=True, reason="This test might be flaky in postgres/mysql")
    def test_mark_success_no_kill(self):
        """
        Test that ensures that mark_success in the UI doesn't cause
        the task to fail, and that the task exits
        """
        dagbag = DagBag(
            dag_folder=TEST_DAG_FOLDER,
            include_examples=False,
        )
        dag = dagbag.dags.get('test_mark_success')
        task = dag.get_task('task1')

        session = settings.Session()

        dag.clear()
        dag.create_dagrun(run_id="test",
                          state=State.RUNNING,
                          execution_date=DEFAULT_DATE,
                          start_date=DEFAULT_DATE,
                          session=session)
        ti = TaskInstance(task=task, execution_date=DEFAULT_DATE)
        ti.refresh_from_db()
        job1 = LocalTaskJob(task_instance=ti, ignore_ti_state=True)
        process = multiprocessing.Process(target=job1.run)
        process.start()
        ti.refresh_from_db()
        for _ in range(0, 50):
            if ti.state == State.RUNNING:
                break
            time.sleep(0.1)
            ti.refresh_from_db()
        self.assertEqual(State.RUNNING, ti.state)
        ti.state = State.SUCCESS
        session.merge(ti)
        session.commit()

        process.join(timeout=10)
        self.assertFalse(process.is_alive())
        ti.refresh_from_db()
        self.assertEqual(State.SUCCESS, ti.state)

    def test_localtaskjob_double_trigger(self):
        dagbag = DagBag(
            dag_folder=TEST_DAG_FOLDER,
            include_examples=False,
        )
        dag = dagbag.dags.get('test_localtaskjob_double_trigger')
        task = dag.get_task('test_localtaskjob_double_trigger_task')

        session = settings.Session()

        dag.clear()
        dr = dag.create_dagrun(run_id="test",
                               state=State.SUCCESS,
                               execution_date=DEFAULT_DATE,
                               start_date=DEFAULT_DATE,
                               session=session)
        ti = dr.get_task_instance(task_id=task.task_id, session=session)
        ti.state = State.RUNNING
        ti.hostname = get_hostname()
        ti.pid = 1
        session.merge(ti)
        session.commit()

        ti_run = TaskInstance(task=task, execution_date=DEFAULT_DATE)
        ti_run.refresh_from_db()
        job1 = LocalTaskJob(task_instance=ti_run,
                            executor=SequentialExecutor())
        from airflow.task.task_runner.standard_task_runner import StandardTaskRunner
        with patch.object(StandardTaskRunner, 'start', return_value=None) as mock_method:
            job1.run()
            mock_method.assert_not_called()

        ti = dr.get_task_instance(task_id=task.task_id, session=session)
        self.assertEqual(ti.pid, 1)
        self.assertEqual(ti.state, State.RUNNING)

        session.close()

    def test_localtaskjob_maintain_heart_rate(self):
        dagbag = DagBag(
            dag_folder=TEST_DAG_FOLDER,
            include_examples=False,
        )
        dag = dagbag.dags.get('test_localtaskjob_double_trigger')
        task = dag.get_task('test_localtaskjob_double_trigger_task')

        session = settings.Session()

        dag.clear()
        dag.create_dagrun(run_id="test",
                          state=State.SUCCESS,
                          execution_date=DEFAULT_DATE,
                          start_date=DEFAULT_DATE,
                          session=session)

        ti_run = TaskInstance(task=task, execution_date=DEFAULT_DATE)
        ti_run.refresh_from_db()
        job1 = LocalTaskJob(task_instance=ti_run,
                            executor=SequentialExecutor())

        # this should make sure we only heartbeat once and exit at the second
        # loop in _execute()
        return_codes = [None, 0]

        def multi_return_code():
            return return_codes.pop(0)

        time_start = time.time()
        from airflow.task.task_runner.standard_task_runner import StandardTaskRunner
        with patch.object(StandardTaskRunner, 'start', return_value=None) as mock_start:
            with patch.object(StandardTaskRunner, 'return_code') as mock_ret_code:
                mock_ret_code.side_effect = multi_return_code
                job1.run()
                self.assertEqual(mock_start.call_count, 1)
                self.assertEqual(mock_ret_code.call_count, 2)
        time_end = time.time()

        self.assertEqual(self.mock_base_job_sleep.call_count, 1)
        self.assertEqual(job1.state, State.SUCCESS)

        # Consider we have patched sleep call, it should not be sleeping to
        # keep up with the heart rate in other unpatched places
        #
        # We already make sure patched sleep call is only called once
        self.assertLess(time_end - time_start, job1.heartrate)
        session.close()

    def test_mark_failure_on_failure_callback(self):
        """
        Test that ensures that mark_failure in the UI fails
        the task, and executes on_failure_callback
        """
        data = {'called': False}

        def check_failure(context):
            self.assertEqual(context['dag_run'].dag_id,
                             'test_mark_failure')
            data['called'] = True

        dag = DAG(dag_id='test_mark_failure',
                  start_date=DEFAULT_DATE,
                  default_args={'owner': 'owner1'})

        task = DummyOperator(
            task_id='test_state_succeeded1',
            dag=dag,
            on_failure_callback=check_failure)

        session = settings.Session()

        dag.clear()
        dag.create_dagrun(run_id="test",
                          state=State.RUNNING,
                          execution_date=DEFAULT_DATE,
                          start_date=DEFAULT_DATE,
                          session=session)
        ti = TaskInstance(task=task, execution_date=DEFAULT_DATE)
        ti.refresh_from_db()
        job1 = LocalTaskJob(task_instance=ti,
                            ignore_ti_state=True,
                            executor=SequentialExecutor())
        from airflow.task.task_runner.standard_task_runner import StandardTaskRunner
        job1.task_runner = StandardTaskRunner(job1)
        process = multiprocessing.Process(target=job1.run)
        process.start()
        ti.refresh_from_db()
        for _ in range(0, 50):
            if ti.state == State.RUNNING:
                break
            time.sleep(0.1)
            ti.refresh_from_db()
        self.assertEqual(State.RUNNING, ti.state)
        ti.state = State.FAILED
        session.merge(ti)
        session.commit()

        job1.heartbeat_callback(session=None)
        self.assertTrue(data['called'])
        process.join(timeout=10)
        self.assertFalse(process.is_alive())

    def test_mark_success_on_success_callback(self):
        """
        Test that ensures that where a task is marked suceess in the UI
        on_success_callback gets executed
        """
        data = {'called': False}

        def success_callback(context):
            self.assertEqual(context['dag_run'].dag_id,
                             'test_mark_success')
            data['called'] = True

        dag = DAG(dag_id='test_mark_success',
                  start_date=DEFAULT_DATE,
                  default_args={'owner': 'owner1'})

        task = DummyOperator(
            task_id='test_state_succeeded1',
            dag=dag,
            on_success_callback=success_callback)

        session = settings.Session()

        dag.clear()
        dag.create_dagrun(run_id="test",
                          state=State.RUNNING,
                          execution_date=DEFAULT_DATE,
                          start_date=DEFAULT_DATE,
                          session=session)
        ti = TaskInstance(task=task, execution_date=DEFAULT_DATE)
        ti.refresh_from_db()
        job1 = LocalTaskJob(task_instance=ti,
                            ignore_ti_state=True,
                            executor=SequentialExecutor())
        from airflow.task.task_runner.standard_task_runner import StandardTaskRunner
        job1.task_runner = StandardTaskRunner(job1)
        process = multiprocessing.Process(target=job1.run)
        process.start()
        ti.refresh_from_db()
        for _ in range(0, 50):
            if ti.state == State.RUNNING:
                break
            time.sleep(0.1)
            ti.refresh_from_db()
        self.assertEqual(State.RUNNING, ti.state)
        ti.state = State.SUCCESS
        session.merge(ti)
        session.commit()

        job1.heartbeat_callback(session=None)
        self.assertTrue(data['called'])
        process.join(timeout=10)
        self.assertFalse(process.is_alive())

    def test_get_zombie_running_tis(self):
        # cases
        # heartbefore expiration
        # heartbeat after expiration
        # job state not in running
        # non running TI
        limit_dttm = timezone.utcnow()
        before_dttm = limit_dttm - timedelta(seconds=5)
        after_dttm = limit_dttm + timedelta(seconds=5)

        expected = set()

        with create_session() as session:
            test_name = 'test_get_zombie_running_tis_redis_'

            ti1, job1 = self.create_ti_with_job(test_name + '1', 'task', State.RUNNING, session)
            _, job2 = self.create_ti_with_job(test_name + '2', 'task', State.RUNNING, session)
            ti3, job3 = self.create_ti_with_job(test_name + '3', 'task', State.RUNNING, session)
            _ = self.create_ti_with_job(test_name + '6', 'task', State.FAILED, session)
            session.commit()
            job1._legacy_heartbeat = before_dttm
            job2._legacy_heartbeat = after_dttm
            job3._legacy_heartbeat = before_dttm
            job3.state = State.FAILED

            expected.add(SimpleTaskInstance(ti1).key)
            expected.add(SimpleTaskInstance(ti3).key)
            session.merge(job1)
            session.merge(job2)
            session.merge(job3)
            session.commit()

            res = LocalTaskJob.get_zombie_running_tis(limit_dttm, session=session)
            res_as_set = {sti.key for sti in res}
            self.assertEqual(expected, res_as_set)

    def test_get_zombie_running_tis_redis(self):
        # cases
        # no heartbeat before expiration
        # no heartbeat after expiration
        # heartbefore expiration
        # heartbeat after expiration
        # job state not in running
        # non running TI
        limit_dttm = timezone.utcnow()
        before_dttm = limit_dttm - timedelta(seconds=5)
        after_dttm = limit_dttm + timedelta(seconds=5)

        expected = set()

        with create_session() as session:
            test_name = 'test_get_zombie_running_tis_redis_'

            ti1, job1 = self.create_ti_with_job(test_name + '1', 'task', State.RUNNING, session)
            _, job2 = self.create_ti_with_job(test_name + '2', 'task', State.RUNNING, session)
            ti3, job3 = self.create_ti_with_job(test_name + '3', 'task', State.RUNNING, session)
            _, job4 = self.create_ti_with_job(test_name + '4', 'task', State.RUNNING, session)
            ti5, job5 = self.create_ti_with_job(test_name + '5', 'task', State.RUNNING, session)
            _ = self.create_ti_with_job(test_name + '6', 'task', State.FAILED, session)
            session.commit()
            job1._legacy_heartbeat = before_dttm
            job2._legacy_heartbeat = after_dttm
            job3._legacy_heartbeat = before_dttm
            job4._legacy_heartbeat = before_dttm
            job5._legacy_heartbeat = before_dttm
            job5.state = State.FAILED

            expected.add(SimpleTaskInstance(ti1).key)
            expected.add(SimpleTaskInstance(ti3).key)
            expected.add(SimpleTaskInstance(ti5).key)
            session.merge(job1)
            session.merge(job2)
            session.merge(job3)
            session.merge(job4)
            session.merge(job5)
            session.commit()

            def mock_method(keys, args):
                self.assertEqual([LocalTaskJob.__name__], keys)
                results = []
                for arg in args:
                    if arg in (job1.id, job2.id):
                        dttm = None
                    elif arg == job3.id:
                        dttm = before_dttm
                    elif arg in (job4.id, job5.id):
                        dttm = after_dttm
                    else:
                        # should not happen
                        self.fail('invalid id')
                    results.append(dttm and (dttm - timezone.utc_epoch()).total_seconds())
                return results

            with patch.object(configuration.conf, 'getint', return_value=3):
                with patch.object(configuration.conf, 'getboolean', return_value=True):
                    with patch('airflow.jobs.BaseJob.redis') as mocked_redis:
                        mocked_redis.register_script.return_value = mock_method
                        res = LocalTaskJob.get_zombie_running_tis(limit_dttm, session=session)
                        res_as_set = {sti.key for sti in res}
                        self.assertEqual(expected, res_as_set)
