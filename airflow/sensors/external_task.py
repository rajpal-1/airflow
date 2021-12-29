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

import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Collection, FrozenSet, Iterable, List, Optional, Union

from airflow.exceptions import AirflowException
from airflow.models import BaseOperatorLink, DagBag, DagModel
from airflow.operators.dummy import DummyOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.settings import SASession
from airflow.triggers.external_task import ExternalTaskMixin, ExternalTaskTrigger
from airflow.utils.context import Context
from airflow.utils.helpers import build_airflow_url_with_query
from airflow.utils.session import provide_session


class ExternalTaskSensorLink(BaseOperatorLink):
    """
    Operator link for ExternalTaskSensor. It allows users to access
    DAG waited with ExternalTaskSensor.
    """

    name = 'External DAG'

    def get_link(self, operator, dttm):
        query = {"dag_id": operator.external_dag_id, "execution_date": dttm.isoformat()}
        return build_airflow_url_with_query(query)


class ExternalTaskSensor(BaseSensorOperator, ExternalTaskMixin):
    """
    Waits for a different DAG or a task in a different DAG to complete for a
    specific logical date.

    :param external_dag_id: The dag_id that contains the task you want to
        wait for
    :type external_dag_id: str
    :param external_task_id: The task_id that contains the task you want to
        wait for. If ``None`` (default value) the sensor waits for the DAG
    :type external_task_id: str or None
    :param external_task_ids: The list of task_ids that you want to wait for.
        If ``None`` (default value) the sensor waits for the DAG. Either
        external_task_id or external_task_ids can be passed to
        ExternalTaskSensor, but not both.
    :type external_task_ids: Iterable of task_ids or None, default is None
    :param allowed_states: Iterable of allowed states, default is ``['success']``
    :type allowed_states: Iterable
    :param failed_states: Iterable of failed or dis-allowed states, default is ``None``
    :type failed_states: Iterable
    :param execution_delta: time difference with the previous execution to
        look at, the default is the same logical date as the current task or DAG.
        For yesterday, use [positive!] datetime.timedelta(days=1). Either
        execution_delta or execution_date_fn can be passed to
        ExternalTaskSensor, but not both.
    :type execution_delta: Optional[datetime.timedelta]
    :param execution_date_fn: function that receives the current execution's logical date as the first
        positional argument and optionally any number of keyword arguments available in the
        context dictionary, and returns the desired logical dates to query.
        Either execution_delta or execution_date_fn can be passed to ExternalTaskSensor,
        but not both.
    :type execution_date_fn: Optional[Callable]
    :param check_existence: Set to `True` to check if the external task exists (when
        external_task_id is not None) or check if the DAG to wait for exists (when
        external_task_id is None), and immediately cease waiting if the external task
        or DAG does not exist (default value: False).
    :type check_existence: bool
    """

    template_fields = ['external_dag_id', 'external_task_id']
    ui_color = '#19647e'

    @property
    def operator_extra_links(self):
        """Return operator extra links"""
        return [ExternalTaskSensorLink()]

    def __init__(
        self,
        *,
        external_dag_id: str,
        external_task_id: Optional[str] = None,
        external_task_ids: Optional[Collection[str]] = None,
        allowed_states: Optional[Iterable[str]] = None,
        failed_states: Optional[Iterable[str]] = None,
        execution_delta: Optional[timedelta] = None,
        execution_date_fn: Optional[Callable] = None,
        check_existence: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if external_task_id is not None and external_task_ids is not None:
            raise ValueError(
                'Only one of `external_task_id` or `external_task_ids` may '
                'be provided to ExternalTaskSensor; not both.'
            )
        if external_task_id is not None:
            external_task_ids = [external_task_id]
        self.external_task_id = external_task_id
        super()._set_external_task(
            external_dag_id=external_dag_id,
            external_task_ids=external_task_ids,
            allowed_states=allowed_states,
            failed_states=failed_states,
        )

        if execution_delta is not None and execution_date_fn is not None:
            raise ValueError(
                'Only one of `execution_delta` or `execution_date_fn` may '
                'be provided to ExternalTaskSensor; not both.'
            )
        self.execution_delta = execution_delta
        self.execution_date_fn = execution_date_fn
        self.check_existence = check_existence
        self._has_checked_existence = False

    def poke(self, context: Context):
        # In poke mode this will check dag existence only once
        if self.check_existence and not self._has_checked_existence:
            self._check_for_existence()

        dttm = self._get_dttm(context)
        return self._check_for_states(dttm=dttm)

    @provide_session
    def _check_for_existence(self, session: SASession) -> None:
        dag_to_wait = session.query(DagModel).filter(DagModel.dag_id == self.external_dag_id).first()

        if not dag_to_wait:
            raise AirflowException(f'The external DAG {self.external_dag_id} does not exist.')

        if not os.path.exists(dag_to_wait.fileloc):
            raise AirflowException(f'The external DAG {self.external_dag_id} was deleted.')

        if self.external_task_ids:
            refreshed_dag_info = DagBag(dag_to_wait.fileloc).get_dag(self.external_dag_id)
            for external_task_id in self.external_task_ids:
                if not refreshed_dag_info.has_task(external_task_id):
                    raise AirflowException(
                        f'The external task {external_task_id} in '
                        f'DAG {self.external_dag_id} does not exist.'
                    )
        self._has_checked_existence = True

    def _get_dttm(self, context: Context) -> List[datetime]:
        if self.execution_delta:
            dttm = context['logical_date'] - self.execution_delta
        elif self.execution_date_fn:
            dttm = self._handle_execution_date_fn(context=context)
        else:
            dttm = context['logical_date']
        dttm = dttm if isinstance(dttm, list) else [dttm]
        return dttm

    def _handle_execution_date_fn(self, context) -> Any:
        """
        This function is to handle backwards compatibility with how this operator was
        previously where it only passes the execution date, but also allow for the newer
        implementation to pass all context variables as keyword arguments, to allow
        for more sophisticated returns of dates to return.
        """
        from airflow.utils.operator_helpers import make_kwargs_callable

        # Remove "logical_date" because it is already a mandatory positional argument
        logical_date = context["logical_date"]
        kwargs = {k: v for k, v in context.items() if k not in {"execution_date", "logical_date"}}
        # Add "context" in the kwargs for backward compatibility (because context used to be
        # an acceptable argument of execution_date_fn)
        kwargs["context"] = context
        if TYPE_CHECKING:
            assert self.execution_date_fn is not None
        kwargs_callable = make_kwargs_callable(self.execution_date_fn)
        return kwargs_callable(logical_date, **kwargs)


class ExternalTaskSensorAsync(ExternalTaskSensor):
    """
    Waits for a different DAG or a task in a different DAG to complete for a specific logical date,
    deferring itself to avoid taking up a worker slot while it is waiting.

    It is a drop-in replacement for ExternalTaskSensor.
    """

    def execute(self, context: Context) -> Any:
        if self.check_existence and not self._has_checked_existence:
            self._check_for_existence()

        self.defer(
            trigger=ExternalTaskTrigger(
                external_dag_id=self.external_dag_id,
                external_task_ids=self.external_task_ids,
                dttm=self._get_dttm(context),
                allowed_states=self.allowed_states,
                failed_states=self.failed_states,
                poke_interval=self.poke_interval,
            ),
            method_name="execute_complete",
            timeout=timedelta(seconds=self.timeout),
        )

    def execute_complete(self, context: Context, event=None) -> None:
        """Callback for when the trigger fires - returns immediately."""
        return None


class ExternalTaskMarker(DummyOperator):
    """
    Use this operator to indicate that a task on a different DAG depends on this task.
    When this task is cleared with "Recursive" selected, Airflow will clear the task on
    the other DAG and its downstream tasks recursively. Transitive dependencies are followed
    until the recursion_depth is reached.

    :param external_dag_id: The dag_id that contains the dependent task that needs to be cleared.
    :type external_dag_id: str
    :param external_task_id: The task_id of the dependent task that needs to be cleared.
    :type external_task_id: str
    :param execution_date: The logical date of the dependent task execution that needs to be cleared.
    :type execution_date: str or datetime
    :param recursion_depth: The maximum level of transitive dependencies allowed. Default is 10.
        This is mostly used for preventing cyclic dependencies. It is fine to increase
        this number if necessary. However, too many levels of transitive dependencies will make
        it slower to clear tasks in the web UI.
    """

    template_fields = ['external_dag_id', 'external_task_id', 'execution_date']
    ui_color = '#19647e'

    # The _serialized_fields are lazily loaded when get_serialized_fields() method is called
    __serialized_fields: Optional[FrozenSet[str]] = None

    def __init__(
        self,
        *,
        external_dag_id: str,
        external_task_id: str,
        execution_date: Optional[Union[str, datetime]] = "{{ logical_date.isoformat() }}",
        recursion_depth: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.external_dag_id = external_dag_id
        self.external_task_id = external_task_id
        if isinstance(execution_date, datetime):
            self.execution_date = execution_date.isoformat()
        elif isinstance(execution_date, str):
            self.execution_date = execution_date
        else:
            raise TypeError(
                f'Expected str or datetime.datetime type for execution_date. Got {type(execution_date)}'
            )

        if recursion_depth <= 0:
            raise ValueError("recursion_depth should be a positive integer")
        self.recursion_depth = recursion_depth

    @classmethod
    def get_serialized_fields(cls):
        """Serialized ExternalTaskMarker contain exactly these fields + templated_fields ."""
        if not cls.__serialized_fields:
            cls.__serialized_fields = frozenset(super().get_serialized_fields() | {"recursion_depth"})
        return cls.__serialized_fields
