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
"""
LocalExecutor

.. seealso::
    For more information on how the LocalExecutor works, take a look at the guide:
    :ref:`executor:LocalExecutor`
"""
import concurrent.futures
import logging
import os
import subprocess
from typing import Any, Dict, Optional, Tuple

from setproctitle import getproctitle, setproctitle

from airflow import settings
from airflow.exceptions import AirflowException
from airflow.executors.base_executor import NOT_STARTED_MESSAGE, PARALLELISM, BaseExecutor, CommandType
from airflow.models.taskinstance import TaskInstanceKey
from airflow.utils.state import State


def _setup():
    # We know we've just started a new process, so lets disconnect from the metadata db now
    settings.engine.pool.dispose()
    settings.engine.dispose()
    setproctitle("airflow worker -- LocalExecutor")


def _execute_work(key: TaskInstanceKey, command: CommandType) -> Tuple[TaskInstanceKey, str]:
    """
    Executes command received and stores result state in queue.

    :param key: the key to identify the task instance
    :param command: the command to execute
    """
    if key is None:
        return

    log = logging.getLogger(__name__)
    log.info("LocalExecutor running %s", command)
    setproctitle(f"airflow worker -- LocalExecutor: {command}")
    try:
        if settings.EXECUTE_TASKS_NEW_PYTHON_INTERPRETER:
            return key, _execute_work_in_subprocess(command)
        else:
            return key, _execute_work_in_fork(command)
    finally:
        setproctitle("airflow worker -- LocalExecutor")


def _execute_work_in_subprocess(command: CommandType) -> str:
    try:
        subprocess.check_call(command, close_fds=True)
        return State.SUCCESS.value
    except subprocess.CalledProcessError as e:
        log = logging.getLogger(__name__)
        log.error("Failed to execute task %s.", str(e))
        return State.FAILED.value


def _execute_work_in_fork(command: CommandType) -> str:
    pid = os.fork()
    if pid:
        # In parent, wait for the child
        pid, ret = os.waitpid(pid, 0)
        return State.SUCCESS if ret == 0 else State.FAILED

    from airflow.sentry import Sentry

    ret = 1
    try:
        import signal

        from airflow.cli.cli_parser import get_parser

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGUSR2, signal.SIG_DFL)

        parser = get_parser()
        # [1:] - remove "airflow" from the start of the command
        args = parser.parse_args(command[1:])
        args.shut_down_logging = False

        setproctitle(f"airflow task supervisor: {command}")

        args.func(args)
        ret = 0
        return State.SUCCESS.value
    except Exception as e:
        log = logging.getLogger(__name__)
        log.exception("Failed to execute task %s.", e)
        return State.FAILED.value
    finally:
        Sentry.flush()
        logging.shutdown()
        os._exit(ret)


class LocalExecutor(BaseExecutor):
    """
    LocalExecutor executes tasks locally in parallel.
    It uses the multiprocessing Python library and queues to parallelize the execution
    of tasks.

    :param parallelism: how many parallel processes are run in the executor
    """

    futures_executor: Optional[concurrent.futures.Executor]
    futures: Dict[concurrent.futures.Future, TaskInstanceKey]

    def __init__(self, parallelism: int = PARALLELISM):
        super().__init__(parallelism=parallelism)
        if self.parallelism < 0:
            raise AirflowException("parallelism must be bigger than or equal to 0")
        self.futures = {}

    def start(self) -> None:
        """Starts the executor"""
        old_proctitle = getproctitle()
        setproctitle("airflow executor -- LocalExecutor")
        setproctitle(old_proctitle)
        # This isn't _truely_ unlimited, but it should be good enough!
        size = 2**30 if self.parallelism == 0 else self.parallelism

        self.futures_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=size,
            initializer=_setup,
        )

    def execute_async(
        self,
        key: TaskInstanceKey,
        command: CommandType,
        queue: Optional[str] = None,
        executor_config: Optional[Any] = None,
    ) -> None:
        """Execute asynchronously."""
        if not self.futures_executor:
            raise AirflowException(NOT_STARTED_MESSAGE)

        self.validate_command(command)

        result = self.futures_executor.submit(_execute_work, key=key, command=command)
        self.futures[result] = key

    def sync(self, timeout=0) -> None:
        """Sync will get called periodically by the heartbeat method."""
        try:
            done, _ = concurrent.futures.wait(self.futures, timeout=timeout)
            for future in done:
                key = self.futures.pop(future)
                exc = future.exception()
                result: Any
                if exc:
                    result = (key, State.FAILED, exc)
                else:
                    result = future.result()
                self.change_state(*result)
        except concurrent.futures.TimeoutError:
            pass

    def end(self) -> None:
        """Ends the executor."""
        if not self.futures_executor:
            raise AirflowException(NOT_STARTED_MESSAGE)
        self.log.info(
            "Shutting down LocalExecutor"
            "; waiting for running tasks to finish.  Signal again if you don't want to wait."
        )
        self.futures_executor.shutdown()
        self.sync(timeout=None)

    def terminate(self):
        """Terminate the executor is not doing anything."""
        if self.futures_executor:
            self.futures_executor.shutdown(wait=False)
