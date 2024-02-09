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

"""AWS Batch Executor. Each Airflow task gets delegated out to an AWS Batch Job."""
from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List

from botocore.exceptions import ClientError, NoCredentialsError
from marshmallow import ValidationError

from airflow.configuration import conf
from airflow.exceptions import AirflowException
from airflow.executors.base_executor import BaseExecutor
from airflow.providers.amazon.aws.executors.utils.exponential_backoff_retry import exponential_backoff_retry
from airflow.providers.amazon.aws.hooks.batch_client import BatchClientHook
from airflow.utils import timezone

if TYPE_CHECKING:
    from airflow.models.taskinstance import TaskInstanceKey
from airflow.providers.amazon.aws.executors.batch.boto_schema import (
    BatchDescribeJobsResponseSchema,
    BatchSubmitJobResponseSchema,
)
from airflow.providers.amazon.aws.executors.batch.utils import (
    CONFIG_DEFAULTS,
    CONFIG_GROUP_NAME,
    AllBatchConfigKeys,
    BatchExecutorException,
    BatchJob,
    BatchJobCollection,
    BatchQueuedJob,
)
from airflow.utils.state import State

CommandType = List[str]
ExecutorConfigType = Dict[str, Any]

INVALID_CREDENTIALS_EXCEPTIONS = [
    "ExpiredTokenException",
    "InvalidClientTokenId",
    "UnrecognizedClientException",
]


class AwsBatchExecutor(BaseExecutor):
    """
    The Airflow Scheduler creates a shell command, and passes it to the executor.

    This Batch Executor simply runs said airflow command a resource provisioned and managed
    by AWS Batch. It then periodically checks in with the launched jobs (via job-ids) to
    determine the status.
    The `submit_job_kwargs` configuration points to a dictionary that returns a dictionary. The
    keys of the resulting dictionary should match the kwargs for the SubmitJob definition per AWS'
    documentation (see below).
    For maximum flexibility, individual tasks can specify `executor_config` as a dictionary, with keys that
    match the request syntax for the SubmitJob definition per AWS' documentation (see link below). The
    `executor_config` will update the `submit_job_kwargs` dictionary when calling the task. This allows
    individual jobs to specify CPU, memory, GPU, env variables, etc.
    Prerequisite: proper configuration of Boto3 library
    .. seealso:: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html for
    authentication and access-key management. You can store an environmental variable, setup aws config from
    console, or use IAM roles.
    .. seealso:: https://docs.aws.amazon.com/batch/latest/APIReference/API_SubmitJob.html for an
    Airflow TaskInstance's executor_config.
    """

    # AWS only allows a maximum number of JOBs in the describe_jobs function
    DESCRIBE_JOBS_BATCH_SIZE = 99

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_workers = BatchJobCollection()
        self.pending_jobs: deque = deque()
        self.attempts_since_last_successful_connection = 0
        self.load_batch_connection(check_connection=False)
        self.IS_BOTO_CONNECTION_HEALTHY = False
        self.submit_job_kwargs = self._load_submit_kwargs()

    def check_health(self):
        """Make a test API call to check the health of the Batch Executor."""
        success_status = "succeeded."
        status = success_status

        try:
            invalid_job_id = "a" * 32
            self.batch.describe_jobs(jobs=[invalid_job_id])
            # If an empty response was received, then that is considered to be the success case.
        except ClientError as ex:
            error_code = ex.response["Error"]["Code"]
            error_message = ex.response["Error"]["Message"]
            status = f"failed because: {error_code}: {error_message}. "
        except Exception as e:
            # Any non-ClientError exceptions. This can include Botocore exceptions for example
            status = f"failed because: {e}. "
        finally:
            msg_prefix = "Batch Executor health check has %s"
            if status == success_status:
                self.IS_BOTO_CONNECTION_HEALTHY = True
                self.log.info(msg_prefix, status)
            else:
                msg_error_suffix = (
                    "The Batch executor will not be able to run Airflow tasks until the issue is addressed."
                )
                raise AirflowException(msg_prefix % status + msg_error_suffix)

    def start(self):
        """Call this when the Executor is run for the first time by the scheduler."""
        check_health = conf.getboolean(
            CONFIG_GROUP_NAME, AllBatchConfigKeys.CHECK_HEALTH_ON_STARTUP, fallback=False
        )

        if not check_health:
            return

        self.log.info("Starting Batch Executor and determining health...")
        try:
            self.check_health()
        except AirflowException:
            self.log.error("Stopping the Airflow Scheduler from starting until the issue is resolved.")
            raise

    def load_batch_connection(self, check_connection: bool = True):
        self.log.info("Loading Connection information")
        aws_conn_id = conf.get(
            CONFIG_GROUP_NAME,
            AllBatchConfigKeys.AWS_CONN_ID,
            fallback=CONFIG_DEFAULTS[AllBatchConfigKeys.AWS_CONN_ID],
        )
        region_name = conf.get(CONFIG_GROUP_NAME, AllBatchConfigKeys.REGION_NAME)
        self.batch = BatchClientHook(aws_conn_id=aws_conn_id, region_name=region_name).conn
        self.attempts_since_last_successful_connection += 1
        self.last_connection_reload = timezone.utcnow()

        if check_connection:
            self.check_health()
            self.attempts_since_last_successful_connection = 0

    def sync(self):
        """Sync will get called periodically by the heartbeat method in the scheduler."""
        if not self.IS_BOTO_CONNECTION_HEALTHY:
            exponential_backoff_retry(
                self.last_connection_reload,
                self.attempts_since_last_successful_connection,
                self.load_batch_connection,
            )
            if not self.IS_BOTO_CONNECTION_HEALTHY:
                return
        try:
            self.sync_running_jobs()
            self.attempt_submit_jobs()
        except (ClientError, NoCredentialsError) as error:
            error_code = error.response["Error"]["Code"]
            if error_code in INVALID_CREDENTIALS_EXCEPTIONS:
                self.IS_BOTO_CONNECTION_HEALTHY = False
                self.log.warning(
                    f"AWS credentials are either missing or expired: {error}.\nRetrying connection"
                )
        except Exception:
            # We catch any and all exceptions because otherwise they would bubble
            # up and kill the scheduler process
            self.log.exception("Failed to sync %s", self.__class__.__name__)

    def sync_running_jobs(self):
        all_job_ids = self.active_workers.get_all_jobs()
        if not all_job_ids:
            self.log.debug("No active Airflow tasks, skipping sync")
            return
        describe_job_response = self._describe_jobs(all_job_ids)

        self.log.debug("Active Workers: %s", describe_job_response)

        for job in describe_job_response:
            if job.get_job_state() == State.FAILED:
                task_key = self.active_workers.pop_by_id(job.job_id)
                self.fail(task_key)
            elif job.get_job_state() == State.SUCCESS:
                task_key = self.active_workers.pop_by_id(job.job_id)
                self.success(task_key)

    def attempt_submit_jobs(self):
        queue_len = len(self.pending_jobs)
        for _ in range(queue_len):
            self.log.info("attempting job run")
            batch_job = self.pending_jobs.popleft()
            key = batch_job.key
            cmd = batch_job.command
            queue = batch_job.queue
            exec_config = batch_job.executor_config

            submit_job_response = self._submit_job(key, cmd, queue, exec_config or {})
            self.active_workers.add_job(submit_job_response["job_id"], key)

    def _describe_jobs(self, job_ids) -> list[BatchJob]:
        all_jobs = []
        for i in range(0, len(job_ids), self.__class__.DESCRIBE_JOBS_BATCH_SIZE):
            batched_job_ids = job_ids[i : i + self.__class__.DESCRIBE_JOBS_BATCH_SIZE]
            if not batched_job_ids:
                continue
            boto_describe_tasks = self.batch.describe_jobs(jobs=batched_job_ids)
            try:
                describe_tasks_response = BatchDescribeJobsResponseSchema().load(boto_describe_tasks)
            except ValidationError as err:
                self.log.error("Batch DescribeJobs API Response: %s", boto_describe_tasks)
                raise BatchExecutorException(
                    f"DescribeJobs API call does not match expected JSON shape. Are you sure that the correct version of Boto3 is installed? {err}"
                )
            all_jobs.extend(describe_tasks_response["jobs"])
        return all_jobs

    def execute_async(self, key: TaskInstanceKey, command: CommandType, queue=None, executor_config=None):
        """Save the task to be executed in the next sync using Boto3's RunTask API."""
        if executor_config and "command" in executor_config:
            raise ValueError('Executor Config should never override "command"')

        self.pending_jobs.append(
            BatchQueuedJob(key=key, command=command, queue=queue, executor_config=executor_config)
        )

    def _submit_job(
        self, key: TaskInstanceKey, cmd: CommandType, queue: str, exec_config: ExecutorConfigType
    ) -> str:
        """
        Override the submit_job_kwargs, and calls the boto3 API submit_job endpoint.

        The command and executor config will be placed in the container-override section of the JSON request,
        before calling Boto3's "submit_job" function.
        """
        submit_job_api = self._submit_job_kwargs(key, cmd, queue, exec_config)
        self.log.info("submitting job with these args %s", submit_job_api)

        boto_run_task = self.batch.submit_job(**submit_job_api)
        submit_job_response = BatchSubmitJobResponseSchema().load(boto_run_task)
        return submit_job_response

    def _submit_job_kwargs(
        self, key: TaskInstanceKey, cmd: CommandType, queue: str, exec_config: ExecutorConfigType
    ) -> dict:
        """
        Override the Airflow command to update the container overrides so kwargs are specific to this task.

        One last chance to modify Boto3's "submit_job" kwarg params before it gets passed into the Boto3
        client. For the latest kwarg parameters:
        .. seealso:: https://docs.aws.amazon.com/batch/latest/APIReference/API_SubmitJob.html
        """
        submit_job_api = deepcopy(self.submit_job_kwargs)
        submit_job_api["containerOverrides"].update(exec_config)
        submit_job_api["containerOverrides"]["command"] = cmd
        return submit_job_api

    def end(self, heartbeat_interval=10):
        """Wait for all currently running tasks to end and prevent any new jobs from running."""
        try:
            while True:
                self.sync()
                if not self.active_workers:
                    break
                time.sleep(heartbeat_interval)
        except Exception:
            # We catch any and all exceptions because otherwise they would bubble
            # up and kill the scheduler process.
            self.log.exception("Failed to end %s", self.__class__.__name__)

    def terminate(self):
        """Kill all Batch Jobs by calling Boto3's TerminateJob API."""
        try:
            for job_id in self.active_workers.get_all_jobs():
                self.batch.terminate_job(jobId=job_id, reason="Airflow Executor received a SIGTERM")
            self.end()
        except Exception:
            # We catch any and all exceptions because otherwise they would bubble
            # up and kill the scheduler process.
            self.log.exception("Failed to terminate %s", self.__class__.__name__)

    @staticmethod
    def _load_submit_kwargs() -> dict:
        from airflow.providers.amazon.aws.executors.batch.batch_executor_config import build_submit_kwargs

        submit_kwargs = build_submit_kwargs()
        # Some checks with some helpful errors
        assert isinstance(submit_kwargs, dict)

        if "containerOverrides" not in submit_kwargs or "command" not in submit_kwargs["containerOverrides"]:
            raise KeyError(
                'SubmitJob API needs kwargs["containerOverrides"]["command"] field,'
                " and value should be NULL or empty."
            )
        return submit_kwargs
