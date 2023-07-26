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
from __future__ import annotations

from unittest import mock

import pytest
from google.cloud.run_v2 import (
    CreateJobRequest,
    GetJobRequest,
    Job,
    ListJobsRequest,
    RunJobRequest,
    UpdateJobRequest,
)

from airflow.exceptions import AirflowException
from airflow.providers.google.cloud.hooks.cloud_run import CloudRunAsyncHook, CloudRunHook
from tests.providers.google.cloud.utils.base_gcp_mock import mock_base_gcp_hook_default_project_id


class TestCloudBathHook:
    @mock.patch(
        "airflow.providers.google.common.hooks.base_google.GoogleBaseHook.__init__",
        new=mock_base_gcp_hook_default_project_id,
    )
    @mock.patch("airflow.providers.google.cloud.hooks.cloud_run.JobsClient")
    def test_get_job(self, mock_batch_service_client):
        cloud_batch_hook = CloudRunHook()

        job_name = "job1"
        region = "region1"
        project_id = "projectid"

        get_job_request = GetJobRequest(name=f"projects/{project_id}/locations/{region}/jobs/{job_name}")

        cloud_batch_hook.get_job(job_name=job_name, region=region, project_id=project_id)
        cloud_batch_hook._client.get_job.assert_called_once_with(get_job_request)

    @mock.patch(
        "airflow.providers.google.common.hooks.base_google.GoogleBaseHook.__init__",
        new=mock_base_gcp_hook_default_project_id,
    )
    @mock.patch("airflow.providers.google.cloud.hooks.cloud_run.JobsClient")
    def test_update_job(self, mock_batch_service_client):
        cloud_batch_hook = CloudRunHook()

        job_name = "job1"
        region = "region1"
        project_id = "projectid"
        job = Job()
        job.name = job.name = f"projects/{project_id}/locations/{region}/jobs/{job_name}"

        update_request = UpdateJobRequest()
        update_request.job = job

        cloud_batch_hook.update_job(job=job, job_name=job_name, region=region, project_id=project_id)

        cloud_batch_hook._client.update_job.assert_called_once_with(update_request)

    @mock.patch(
        "airflow.providers.google.common.hooks.base_google.GoogleBaseHook.__init__",
        new=mock_base_gcp_hook_default_project_id,
    )
    @mock.patch("airflow.providers.google.cloud.hooks.cloud_run.JobsClient")
    def test_create_job(self, mock_batch_service_client):
        cloud_batch_hook = CloudRunHook()

        job_name = "job1"
        region = "region1"
        project_id = "projectid"
        job = Job()

        create_request = CreateJobRequest()
        create_request.job = job
        create_request.job_id = job_name
        create_request.parent = f"projects/{project_id}/locations/{region}"

        cloud_batch_hook.create_job(job=job, job_name=job_name, region=region, project_id=project_id)

        cloud_batch_hook._client.create_job.assert_called_once_with(create_request)

    @mock.patch(
        "airflow.providers.google.common.hooks.base_google.GoogleBaseHook.__init__",
        new=mock_base_gcp_hook_default_project_id,
    )
    @mock.patch("airflow.providers.google.cloud.hooks.cloud_run.JobsClient")
    def test_execute_job(self, mock_batch_service_client):
        cloud_batch_hook = CloudRunHook()

        job_name = "job1"
        region = "region1"
        project_id = "projectid"
        run_job_request = RunJobRequest(name=f"projects/{project_id}/locations/{region}/jobs/{job_name}")

        cloud_batch_hook.execute_job(job_name=job_name, region=region, project_id=project_id)
        cloud_batch_hook._client.run_job.assert_called_once_with(request=run_job_request)

    @mock.patch(
        "airflow.providers.google.common.hooks.base_google.GoogleBaseHook.__init__",
        new=mock_base_gcp_hook_default_project_id,
    )
    @mock.patch("airflow.providers.google.cloud.hooks.cloud_run.JobsClient")
    def test_list_jobs(self, mock_batch_service_client):

        number_of_jobs = 3
        region = "us-central1"
        project_id = "test_project_id"

        page = self._mock_pager(number_of_jobs)
        mock_batch_service_client.return_value.list_jobs.return_value = page
        cloud_batch_hook = CloudRunHook()

        jobs_list = cloud_batch_hook.list_jobs(region=region, project_id=project_id)

        for i in range(number_of_jobs):
            assert jobs_list[i].name == f"name{i}"

        expected_list_jobs_request: ListJobsRequest = ListJobsRequest(
            parent=f"projects/{project_id}/locations/{region}"
        )
        mock_batch_service_client.return_value.list_jobs.assert_called_once_with(
            request=expected_list_jobs_request
        )

    @mock.patch(
        "airflow.providers.google.common.hooks.base_google.GoogleBaseHook.__init__",
        new=mock_base_gcp_hook_default_project_id,
    )
    @mock.patch("airflow.providers.google.cloud.hooks.cloud_run.JobsClient")
    def test_list_jobs_show_deleted(self, mock_batch_service_client):

        number_of_jobs = 3
        region = "us-central1"
        project_id = "test_project_id"

        page = self._mock_pager(number_of_jobs)
        mock_batch_service_client.return_value.list_jobs.return_value = page
        cloud_batch_hook = CloudRunHook()

        jobs_list = cloud_batch_hook.list_jobs(region=region, project_id=project_id, show_deleted=True)

        for i in range(number_of_jobs):
            assert jobs_list[i].name == f"name{i}"

        expected_list_jobs_request: ListJobsRequest = ListJobsRequest(
            parent=f"projects/{project_id}/locations/{region}", show_deleted=True
        )
        mock_batch_service_client.return_value.list_jobs.assert_called_once_with(
            request=expected_list_jobs_request
        )

    @mock.patch(
        "airflow.providers.google.common.hooks.base_google.GoogleBaseHook.__init__",
        new=mock_base_gcp_hook_default_project_id,
    )
    @mock.patch("airflow.providers.google.cloud.hooks.cloud_run.JobsClient")
    def test_list_jobs_with_limit(self, mock_batch_service_client):

        number_of_jobs = 3
        limit = 2
        region = "us-central1"
        project_id = "test_project_id"

        page = self._mock_pager(number_of_jobs)
        mock_batch_service_client.return_value.list_jobs.return_value = page
        cloud_batch_hook = CloudRunHook()

        jobs_list = cloud_batch_hook.list_jobs(region=region, project_id=project_id, limit=limit)

        assert len(jobs_list) == limit
        for i in range(limit):
            assert jobs_list[i].name == f"name{i}"

    @mock.patch(
        "airflow.providers.google.common.hooks.base_google.GoogleBaseHook.__init__",
        new=mock_base_gcp_hook_default_project_id,
    )
    @mock.patch("airflow.providers.google.cloud.hooks.cloud_run.JobsClient")
    def test_list_jobs_with_limit_greater_then_range(self, mock_batch_service_client):

        number_of_jobs = 3
        limit = 5
        region = "us-central1"
        project_id = "test_project_id"

        page = self._mock_pager(number_of_jobs)
        mock_batch_service_client.return_value.list_jobs.return_value = page
        cloud_batch_hook = CloudRunHook()

        jobs_list = cloud_batch_hook.list_jobs(region=region, project_id=project_id, limit=limit)

        assert len(jobs_list) == number_of_jobs
        for i in range(number_of_jobs):
            assert jobs_list[i].name == f"name{i}"

    @mock.patch(
        "airflow.providers.google.common.hooks.base_google.GoogleBaseHook.__init__",
        new=mock_base_gcp_hook_default_project_id,
    )
    @mock.patch("airflow.providers.google.cloud.hooks.cloud_run.JobsClient")
    def test_list_jobs_with_limit_less_than_zero(self, mock_batch_service_client):

        number_of_jobs = 3
        limit = -1
        region = "us-central1"
        project_id = "test_project_id"

        page = self._mock_pager(number_of_jobs)
        mock_batch_service_client.return_value.list_jobs.return_value = page
        cloud_batch_hook = CloudRunHook()

        with pytest.raises(expected_exception=AirflowException):
            cloud_batch_hook.list_jobs(region=region, project_id=project_id, limit=limit)

    def _mock_pager(self, number_of_jobs):
        mock_pager = []
        for i in range(number_of_jobs):
            mock_pager.append(Job(name=f"name{i}"))

        return mock_pager


class TestCloudRunAsyncHook:
    @pytest.mark.asyncio
    @mock.patch(
        "airflow.providers.google.common.hooks.base_google.GoogleBaseHook.__init__",
        new=mock_base_gcp_hook_default_project_id,
    )
    @mock.patch("airflow.providers.google.cloud.hooks.cloud_run.JobsAsyncClient")
    async def test_get_operation(self, mock_client):
        expected_operation = {"name": "somename"}

        async def _get_operation(name):
            return expected_operation

        operation_name = "operationname"
        mock_client.return_value = mock.MagicMock()
        mock_client.return_value.get_operation = _get_operation
        hook = CloudRunAsyncHook()

        returned_operation = await hook.get_operation(operation_name=operation_name)

        assert returned_operation == expected_operation
