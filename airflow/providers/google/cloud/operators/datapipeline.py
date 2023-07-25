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
"""This module contains Google Data Pipelines operators."""
from __future__ import annotations

from airflow import AirflowException
from airflow.providers.google.cloud.hooks.datapipeline import DEFAULT_DATAPIPELINE_LOCATION, DataPipelineHook
from airflow.providers.google.cloud.operators.cloud_base import GoogleCloudBaseOperator


class CreateDataPipelineOperator(GoogleCloudBaseOperator):
    """
    Creates a new Data Pipelines instance from the Data Pipelines API.

    :param body: The request body (contains instance of Pipeline). See:
        https://cloud.google.com/dataflow/docs/reference/data-pipelines/rest/v1/projects.locations.pipelines/create#request-body
    :param project_id: The ID of the GCP project that owns the job.
    :param location: The location to direct the Data Pipelines instance to (example_dags uses uscentral-1).
    :param gcp_conn_id: The connection ID to connect to the Google Cloud
        Platform.

    Returns the created Data Pipelines instance in JSON representation.
    """

    def __init__(
        self,
        *,
        body: dict,
        project_id: str | None = None,
        location: str = DEFAULT_DATAPIPELINE_LOCATION,
        gcp_conn_id: str = "google_cloud_default",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.body = body
        self.project_id = project_id
        self.location = location
        self.gcp_conn_id = gcp_conn_id
        self.datapipeline_hook: DataPipelineHook | None = None
        self.body["pipelineSources"] = {"airflow": "airflow"}

    def execute(self, context: Context):
        if self.body is None:
            raise AirflowException(
                "Request Body not given; cannot create a Data Pipeline without the Request Body."
            )
        if self.project_id is None:
            raise AirflowException(
                "Project ID not given; cannot create a Data Pipeline without the Project ID."
            )
        if self.location is None:
            raise AirflowException("location not given; cannot create a Data Pipeline without the location.")

        self.datapipeline_hook = DataPipelineHook(gcp_conn_id=self.gcp_conn_id)

        self.data_pipeline = self.datapipeline_hook.create_data_pipeline(
            project_id=self.project_id,
            body=self.body,
            location=self.location,
        )

        if "error" in self.data_pipeline:
            raise AirflowException(self.data_pipeline.get("error").get("message"))

        return self.data_pipeline
