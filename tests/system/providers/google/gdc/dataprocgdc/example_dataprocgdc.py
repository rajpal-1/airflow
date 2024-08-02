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
Example Airflow DAG that shows how to create and monitor a Spark Application or App Environment
 in a DPGDC cluster
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.google.gdc.operators.dataprocgdc import (
    DataprocGdcCreateAppEnvironmentKrmOperator,
    DataprocGDCSubmitSparkJobKrmOperator,
)
from airflow.providers.google.gdc.sensors.dataprocgdc import DataprocGDCKrmSensor

DAG_ID = "example_dag_dpgdc-krm"

with DAG(
    DAG_ID, schedule="@once", catchup=False, start_date=datetime(2024, 2, 5), tags=["example", "dataprocgdc"]
) as dag:
    # [START howto_operator_dataprocGDC_submit_spark]
    submit_spark_job_operator = DataprocGDCSubmitSparkJobKrmOperator(
        task_id="example-dataprocgdc-submitspark-operator",
        application_file="sparkpi.yaml",
        namespace="default",
        kubernetes_conn_id="myk8s",
    )
    # [END howto_operator_dataprocGDC_submit_spark]

    # [START howto_operator_dataprocGDC_create_appenv]
    create_app_env_operator = DataprocGdcCreateAppEnvironmentKrmOperator(
        task_id="example-dataprocgdc-appenv-operator",
        application_file="appEnv.yaml",
        namespace="default",
        kubernetes_conn_id="myk8s",
    )
    # [END howto_operator_dataprocGDC_create_appenv]

    # [START howto_sensor_dataprocGDC_spark]
    sensor = DataprocGDCKrmSensor(
        task_id="example-dataprocgdc-sensor",
        application_name="simple-spark",
        attach_log=True,
        namespace="default",
        kubernetes_conn_id="myk8s",
    )
    # [END howto_sensor_dataprocGDC_spark]

from tests.system.utils import get_test_run  # noqa: E402

# Needed to run the example DAG with pytest (see: tests/system/README.md#run_via_pytest)
test_run = get_test_run(dag)
