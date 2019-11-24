# -*- coding: utf-8 -*-
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

import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

from airflow import DAG
from airflow.contrib.operators.emr_add_steps_operator import EmrAddStepsOperator
from airflow.models import TaskInstance
from airflow.utils import timezone

DEFAULT_DATE = timezone.datetime(2017, 1, 1)

ADD_STEPS_SUCCESS_RETURN = {
    'ResponseMetadata': {
        'HTTPStatusCode': 200
    },
    'StepIds': ['s-2LH3R5GW3A53T']
}

LIST_CLUSTERS_RETURN = {
    'Clusters':
        [
            {
                'Name': 'test_cluster',
                'Id': 'j-1231231234'
            }
        ]
}


class TestEmrAddStepsOperator(unittest.TestCase):
    # When
    _config = [{
        'Name': 'test_step',
        'ActionOnFailure': 'CONTINUE',
        'HadoopJarStep': {
            'Jar': 'command-runner.jar',
            'Args': [
                '/usr/lib/spark/bin/run-example',
                '{{ macros.ds_add(ds, -1) }}',
                '{{ ds }}'
            ]
        }
    }]

    def setUp(self):
        self.args = {
            'owner': 'airflow',
            'start_date': DEFAULT_DATE
        }

        # Mock out the emr_client (moto has incorrect response)
        self.emr_client_mock = MagicMock()

        # Mock out the emr_client creator
        emr_session_mock = MagicMock()
        emr_session_mock.client.return_value = self.emr_client_mock
        self.boto3_session_mock = MagicMock(return_value=emr_session_mock)

        self.operator = EmrAddStepsOperator(
            task_id='test_task',
            job_flow_id='j-8989898989',
            aws_conn_id='aws_default',
            steps=self._config,
            dag=DAG('test_dag_id', default_args=self.args)
        )

    def test_init(self):
        self.assertEqual(self.operator.job_flow_id, 'j-8989898989')
        self.assertEqual(self.operator.aws_conn_id, 'aws_default')

    def test_render_template(self):
        ti = TaskInstance(self.operator, DEFAULT_DATE)
        ti.render_templates()

        expected_args = [{
            'Name': 'test_step',
            'ActionOnFailure': 'CONTINUE',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': [
                    '/usr/lib/spark/bin/run-example',
                    (DEFAULT_DATE - timedelta(days=1)).strftime("%Y-%m-%d"),
                    DEFAULT_DATE.strftime("%Y-%m-%d"),
                ]
            }
        }]

        self.assertListEqual(self.operator.steps, expected_args)

    def test_execute_returns_step_id(self):
        self.emr_client_mock.add_job_flow_steps.return_value = ADD_STEPS_SUCCESS_RETURN

        with patch('boto3.session.Session', self.boto3_session_mock):
            self.assertEqual(self.operator.execute(None), ['s-2LH3R5GW3A53T'])

    def test_init_with_cluster_name(self):
        self.emr_client_mock.list_clusters.return_value = LIST_CLUSTERS_RETURN

        with patch('boto3.session.Session', self.boto3_session_mock):
            operator = EmrAddStepsOperator(
                task_id='test_task',
                job_flow_name='test_cluster',
                aws_conn_id='aws_default',
                dag=DAG('test_dag_id', default_args=self.args)
            )

            self.assertEqual(operator.job_flow_id, 'j-1231231234')


if __name__ == '__main__':
    unittest.main()
