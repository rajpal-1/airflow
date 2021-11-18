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

import unittest

import boto3

from airflow.providers.amazon.aws.operators.redshift_pause_cluster import RedshiftPauseClusterOperator

try:
    from moto import mock_redshift
except ImportError:
    mock_redshift = None


class TestPauseClusterOperator(unittest.TestCase):
    @staticmethod
    def _create_clusters():
        client = boto3.client('redshift', region_name='us-east-1')
        client.create_cluster(
            ClusterIdentifier='test_cluster_to_pause',
            NodeType='dc1.large',
            MasterUsername='admin',
            MasterUserPassword='mock_password',
        )
        client.create_cluster(
            ClusterIdentifier='test_cluster_to_resume',
            NodeType='dc1.large',
            MasterUsername='admin',
            MasterUserPassword='mock_password',
        )
        if not client.describe_clusters()['Clusters']:
            raise ValueError('AWS not properly mocked')

    def test_init(self):
        redshift_operator = RedshiftPauseClusterOperator(
            task_id="task_test",
            cluster_identifier="test_cluster",
            aws_conn_id="aws_conn_test",
            check_interval=3,
        )
        assert redshift_operator.task_id == "task_test"
        assert redshift_operator.cluster_identifier == "test_cluster"
        assert redshift_operator.aws_conn_id == "aws_conn_test"
        assert redshift_operator.check_interval == 3
