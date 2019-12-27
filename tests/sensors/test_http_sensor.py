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
from unittest.mock import patch

import mock
import requests

from airflow import DAG
from airflow.exceptions import AirflowException, AirflowSensorTimeout
from airflow.models import TaskInstance
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.sensors.http_sensor import HttpSensor
from airflow.utils.timezone import datetime

DEFAULT_DATE = datetime(2015, 1, 1)
DEFAULT_DATE_ISO = DEFAULT_DATE.isoformat()
TEST_DAG_ID = 'unit_test_dag'


class TestHttpSensor(unittest.TestCase):
    def setUp(self):
        args = {
            'owner': 'airflow',
            'start_date': DEFAULT_DATE
        }
        self.dag = DAG(TEST_DAG_ID, default_args=args)

    @patch("airflow.hooks.http_hook.requests.Session.send")
    def test_poke_exception(self, mock_session_send):
        """
        Exception occurs in poke function should not be ignored.
        """
        response = requests.Response()
        response.status_code = 200
        mock_session_send.return_value = response

        def resp_check(_):
            raise AirflowException('AirflowException raised here!')

        task = HttpSensor(
            task_id='http_sensor_poke_exception',
            http_conn_id='http_default',
            endpoint='',
            request_params={},
            response_check=resp_check,
            timeout=5,
            poke_interval=1)
        with self.assertRaisesRegex(AirflowException, 'AirflowException raised here!'):
            task.execute(context={})

    @patch("airflow.hooks.http_hook.requests.Session.send")
    def test_head_method(self, mock_session_send):
        def resp_check(_):
            return True

        task = HttpSensor(
            dag=self.dag,
            task_id='http_sensor_head_method',
            http_conn_id='http_default',
            endpoint='',
            request_params={},
            method='HEAD',
            response_check=resp_check,
            timeout=5,
            poke_interval=1)

        task.execute(context={})

        args, kwargs = mock_session_send.call_args
        received_request = args[0]

        prep_request = requests.Request(
            'HEAD',
            'https://www.httpbin.org',
            {}).prepare()

        self.assertEqual(prep_request.url, received_request.url)
        self.assertTrue(prep_request.method, received_request.method)

    @patch("airflow.hooks.http_hook.requests.Session.send")
    def test_poke_context(self, mock_session_send):
        response = requests.Response()
        response.status_code = 200
        mock_session_send.return_value = response

        def resp_check(_, execution_date):
            if execution_date == DEFAULT_DATE:
                return True
            raise AirflowException('AirflowException raised here!')

        task = HttpSensor(
            task_id='http_sensor_poke_exception',
            http_conn_id='http_default',
            endpoint='',
            request_params={},
            response_check=resp_check,
            timeout=5,
            poke_interval=1,
            dag=self.dag)

        task_instance = TaskInstance(task=task, execution_date=DEFAULT_DATE)
        task.execute(task_instance.get_template_context())

    @patch("airflow.hooks.http_hook.requests.Session.send")
    def test_logging_head_error_request(
        self,
        mock_session_send
    ):
        def resp_check(_):
            return True

        response = requests.Response()
        response.status_code = 404
        response.reason = 'Not Found'
        response._content = b'This endpoint doesnt exist'
        mock_session_send.return_value = response

        task = HttpSensor(
            dag=self.dag,
            task_id='http_sensor_head_method',
            http_conn_id='http_default',
            endpoint='',
            request_params={},
            method='HEAD',
            response_check=resp_check,
            timeout=5,
            poke_interval=1
        )

        with mock.patch.object(task.hook.log, 'error') as mock_errors:
            with self.assertRaises(AirflowSensorTimeout):
                task.execute(None)

            self.assertTrue(mock_errors.called)
            calls = [
                mock.call('HTTP error: %s', 'Not Found'),
                mock.call('This endpoint doesnt exist'),
                mock.call('HTTP error: %s', 'Not Found'),
                mock.call('This endpoint doesnt exist'),
                mock.call('HTTP error: %s', 'Not Found'),
                mock.call('This endpoint doesnt exist'),
                mock.call('HTTP error: %s', 'Not Found'),
                mock.call('This endpoint doesnt exist'),
                mock.call('HTTP error: %s', 'Not Found'),
                mock.call('This endpoint doesnt exist'),
                mock.call('HTTP error: %s', 'Not Found'),
                mock.call('This endpoint doesnt exist'),
            ]
            mock_errors.assert_has_calls(calls)


class FakeSession:
    def __init__(self):
        self.response = requests.Response()
        self.response.status_code = 200
        self.response._content = 'apache/airflow'.encode('ascii', 'ignore')

    def send(self, *args, **kwargs):
        return self.response

    def prepare_request(self, request):
        if 'date' in request.params:
            self.response._content += (
                '/' + request.params['date']
            ).encode('ascii', 'ignore')
        return self.response


class TestHttpOpSensor(unittest.TestCase):
    def setUp(self):
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE_ISO}
        dag = DAG(TEST_DAG_ID, default_args=args)
        self.dag = dag

    @mock.patch('requests.Session', FakeSession)
    def test_get(self):
        op = SimpleHttpOperator(
            task_id='get_op',
            method='GET',
            endpoint='/search',
            data={"client": "ubuntu", "q": "airflow"},
            headers={},
            dag=self.dag)
        op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)

    @mock.patch('requests.Session', FakeSession)
    def test_get_response_check(self):
        op = SimpleHttpOperator(
            task_id='get_op',
            method='GET',
            endpoint='/search',
            data={"client": "ubuntu", "q": "airflow"},
            response_check=lambda response: ("apache/airflow" in response.text),
            headers={},
            dag=self.dag)
        op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)

    @mock.patch('requests.Session', FakeSession)
    def test_sensor(self):
        sensor = HttpSensor(
            task_id='http_sensor_check',
            http_conn_id='http_default',
            endpoint='/search',
            request_params={"client": "ubuntu", "q": "airflow", 'date': '{{ds}}'},
            headers={},
            response_check=lambda response: (
                "apache/airflow/" + DEFAULT_DATE.strftime('%Y-%m-%d')
                in response.text),
            poke_interval=5,
            timeout=15,
            dag=self.dag)
        sensor.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)
