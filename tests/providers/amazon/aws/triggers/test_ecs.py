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
from unittest.mock import AsyncMock

import pytest
from botocore.exceptions import WaiterError

from airflow import AirflowException
from airflow.providers.amazon.aws.hooks.ecs import EcsHook
from airflow.providers.amazon.aws.hooks.logs import AwsLogsHook
from airflow.providers.amazon.aws.triggers.ecs import ClusterWaiterTrigger, TaskDoneTrigger
from airflow.triggers.base import TriggerEvent


class TestClusterWaiterTrigger:
    @pytest.mark.asyncio
    @mock.patch.object(EcsHook, "async_conn")
    async def test_run_max_attempts(self, client_mock):
        a_mock = mock.MagicMock()
        client_mock.__aenter__.return_value = a_mock
        wait_mock = AsyncMock()
        wait_mock.side_effect = WaiterError("name", "reason", {"clusters": [{"status": "my_status"}]})
        a_mock.get_waiter().wait = wait_mock

        max_attempts = 5
        trigger = ClusterWaiterTrigger("my_waiter", "cluster_arn", 0, max_attempts, None, None)

        with pytest.raises(AirflowException):
            generator = trigger.run()
            await generator.asend(None)

        assert wait_mock.call_count == max_attempts

    @pytest.mark.asyncio
    @mock.patch.object(EcsHook, "async_conn")
    async def test_run_success(self, client_mock):
        a_mock = mock.MagicMock()
        client_mock.__aenter__.return_value = a_mock
        wait_mock = AsyncMock()
        a_mock.get_waiter().wait = wait_mock

        trigger = ClusterWaiterTrigger("my_waiter", "cluster_arn", 0, 5, None, None)

        generator = trigger.run()
        response: TriggerEvent = await generator.asend(None)

        assert response.payload["status"] == "success"
        assert response.payload["arn"] == "cluster_arn"

    @pytest.mark.asyncio
    @mock.patch.object(EcsHook, "async_conn")
    async def test_run_error(self, client_mock):
        a_mock = mock.MagicMock()
        client_mock.__aenter__.return_value = a_mock
        wait_mock = AsyncMock()
        wait_mock.side_effect = WaiterError("terminal failure", "reason", {})
        a_mock.get_waiter().wait = wait_mock

        trigger = ClusterWaiterTrigger("my_waiter", "cluster_arn", 0, 5, None, None)

        with pytest.raises(WaiterError):
            generator = trigger.run()
            await generator.asend(None)


class TestTaskDoneTrigger:
    @pytest.mark.asyncio
    @mock.patch.object(EcsHook, "async_conn")
    # this mock is only necessary to avoid a "No module named 'aiobotocore'" error in the LatestBoto CI step
    @mock.patch.object(AwsLogsHook, "async_conn")
    async def test_run_until_error(self, _, client_mock):
        a_mock = mock.MagicMock()
        client_mock.__aenter__.return_value = a_mock
        wait_mock = AsyncMock()
        wait_mock.side_effect = [
            WaiterError("name", "reason", {"tasks": [{"lastStatus": "my_status"}]}),
            WaiterError("name", "reason", {"tasks": [{"lastStatus": "my_status"}]}),
            WaiterError("terminal failure", "reason", {}),
        ]
        a_mock.get_waiter().wait = wait_mock

        trigger = TaskDoneTrigger("cluster", "task_arn", 0, 10, None, None)

        with pytest.raises(WaiterError):
            generator = trigger.run()
            await generator.asend(None)

        assert wait_mock.call_count == 3

    @pytest.mark.asyncio
    @mock.patch.object(EcsHook, "async_conn")
    # this mock is only necessary to avoid a "No module named 'aiobotocore'" error in the LatestBoto CI step
    @mock.patch.object(AwsLogsHook, "async_conn")
    async def test_run_success(self, _, client_mock):
        a_mock = mock.MagicMock()
        client_mock.__aenter__.return_value = a_mock
        wait_mock = AsyncMock()
        a_mock.get_waiter().wait = wait_mock

        trigger = TaskDoneTrigger("cluster", "my_task_arn", 0, 10, None, None)

        generator = trigger.run()
        response: TriggerEvent = await generator.asend(None)

        assert response.payload["status"] == "success"
        assert response.payload["task_arn"] == "my_task_arn"
