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

import asyncio
import time
from typing import AsyncIterator

from airflow.providers.microsoft.azure.hooks.powerbi import (
    PowerBIAsyncHook,
    PowerBIDatasetRefreshStatus,
)
from airflow.triggers.base import BaseTrigger, TriggerEvent


class PowerBITrigger(BaseTrigger):
    """
    Triggers when Power BI dataset refresh is completed.

    Wait for termination will always be True.

    :param powerbi_conn_id: The connection Id to connect to PowerBI.
    :param dataset_id: The dataset Id to refresh.
    :param group_id: The workspace Id where dataset is located.
    :param dataset_refresh_id: The dataset refresh Id.
    :param end_time: Time in seconds when trigger should stop polling.
    :param check_interval: Time in seconds to wait between each poll.
    :param wait_for_termination: Wait for the dataset refresh to complete or fail.
    """

    def __init__(
        self,
        powerbi_conn_id: str,
        dataset_id: str,
        group_id: str,
        dataset_refresh_id: str,
        end_time: float,
        check_interval: int = 60,
        wait_for_termination: bool = True,
    ):
        super().__init__()
        self.powerbi_conn_id = powerbi_conn_id
        self.dataset_id = dataset_id
        self.group_id = group_id
        self.dataset_refresh_id = dataset_refresh_id
        self.end_time = end_time
        self.check_interval = check_interval
        self.wait_for_termination = wait_for_termination

    def serialize(self):
        """Serialize the trigger instance."""
        return (
            "airflow.providers.microsoft.azure.triggers.powerbi.PowerBITrigger",
            {
                "powerbi_conn_id": self.powerbi_conn_id,
                "dataset_id": self.dataset_id,
                "group_id": self.group_id,
                "dataset_refresh_id": self.dataset_refresh_id,
                "end_time": self.end_time,
                "check_interval": self.check_interval,
                "wait_for_termination": self.wait_for_termination,
            },
        )

    async def run(self) -> AsyncIterator[TriggerEvent]:
        """Make async connection to the PowerBI and polls for the dataset refresh status."""
        hook = PowerBIAsyncHook(powerbi_conn_id=self.powerbi_conn_id)
        try:
            while self.end_time > time.time():
                dataset_refresh_status = await hook.get_dataset_refresh_status(
                    dataset_id=self.dataset_id,
                    group_id=self.group_id,
                    dataset_refresh_id=self.dataset_refresh_id,
                )
                if dataset_refresh_status == PowerBIDatasetRefreshStatus.COMPLETED:
                    yield TriggerEvent(
                        {
                            "status": {dataset_refresh_status},
                            "message": f"The dataset refresh {self.dataset_refresh_id} has {dataset_refresh_status}.",
                            "dataset_refresh_id": self.dataset_refresh_id,
                        }
                    )
                    return
                elif dataset_refresh_status == PowerBIDatasetRefreshStatus.FAILED:
                    yield TriggerEvent(
                        {
                            "status": {dataset_refresh_status},
                            "message": f"The dataset refresh {self.dataset_refresh_id} has {dataset_refresh_status}.",
                            "dataset_refresh_id": self.dataset_refresh_id,
                        }
                    )
                    return
                self.log.info(
                    "Sleeping for %s. The dataset refresh status is %s.",
                    self.check_interval,
                    dataset_refresh_status,
                )
                await asyncio.sleep(self.check_interval)

            yield TriggerEvent(
                {
                    "status": "error",
                    "message": f"Timeout: The dataset refresh {self.dataset_refresh_id} has {dataset_refresh_status}.",
                    "dataset_refresh_id": self.dataset_refresh_id,
                }
            )
            return
        except Exception as error:
            if self.dataset_refresh_id:
                try:
                    self.log.info(
                        "Unexpected error %s caught. Cancel pipeline run %s", error, self.dataset_refresh_id
                    )
                    await hook.cancel_dataset_refresh(
                        dataset_id=self.dataset_id,
                        group_id=self.group_id,
                        dataset_refresh_id=self.dataset_refresh_id,
                    )
                except Exception as e:
                    yield TriggerEvent(
                        {
                            "status": "error",
                            "message": f"An error occurred while canceling pipeline: {e}",
                            "dataset_refresh_id": self.dataset_refresh_id,
                        }
                    )
                    return
            yield TriggerEvent(
                {
                    "status": "error",
                    "message": f"An error occurred: {error}",
                    "dataset_refresh_id": self.dataset_refresh_id,
                }
            )
            return
