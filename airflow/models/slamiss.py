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

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Column, Index, String, Text

from airflow.models.base import COLLATION_ARGS, ID_LEN, Base
from airflow.utils.sqlalchemy import UtcDateTime

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.orm import Mapped


class SlaMiss(Base):
    """
    Model that stores a history of the SLA that have been missed.

    It is used to keep track of SLA failures over time and to avoid double triggering alert emails.
    """

    __tablename__ = "sla_miss"
    _table_args_ = lambda: (
        Column("task_id", String(ID_LEN, **COLLATION_ARGS), primary_key=True),
        dag_id := Column("dag_id", String(ID_LEN, **COLLATION_ARGS), primary_key=True),
        Column("execution_date", UtcDateTime(), primary_key=True),
        Column("email_sent", Boolean(), default=False),
        Column("timestamp", UtcDateTime()),
        Column("description", Text()),
        Column("notification_sent", Boolean(), default=False),
        Index("sm_dag", dag_id, unique=False),
    )

    task_id: Mapped[str]
    dag_id: Mapped[str]
    execution_date: Mapped[datetime]
    email_sent: Mapped[bool | None]
    timestamp: Mapped[datetime | None]
    description: Mapped[str | None]
    notification_sent: Mapped[bool | None]

    def __repr__(self):
        return str((self.dag_id, self.task_id, self.execution_date.isoformat()))
