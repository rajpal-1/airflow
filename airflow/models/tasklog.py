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

from sqlalchemy import Column, Integer, Text

from airflow.models.base import Base
from airflow.utils import timezone
from airflow.utils.sqlalchemy import UtcDateTime

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.orm import Mapped


class LogTemplate(Base):
    """Changes to ``log_filename_template`` and ``elasticsearch_id``.

    This table is automatically populated when Airflow starts up, to store the
    config's value if it does not match the last row in the table.
    """

    __tablename__ = "log_template"
    _table_args_ = lambda: (
        Column("id", Integer(), primary_key=True, autoincrement=True),
        Column("filename", Text(), nullable=False),
        Column("elasticsearch_id", Text(), nullable=False),
        Column("created_at", UtcDateTime(), nullable=False, default=timezone.utcnow),
    )

    id: Mapped[int]
    filename: Mapped[str]
    elasticsearch_id: Mapped[str]
    created_at: Mapped[datetime]

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={getattr(self, k)}" for k in ("filename", "elasticsearch_id"))
        return f"LogTemplate({attrs})"
