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

from sqlalchemy import Column, Integer, String, Text

from airflow.models.base import Base
from airflow.utils.sqlalchemy import UtcDateTime

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.orm import Mapped


class ImportError(Base):
    """Stores all Import Errors which are recorded when parsing DAGs and displayed on the Webserver."""

    __tablename__ = "import_error"
    _table_args_ = lambda: (
        Column("id", Integer(), primary_key=True),
        Column("timestamp", UtcDateTime()),
        Column("filename", String(1024)),
        Column("stacktrace", Text()),
        Column("processor_subdir", String(2000), nullable=True),
    )
    id: Mapped[int]
    timestamp: Mapped[datetime | None]
    filename: Mapped[str | None]
    stacktrace: Mapped[str | None]
    processor_subdir: Mapped[str | None]
