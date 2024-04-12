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

from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKeyConstraint, String, Text, delete, false, select

from airflow.api_internal.internal_api_call import internal_api_call
from airflow.models.base import Base, StringID
from airflow.utils import timezone
from airflow.utils.retries import retry_db_transaction
from airflow.utils.session import NEW_SESSION, provide_session
from airflow.utils.sqlalchemy import UtcDateTime

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.orm import Mapped, Session


class DagWarning(Base):
    """
    A table to store DAG warnings.

    DAG warnings are problems that don't rise to the level of failing the DAG parse
    but which users should nonetheless be warned about.  These warnings are recorded
    when parsing DAG and displayed on the Webserver in a flash message.
    """

    __tablename__ = "dag_warning"
    _table_args_ = lambda: (
        Column("dag_id", StringID(), primary_key=True),
        Column("warning_type", String(50), primary_key=True),
        Column("message", Text(), nullable=False),
        Column("timestamp", UtcDateTime(), nullable=False, default=timezone.utcnow),
        ForeignKeyConstraint(
            ("dag_id",),
            ["dag.dag_id"],
            name="dcw_dag_id_fkey",
            ondelete="CASCADE",
        ),
    )

    dag_id: Mapped[str]
    warning_type: Mapped[str]
    message: Mapped[str]
    timestamp: Mapped[datetime]

    def __init__(self, dag_id: str, error_type: str, message: str, **kwargs):
        super().__init__(**kwargs)
        self.dag_id = dag_id
        self.warning_type = DagWarningType(error_type).value  # make sure valid type
        self.message = message

    def __eq__(self, other) -> bool:
        return self.dag_id == other.dag_id and self.warning_type == other.warning_type

    def __hash__(self) -> int:
        return hash((self.dag_id, self.warning_type))

    @classmethod
    @internal_api_call
    @provide_session
    def purge_inactive_dag_warnings(cls, session: Session = NEW_SESSION) -> None:
        """
        Deactivate DagWarning records for inactive dags.

        :return: None
        """
        cls._purge_inactive_dag_warnings_with_retry(session)

    @classmethod
    @retry_db_transaction
    def _purge_inactive_dag_warnings_with_retry(cls, session: Session) -> None:
        from airflow.models.dag import DagModel

        if session.get_bind().dialect.name == "sqlite":
            dag_ids_stmt = select(DagModel.dag_id).where(DagModel.is_active == false())
            query = delete(cls).where(cls.dag_id.in_(dag_ids_stmt.scalar_subquery()))
        else:
            query = delete(cls).where(cls.dag_id == DagModel.dag_id, DagModel.is_active == false())

        session.execute(query.execution_options(synchronize_session=False))
        session.commit()


class DagWarningType(str, Enum):
    """
    Enum for DAG warning types.

    This is the set of allowable values for the ``warning_type`` field
    in the DagWarning model.
    """

    NONEXISTENT_POOL = "non-existent pool"
