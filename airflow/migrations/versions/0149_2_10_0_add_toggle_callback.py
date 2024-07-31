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
add toggle callback

Revision ID: 771ae9d1d541
Revises: ec3471c1e067
Create Date: 2024-07-16 06:36:10.959083

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "771ae9d1d541"
down_revision = "ec3471c1e067"
branch_labels = None
depends_on = None
airflow_version = "2.10.0"


def upgrade():
    """Apply add toggle callback"""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "toggle_dag",
        sa.Column("dag_id", sa.String(length=250), nullable=False),
        sa.Column("last_verified_time", airflow.utils.sqlalchemy.UtcDateTime(timezone=True), nullable=True),
        sa.Column("is_dag_paused", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("dag_id", name=op.f("toggle_dag_pkey")),
    )
    with op.batch_alter_table("toggle_dag", schema=None) as batch_op:
        batch_op.create_index("toggle_dag", ["dag_id"], unique=False)

    # ### end Alembic commands ###


def downgrade():
    """Unapply add toggle callback"""
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("toggle_dag", schema=None) as batch_op:
        batch_op.drop_index("toggle_dag")

    op.drop_table("toggle_dag")
    # ### end Alembic commands ###
