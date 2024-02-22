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

"""add new executor field to db

Revision ID: 677fdbb7fc54
Revises: ab34f260b71c
Create Date: 2024-03-11 15:26:59.186579

"""

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = '677fdbb7fc54'
down_revision = 'ab34f260b71c'
branch_labels = None
depends_on = None
airflow_version = '2.9.0'


def upgrade():
    """Apply add executor field to task instance"""
    # NOTE: I cannot find a straightforward sdk doc for SA types. I don't know
    # if I need default=None. No idea how to tell what are the defaults without docs
    op.add_column('task_instance', sa.Column('executor', sa.String(length=1000), default=None))


def downgrade():
    """Unapply add executor field to task instance"""
    op.drop_column('task_instance', 'executor')
