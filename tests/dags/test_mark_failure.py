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
from datetime import datetime

from jobs.test_local_task_job import data
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator

DEFAULT_DATE = datetime(2016, 1, 1)


def check_failure(context):
    context
    data['called'] = True


args = {
    'owner': 'airflow',
    'start_date': DEFAULT_DATE,
}

dag = DAG(dag_id='test_mark_success', default_args=args)

task = BashOperator(
    task_id='task1',
    bash_command='sleep 600',
    on_failure_callback=check_failure,
    dag=dag)
