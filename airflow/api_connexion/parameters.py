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
from functools import wraps
from typing import Callable, Dict

from jsonschema import draft4_format_checker
from pendulum.parsing import ParserError

from airflow.api_connexion.exceptions import BadRequest
from airflow.configuration import conf
from airflow.utils import timezone

# Database entity fields
dag_id = "dag_id"
pool_id = "pool_id"


@draft4_format_checker.checks('timezone-datetime')
def is_custom_date(val):
    """ Custom datetime format"""
    if val is None:  # this avoids validating nullable fields
        return True
    if not isinstance(val, str):
        return False
    if ' ' in val:
        return False
    try:
        _ = timezone.parse(val)
    except (ParserError, TypeError):
        return False
    return True


def check_limit(value: int):
    """
    This checks the limit passed to view and raises BadRequest if
    limit exceed user configured value
    """
    max_val = conf.getint("api", "maximum_page_limit")  # user configured max page limit
    fallback = conf.getint("api", "fallback_page_limit")

    if value > max_val:
        return max_val
    if value == 0:
        return fallback
    if value < 0:
        raise BadRequest("Page limit must be a positive integer")
    return value


def format_parameters(params_formatters: Dict[str, Callable[..., bool]]):
    """
    Decorator factory that create decorator that convert parameters using given formatters.

    Using it allows you to separate parameter formatting from endpoint logic.

    :param params_formatters: Map of key name and formatter function
    """

    def format_parameters_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            for key, formatter in params_formatters.items():
                if key in kwargs:
                    kwargs[key] = formatter(kwargs[key])
            return func(*args, **kwargs)

        return wrapped_function

    return format_parameters_decorator
