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

import datetime

import flask
import flask_login
from flask import session as flask_session

from airflow.configuration import conf


def init_logout_timeout(app):
    @app.before_request
    def before_request():
        _force_log_out_after = conf.getint('webserver', 'FORCE_LOG_OUT_AFTER', fallback=0)
        if _force_log_out_after > 0:
            flask.session.permanent = True
            app.permanent_session_lifetime = datetime.timedelta(minutes=_force_log_out_after)
            flask.session.modified = True
            flask.g.user = flask_login.current_user


def init_permanent_session(app):
    @app.before_request
    def make_session_permanent():
        flask_session.permanent = True
