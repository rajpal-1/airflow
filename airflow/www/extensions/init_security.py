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

import logging

from flask import redirect, request

from airflow.configuration import conf
from airflow.www.extensions.init_auth_manager import get_auth_manager

log = logging.getLogger(__name__)


def init_xframe_protection(app):
    """
    Add X-Frame-Options header.

    Use it to avoid click-jacking attacks, by ensuring that their content is not embedded into other sites.

    See also: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/X-Frame-Options
    """
    x_frame_enabled = conf.getboolean("webserver", "X_FRAME_ENABLED", fallback=True)
    if x_frame_enabled:
        return

    def apply_caching(response):
        response.headers["X-Frame-Options"] = "DENY"
        return response

    app.after_request(apply_caching)


def init_cache_control(app):
    def apply_cache_control(response):
        if "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"
        return response

    app.after_request(apply_cache_control)


def init_check_user_active(app):
    @app.before_request
    def check_user_active():
        url_logout = get_auth_manager().get_url_logout()
        if request.path == url_logout:
            return
        if get_auth_manager().is_logged_in() and not get_auth_manager().get_user().is_active:
            return redirect(url_logout)
