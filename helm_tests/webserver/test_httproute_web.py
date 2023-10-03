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

import jmespath
import pytest

from tests.charts.helm_template_generator import render_chart


class TestHttpRouteWeb:
    """Tests httpRoute web."""

    def test_should_pass_validation_with_httproute_enabled_and_gateway_set(self):
        render_chart(
            values={"httpRoute": {"web": {"enabled": True,
                                          "gateway": {"name": "test-gateway", "namespace": "test"}}}},
            show_only=["templates/webserver/webserver-httproute.yaml"],
        )  # checks that no validation exception is raised

    def test_should_allow_more_than_one_annotation(self):
        docs = render_chart(
            values={"httpRoute": {"web": {"enabled": True,
                                          "annotations": {"aa": "bb", "cc": "dd"},
                                          "gateway": {"name": "test-gateway", "namespace": "test"}}}},
            show_only=["templates/webserver/webserver-httproute.yaml"],
        )
        assert {"aa": "bb", "cc": "dd"} == jmespath.search("metadata.annotations", docs[0])

    def test_should_set_httproute_gateway_name(self):
        docs = render_chart(
            values={"httpRoute": {"web": {"enabled": True,
                                          "gateway": {"name": "foo"}}}},
            show_only=["templates/webserver/webserver-httproute.yaml"],
        )
        assert "foo" == jmespath.search("spec.parentRefs[0].name", docs[0])

    def test_should_httproute_hostnames_entry_not_exist(self):
        docs = render_chart(
            values={"httpRoute": {"web": {"enabled": True,
                                          "gateway": {"name": "test-gateway", "namespace": "test"}}}},
            show_only=["templates/webserver/webserver-httproute.yaml"],
        )
        assert not jmespath.search("spec.hostnames", docs[0])

    @pytest.mark.parametrize(
        "global_value, web_value, expected",
        [
            (None, None, False),
            (None, False, False),
            (None, True, True),
            (False, None, False),
            (True, None, True),
            (False, True, True),  # We will deploy it if _either_ are true
            (True, False, True),
        ],
    )
    def test_httproute_created(self, global_value, web_value, expected):
        values={"httpRoute": {"web": {"gateway": {"name": "test-gateway", "namespace": "test"}}}},
        if global_value is not None:
            values["httpRoute"]["enabled"] = global_value
        if web_value is not None:
            values["httpRoute"]["web"] = {"enabled": web_value}
        if values["httpRoute"] == {}:
            del values["httpRoute"]
        docs = render_chart(values=values, show_only=["templates/webserver/webserver-httproute.yaml"])
        assert expected == (1 == len(docs))

    def test_should_add_component_specific_labels(self):
        docs = render_chart(
            values={
                "httpRoute": {"web": {"enabled": True,
                                      "gateway": {"name": "test-gateway", "namespace": "test"}}},
                "webserver": {
                    "labels": {"test_label": "test_label_value"},
                },
            },
            show_only=["templates/webserver/webserver-httproute.yaml"],
        )
        assert "test_label" in jmespath.search("metadata.labels", docs[0])
        assert jmespath.search("metadata.labels", docs[0])["test_label"] == "test_label_value"
