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
"""Hook for sending or receiving data from PagerDuty as well as creating PagerDuty incidents."""
import warnings
from typing import Any, Dict, List, Optional

import pdpyras

from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook


class PagerdutyHook(BaseHook):
    """
    The PagerdutyHook can be used to interact with both the PagerDuty API and the PagerDuty Events API.

    Takes both PagerDuty API token directly and connection that has PagerDuty API token.
    If both supplied, PagerDuty API token will be used.
    In these cases, the PagerDuty API token refers to an account token:
    https://support.pagerduty.com/docs/generating-api-keys#generating-a-general-access-rest-api-key
    https://support.pagerduty.com/docs/generating-api-keys#generating-a-personal-rest-api-key

    In order to send events (with the Pagerduty Events API), you will also need to specify the
    routing_key in the `extra` field

    :param token: PagerDuty API token
    :param pagerduty_conn_id: connection that has PagerDuty API token in the password field
    """

    conn_name_attr = "pagerduty_conn_id"
    default_conn_name = "pagerduty_default"
    conn_type = "pagerduty"
    hook_name = "Pagerduty"

    @staticmethod
    def get_ui_field_behaviour() -> Dict:
        """Returns custom field behaviour"""
        return {
            "hidden_fields": ['port', 'login', 'extra', 'schema'],
            "relabeling": {
                'password': 'Pagerduty API token',
            },
        }

    def __init__(self, token: Optional[str] = None, pagerduty_conn_id: Optional[str] = None) -> None:
        super().__init__()
        self.routing_key = None
        self._session = None

        if pagerduty_conn_id is not None:
            conn = self.get_connection(pagerduty_conn_id)
            self.token = conn.get_password()

            routing_key = conn.extra_dejson.get("extra__pagerduty__routing_key")
            if routing_key:
                self.routing_key = routing_key

        if token is not None:  # token takes higher priority
            self.token = token

        if self.token is None:
            raise AirflowException('Cannot get token: No valid api token nor pagerduty_conn_id supplied.')

    def get_session(self) -> pdpyras.APISession:
        """
        Returns `pdpyras.APISession` for use with sending or receiving data through the PagerDuty REST API.

        The `pdpyras` library supplies a class `pdpyras.APISession` extending `requests.Session` from the
        Requests HTTP library.

        Documentation on how to use the `APISession` class can be found at:
        https://pagerduty.github.io/pdpyras/#data-access-abstraction
        """
        self._session = pdpyras.APISession(self.token)
        return self._session

    def create_event(
        self,
        summary: str,
        severity: str,
        source: str = "airflow",
        action: str = "trigger",
        routing_key: Optional[str] = None,
        dedup_key: Optional[str] = None,
        custom_details: Optional[Any] = None,
        group: Optional[str] = None,
        component: Optional[str] = None,
        class_type: Optional[str] = None,
        images: Optional[List[Any]] = None,
        links: Optional[List[Any]] = None,
    ) -> Dict:
        """
        Create event for service integration.

        :param summary: Summary for the event
        :type summary: str
        :param severity: Severity for the event, needs to be one of: info, warning, error, critical
        :type severity: str
        :param source: Specific human-readable unique identifier, such as a
            hostname, for the system having the problem.
        :type source: str
        :param action: Event action, needs to be one of: trigger, acknowledge,
            resolve. Default to trigger if not specified.
        :type action: str
        :param routing_key: Integration key. If not specified, will try to read
            from connection's extra json blob.
        :type routing_key: str
        :param dedup_key: A string which identifies the alert triggered for the given event.
            Required for the actions acknowledge and resolve.
        :type dedup_key: str
        :param custom_details: Free-form details from the event. Can be a dictionary or a string.
            If a dictionary is passed it will show up in PagerDuty as a table.
        :type custom_details: dict or str
        :param group: A cluster or grouping of sources. For example, sources
            “prod-datapipe-02” and “prod-datapipe-03” might both be part of “prod-datapipe”
        :type group: str
        :param component: The part or component of the affected system that is broken.
        :type component: str
        :param class_type: The class/type of the event.
        :type class_type: str
        :param images: List of images to include. Each dictionary in the list accepts the following keys:
            `src`: The source (URL) of the image being attached to the incident. This image must be served via
            HTTPS.
            `href`: [Optional] URL to make the image a clickable link.
            `alt`: [Optional] Alternative text for the image.
        :type images: list[dict]
        :param links: List of links to include. Each dictionary in the list accepts the following keys:
            `href`: URL of the link to be attached.
            `text`: [Optional] Plain text that describes the purpose of the link, and can be used as the
            link's text.
        :type links: list[dict]
        :return: PagerDuty Events API v2 response.
        :rtype: dict
        """
        warnings.warn(
            "This method will be deprecated. Please use the "
            "`airflow.providers.pagerduty.hooks.PagerdutyEventsHook` to interact with the Events API",
            DeprecationWarning,
            stacklevel=2,
        )

        routing_key = self.routing_key or routing_key

        return PagerdutyEventsHook(integration_key=routing_key).create_event(
            summary=summary,
            severity=severity,
            source=source,
            action=action,
            dedup_key=dedup_key,
            custom_details=custom_details,
            group=group,
            component=component,
            class_type=class_type,
            images=images,
            links=links,
        )


class PagerdutyEventsHook(BaseHook):
    """
    This class can be used to interact with the Pagerduty Events API.

    It takes both an Events API token and a PagerDuty connection with the Events API token
     (i.e. Integration key) as the password. If both supplied, the token will be used.

    :param integration_key: PagerDuty Events API token
    :param pagerduty_conn_id: connection that has PagerDuty integration key in the Pagerduty API token field
    """

    def __init__(
        self, integration_key: Optional[str] = None, pagerduty_conn_id: Optional[str] = None
    ) -> None:
        super().__init__()
        self.integration_key = None
        self._session = None

        if pagerduty_conn_id is not None:
            conn = self.get_connection(pagerduty_conn_id)
            self.integration_key = conn.get_password()

        if integration_key is not None:  # token takes higher priority
            self.integration_key = integration_key

        if self.integration_key is None:
            raise AirflowException(
                'Cannot get token: No valid integration key nor pagerduty_conn_id supplied.'
            )

    def create_event(
        self,
        summary: str,
        severity: str,
        source: str = 'airflow',
        action: str = 'trigger',
        dedup_key: Optional[str] = None,
        custom_details: Optional[Any] = None,
        group: Optional[str] = None,
        component: Optional[str] = None,
        class_type: Optional[str] = None,
        images: Optional[List[Any]] = None,
        links: Optional[List[Any]] = None,
    ) -> Dict:
        """
        Create event for service integration.

        :param summary: Summary for the event
        :type summary: str
        :param severity: Severity for the event, needs to be one of: info, warning, error, critical
        :type severity: str
        :param source: Specific human-readable unique identifier, such as a
            hostname, for the system having the problem.
        :type source: str
        :param action: Event action, needs to be one of: trigger, acknowledge,
            resolve. Default to trigger if not specified.
        :type action: str
        :param dedup_key: A string which identifies the alert triggered for the given event.
            Required for the actions acknowledge and resolve.
        :type dedup_key: str
        :param custom_details: Free-form details from the event. Can be a dictionary or a string.
            If a dictionary is passed it will show up in PagerDuty as a table.
        :type custom_details: dict or str
        :param group: A cluster or grouping of sources. For example, sources
            “prod-datapipe-02” and “prod-datapipe-03” might both be part of “prod-datapipe”
        :type group: str
        :param component: The part or component of the affected system that is broken.
        :type component: str
        :param class_type: The class/type of the event.
        :type class_type: str
        :param images: List of images to include. Each dictionary in the list accepts the following keys:
            `src`: The source (URL) of the image being attached to the incident. This image must be served via
            HTTPS.
            `href`: [Optional] URL to make the image a clickable link.
            `alt`: [Optional] Alternative text for the image.
        :type images: list[dict]
        :param links: List of links to include. Each dictionary in the list accepts the following keys:
            `href`: URL of the link to be attached.
            `text`: [Optional] Plain text that describes the purpose of the link, and can be used as the
            link's text.
        :type links: list[dict]
        :return: PagerDuty Events API v2 response.
        :rtype: dict
        """
        payload = {
            "summary": summary,
            "severity": severity,
            "source": source,
        }
        if custom_details is not None:
            payload["custom_details"] = custom_details
        if component:
            payload["component"] = component
        if group:
            payload["group"] = group
        if class_type:
            payload["class"] = class_type

        actions = ('trigger', 'acknowledge', 'resolve')
        if action not in actions:
            raise ValueError(f"Event action must be one of: {', '.join(actions)}")
        data = {
            "event_action": action,
            "payload": payload,
        }
        if dedup_key:
            data["dedup_key"] = dedup_key
        elif action != 'trigger':
            raise ValueError(
                "The dedup_key property is required for event_action=%s events, and it must \
                be a string."
                % action
            )
        if images is not None:
            data["images"] = images
        if links is not None:
            data["links"] = links

        session = pdpyras.EventsAPISession(self.integration_key)
        resp = session.post("/v2/enqueue", json=data)
        resp.raise_for_status()
        return resp.json()
