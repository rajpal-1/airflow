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

"""Connect to Asana."""

from typing import Any, Dict

from asana import Client
from cached_property import cached_property

from airflow.hooks.base import BaseHook


class AsanaHook(BaseHook):
    """Wrapper around Asana Python client library."""

    conn_name_attr = "asana_conn_id"
    default_conn_name = "asana_default"
    conn_type = "asana"
    hook_name = "Asana"

    def __init__(self, conn_id: str = default_conn_name, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.connection = self.get_connection(conn_id)
        extras = self.connection.extra_dejson
        self.workspace = extras.get("extra__asana__workspace") or None
        self.project = extras.get("extra__asana__project") or None

    def get_conn(self) -> Client:
        return self.client

    @staticmethod
    def get_connection_form_widgets() -> Dict[str, Any]:
        """Returns connection widgets to add to connection form"""
        from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
        from flask_babel import lazy_gettext
        from wtforms import StringField

        return {
            "extra__asana__workspace": StringField(lazy_gettext("Workspace"), widget=BS3TextFieldWidget()),
            "extra__asana__project": StringField(lazy_gettext("Project"), widget=BS3TextFieldWidget()),
        }

    @staticmethod
    def get_ui_field_behaviour() -> Dict:
        """Returns custom field behaviour"""
        return {
            "hidden_fields": ["port", "host", "login", "schema"],
            "relabeling": {},
            "placeholders": {
                "password": "Asana personal access token",
                "extra__asana__workspace": "Asana workspace gid",
                "extra__asana__project": "Asana project gid",
            },
        }

    @cached_property
    def client(self) -> Client:
        """Instantiates python-asana Client"""
        if not self.connection.password:
            raise ValueError(
                "Asana connection password must contain a personal access token: "
                "https://developers.asana.com/docs/personal-access-token"
            )

        return Client.access_token(self.connection.password)

    def create_task(self, task_name: str, params: dict) -> dict:
        """
        Creates an Asana task. For a complete list of possible parameters, see
        https://developers.asana.com/docs/create-a-task
        """
        merged_params = self._merge_create_task_parameters(task_name, params)
        self._validate_create_task_parameters(merged_params)
        response = self.client.tasks.create(params=merged_params)  # pylint: disable=no-member
        return response

    def _merge_create_task_parameters(self, task_name: str, task_params: dict) -> dict:
        """Merge create_task parameters with default params from the connection."""
        merged_params = {"name": task_name}
        if self.project:
            merged_params["projects"] = [self.project]
        elif self.workspace and not (task_params and ("projects" in task_params)):
            merged_params["workspace"] = self.workspace
        if task_params:
            merged_params.update(task_params)
        return merged_params

    @staticmethod
    def _validate_create_task_parameters(params: dict) -> None:
        """Check that user provided minimal create parameters."""
        required_parameters = {"workspace", "projects", "parent"}
        if required_parameters.isdisjoint(params):
            raise ValueError(
                f"You must specify at least one of {required_parameters} in the create_task parameters"
            )

    def delete_task(self, task_id: str) -> dict:
        """Deletes an Asana task."""
        response = self.client.tasks.delete_task(task_id)  # pylint: disable=no-member
        return response

    def find_task(self, params: dict) -> list:
        """
        Retrieves a list of Asana tasks that match search parameters. For a complete list of parameters,
        see https://bit.ly/3uIqMj0
        """
        merged_params = self._merge_find_task_parameters(params)
        self._validate_find_task_parameters(merged_params)
        response = self.client.tasks.find_all(params=merged_params)  # pylint: disable=no-member
        return list(response)

    def _merge_find_task_parameters(self, search_parameters: dict) -> dict:
        """Merge find_task parameters with default params from the connection."""
        merged_params = {}
        if self.project:
            merged_params["project"] = self.project
        # Only use default workspace if user did not provide a project id
        elif self.workspace and not (search_parameters and ("project" in search_parameters)):
            merged_params["workspace"] = self.workspace
        if search_parameters:
            merged_params.update(search_parameters)
        return merged_params

    @staticmethod
    def _validate_find_task_parameters(params: dict) -> None:
        """Check that user provided minimal search parameters."""
        one_of_list = {"project", "section", "tag", "user_task_list"}
        both_of_list = {"assignee", "workspace"}
        contains_both = both_of_list.issubset(params)
        contains_one = not one_of_list.isdisjoint(params)
        if not (contains_both or contains_one):
            raise ValueError(
                f"You must specify at least one of {one_of_list} "
                f"or both of {both_of_list} in the find_task parameters."
            )

    def update_task(self, task_id: str, params: dict) -> dict:
        """
        Updates an existing Asana task. For a complete list of possible parameters, see
        https://developers.asana.com/docs/update-a-task
        """
        response = self.client.tasks.update(task_id, params)  # pylint: disable=no-member
        return response

    def create_project(self, params: dict) -> dict:
        """
        Creates a new project. See
        https://developers.asana.com/docs/create-a-project#create-a-project-parameters
        for a list of possible parameters; you must specify at least one of
        'workspace' and 'team'.
        """
        merged_params = self._merge_project_parameters(params)
        self._validate_create_project_parameters(merged_params)
        response = self.client.projects.create(merged_params)  # pylint: disable=no-member
        return response

    @staticmethod
    def _validate_create_project_parameters(params: dict) -> None:
        """Check that user provided minimal create parameters."""
        required_parameters = {"workspace", "team"}
        if required_parameters.isdisjoint(params):
            raise ValueError(
                f"You must specify at least one of {required_parameters} in the create_project parameters"
            )

    def _merge_project_parameters(self, params: dict) -> dict:
        """Merge parameters passed in to a project method with default params from the connection."""
        merged_params = {} if self.workspace is None else {"workspace": self.workspace}
        merged_params.update(params)
        return merged_params

    def find_project(self, params: dict) -> list:
        """
        Searches for projects matching criteria. See https://bit.ly/31ZrZq6
        for a list of possible parameters.
        """
        merged_params = self._merge_project_parameters(params)
        response = self.client.projects.find_all(merged_params)  # pylint: disable=no-member
        return list(response)

    def update_project(self, project_id: str, params: dict) -> dict:
        """
        Updates an existing project. See
        https://developers.asana.com/docs/update-a-project#update-a-project-parameters
        for a list of possible parameters
        """
        response = self.client.projects.update(project_id, params)  # pylint: disable=no-member
        return response

    def delete_project(self, project_id: str) -> dict:
        """Deletes a project."""
        response = self.client.projects.delete(project_id)  # pylint: disable=no-member
        return response
