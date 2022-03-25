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
import warnings
from typing import Any, Dict

from azure.mgmt.containerinstance.models import AzureFileVolume, Volume

from airflow.hooks.base import BaseHook


class AzureContainerVolumeHook(BaseHook):
    """
    A hook which wraps an Azure Volume.

    :param azure_container_volume_conn_id: Reference to the
        :ref:`Azure Container Volume connection id <howto/connection:azure_container_volume>`
        of an Azure account of which container volumes should be used.
    """

    conn_name_attr = "azure_container_volume_conn_id"
    default_conn_name = 'azure_container_volume_default'
    conn_type = 'azure_container_volume'
    hook_name = 'Azure Container Volume'

    _EXTRA_PREFIX_DEPRECATED = True
    """This attribute lets the webserver know whether the hook has been updated to handle the
     deprecation of the `extra__...` prefix in custom fields."""

    def __init__(self, azure_container_volume_conn_id: str = 'azure_container_volume_default') -> None:
        super().__init__()
        self.conn_id = azure_container_volume_conn_id

    @staticmethod
    def get_connection_form_widgets() -> Dict[str, Any]:
        """Returns connection widgets to add to connection form"""
        from flask_appbuilder.fieldwidgets import BS3PasswordFieldWidget
        from flask_babel import lazy_gettext
        from wtforms import PasswordField

        return {
            "connection_string": PasswordField(
                lazy_gettext('Blob Storage Connection String (optional)'), widget=BS3PasswordFieldWidget()
            ),
        }

    @staticmethod
    def get_ui_field_behaviour() -> Dict[str, Any]:
        """Returns custom field behaviour"""
        return {
            "hidden_fields": ['schema', 'port', 'host', "extra"],
            "relabeling": {
                'login': 'Azure Client ID',
                'password': 'Azure Secret',
            },
            "placeholders": {
                'login': 'client_id (token credentials auth)',
                'password': 'secret (token credentials auth)',
                'extra__azure_container_volume__connection_string': 'connection string auth',
            },
        }

    def get_storagekey(self) -> str:
        """Get Azure File Volume storage key"""
        conn = self.get_connection(self.conn_id)
        extras = conn.extra_dejson

        connection_string = self._get_field(extras, 'connection_string')
        if connection_string:
            for keyvalue in connection_string.split(";"):
                key, value = keyvalue.split("=", 1)
                if key == "AccountKey":
                    return value
        return conn.password

    def get_file_volume(
        self, mount_name: str, share_name: str, storage_account_name: str, read_only: bool = False
    ) -> Volume:
        """Get Azure File Volume"""
        return Volume(
            name=mount_name,
            azure_file=AzureFileVolume(
                share_name=share_name,
                storage_account_name=storage_account_name,
                read_only=read_only,
                storage_account_key=self.get_storagekey(),
            ),
        )

    def _get_field(self, extras, field_name: str, default: Any = None) -> Any:
        """Fetches a field from extras, and returns it."""
        long_f = f'extra__{self.conn_type}__{field_name}'
        if long_f in extras:
            warnings.warn(
                f"Extra param {long_f!r} in conn {self.conn_id!r} has been renamed to {field_name}. "
                f"Please update your connection prior to the next major release for this provider.",
                DeprecationWarning,
            )
            return extras[long_f]
        elif field_name in extras:
            return extras[field_name]
        else:
            return default
