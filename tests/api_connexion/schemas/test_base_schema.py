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

import unittest

from marshmallow import fields

from airflow.api_connexion.schemas.base_schema import BaseSchema
from airflow.exceptions import AirflowException


class TestBaseSchema(unittest.TestCase):

    def test_get_envelope_key_raises(self):
        class MySchema(BaseSchema):
            name = fields.Str()
            number = fields.Int()

        schema = MySchema(many=True)
        with self.assertRaises(AirflowException):
            schema.dump([{'name': 'test'}, {'name': 'test2'}])

    def test_get_envelope_key_pass(self):

        class Model:
            def __init__(self, name=None, number=None):
                self.name = name
                self.number = number

        class MySchema(BaseSchema):
            __envelope__ = {'many': 'children'}
            STRING_FIELDS = ['name']
            INTEGER_FIELDS = ['number']
            name = fields.Str()
            number = fields.Int()

        schema = MySchema(many=True)
        obj1 = Model(name='test', number=5)
        obj2 = Model(name='test2', number=6)

        result = schema.dump([obj1, obj2])
        self.assertEqual(
            result[0],
            {
                'children': [
                    {'name': 'test', 'number': 5},
                    {'name': 'test2', 'number': 6}
                ],
                'total_entries': 2
            }
        )

    def test_process_list_obj(self):
        class Model:
            def __init__(self, name=None, number=None):
                self.name = name
                self.number = number

        class MySchema(BaseSchema):
            __envelope__ = {'many': 'children'}
            STRING_FIELDS = ['name']
            INTEGER_FIELDS = ['number']
            name = fields.Str()
            number = fields.Int()

        schema = MySchema(many=True)
        obj1 = Model(name=None, number=None)
        obj2 = Model(name=None, number=None)

        result = schema.dump([obj1, obj2])
        self.assertEqual(
            result[0],
            {
                'children': [
                    {'name': '', 'number': 0},
                    {'name': '', 'number': 0}
                ],
                'total_entries': 2
            }
        )

    def test_process_obj(self):
        class Model:
            def __init__(self, name=None, number=None):
                self.name = name
                self.number = number

        class MySchema(BaseSchema):
            __envelope__ = {'many': 'children'}
            STRING_FIELDS = ['name']
            INTEGER_FIELDS = ['number']
            name = fields.Str()
            number = fields.Int()

        schema = MySchema()
        obj = Model(name=None, number=None)

        result = schema.dump(obj)
        self.assertEqual(result[0],
                         {'name': '', 'number': 0},
                         )
