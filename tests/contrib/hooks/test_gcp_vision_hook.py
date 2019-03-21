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

import unittest

from google.cloud.vision import enums
from google.cloud.vision_v1 import ProductSearchClient
from google.cloud.vision_v1.proto.product_search_service_pb2 import ProductSet, Product, ReferenceImage
from google.protobuf.json_format import MessageToDict
from parameterized import parameterized

from airflow.contrib.hooks.gcp_vision_hook import CloudVisionHook
from tests.contrib.utils.base_gcp_mock import mock_base_gcp_hook_default_project_id

try:
    from unittest import mock
except ImportError:
    try:
        import mock
    except ImportError:
        mock = None

from airflow import AirflowException

PROJECT_ID_TEST = 'project-id'
PROJECT_ID_TEST_2 = 'project-id-2'
LOC_ID_TEST = 'loc-id'
LOC_ID_TEST_2 = 'loc-id-2'
PRODUCTSET_ID_TEST = 'ps-id'
PRODUCTSET_ID_TEST_2 = 'ps-id-2'
PRODUCTSET_NAME_TEST = 'projects/{}/locations/{}/productSets/{}'.format(
    PROJECT_ID_TEST, LOC_ID_TEST, PRODUCTSET_ID_TEST
)
PRODUCT_ID_TEST = 'p-id'
PRODUCT_ID_TEST_2 = 'p-id-2'
PRODUCT_NAME_TEST = "projects/{}/locations/{}/products/{}".format(
    PROJECT_ID_TEST, LOC_ID_TEST, PRODUCT_ID_TEST
)
PRODUCT_NAME = "projects/{}/locations/{}/products/{}".format(PROJECT_ID_TEST, LOC_ID_TEST, PRODUCT_ID_TEST)
REFERENCE_IMAGE_ID_TEST = 'ri-id'
REFERENCE_IMAGE_GEN_ID_TEST = 'ri-id'
ANNOTATE_IMAGE_REQUEST = {
    'image': {'source': {'image_uri': "gs://bucket-name/object-name"}},
    'features': [{'type': enums.Feature.Type.LOGO_DETECTION}],
}
REFERENCE_IMAGE_NAME_TEST = "projects/{}/locations/{}/products/{}/referenceImages/{}".format(
    PROJECT_ID_TEST, LOC_ID_TEST, PRODUCTSET_ID_TEST, REFERENCE_IMAGE_ID_TEST
)
REFERENCE_IMAGE_TEST = ReferenceImage(name=REFERENCE_IMAGE_GEN_ID_TEST)
REFERENCE_IMAGE_WITHOUT_ID_NAME = ReferenceImage()


class TestGcpVisionHook(unittest.TestCase):
    def setUp(self):
        with mock.patch(
            'airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.__init__',
            new=mock_base_gcp_hook_default_project_id,
        ):
            self.hook = CloudVisionHook(gcp_conn_id='test')

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_create_productset_explicit_id(self, get_conn):
        # Given
        create_product_set_method = get_conn.return_value.create_product_set
        create_product_set_method.return_value = None
        parent = ProductSearchClient.location_path(PROJECT_ID_TEST, LOC_ID_TEST)
        product_set = ProductSet()
        # When
        result = self.hook.create_product_set(
            location=LOC_ID_TEST,
            product_set_id=PRODUCTSET_ID_TEST,
            product_set=product_set,
            project_id=PROJECT_ID_TEST,
            retry=None,
            timeout=None,
            metadata=None,
        )

        # Then
        # ProductSet ID was provided explicitly in the method call above, should be returned from the method
        self.assertEqual(result, PRODUCTSET_ID_TEST)
        create_product_set_method.assert_called_once_with(
            parent=parent,
            product_set=product_set,
            product_set_id=PRODUCTSET_ID_TEST,
            retry=None,
            timeout=None,
            metadata=None,
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_create_productset_autogenerated_id(self, get_conn):
        # Given
        autogenerated_id = 'autogen-id'
        response_product_set = ProductSet(
            name=ProductSearchClient.product_set_path(PROJECT_ID_TEST, LOC_ID_TEST, autogenerated_id)
        )
        create_product_set_method = get_conn.return_value.create_product_set
        create_product_set_method.return_value = response_product_set
        parent = ProductSearchClient.location_path(PROJECT_ID_TEST, LOC_ID_TEST)
        product_set = ProductSet()
        # When
        result = self.hook.create_product_set(
            location=LOC_ID_TEST, product_set_id=None, product_set=product_set, project_id=PROJECT_ID_TEST
        )
        # Then
        # ProductSet ID was not provided in the method call above. Should be extracted from the API response
        # and returned.
        self.assertEqual(result, autogenerated_id)
        create_product_set_method.assert_called_once_with(
            parent=parent,
            product_set=product_set,
            product_set_id=None,
            retry=None,
            timeout=None,
            metadata=None,
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_create_productset_autogenerated_id_wrong_api_response(self, get_conn):
        # Given
        response_product_set = None
        create_product_set_method = get_conn.return_value.create_product_set
        create_product_set_method.return_value = response_product_set
        parent = ProductSearchClient.location_path(PROJECT_ID_TEST, LOC_ID_TEST)
        product_set = ProductSet()
        # When
        with self.assertRaises(AirflowException) as cm:
            self.hook.create_product_set(
                location=LOC_ID_TEST,
                product_set_id=None,
                product_set=product_set,
                project_id=PROJECT_ID_TEST,
                retry=None,
                timeout=None,
                metadata=None,
            )
        # Then
        # API response was wrong (None) and thus ProductSet ID extraction should fail.
        err = cm.exception
        self.assertIn('Unable to get name from response...', str(err))
        create_product_set_method.assert_called_once_with(
            parent=parent,
            product_set=product_set,
            product_set_id=None,
            retry=None,
            timeout=None,
            metadata=None,
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_get_productset(self, get_conn):
        # Given
        name = ProductSearchClient.product_set_path(PROJECT_ID_TEST, LOC_ID_TEST, PRODUCTSET_ID_TEST)
        response_product_set = ProductSet(name=name)
        get_product_set_method = get_conn.return_value.get_product_set
        get_product_set_method.return_value = response_product_set
        # When
        response = self.hook.get_product_set(
            location=LOC_ID_TEST, product_set_id=PRODUCTSET_ID_TEST, project_id=PROJECT_ID_TEST
        )
        # Then
        self.assertTrue(response)
        self.assertEqual(response, MessageToDict(response_product_set))
        get_product_set_method.assert_called_once_with(name=name, retry=None, timeout=None, metadata=None)

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_update_productset_no_explicit_name(self, get_conn):
        # Given
        product_set = ProductSet()
        update_product_set_method = get_conn.return_value.update_product_set
        update_product_set_method.return_value = product_set
        productset_name = ProductSearchClient.product_set_path(
            PROJECT_ID_TEST, LOC_ID_TEST, PRODUCTSET_ID_TEST
        )
        # When
        result = self.hook.update_product_set(
            location=LOC_ID_TEST,
            product_set_id=PRODUCTSET_ID_TEST,
            product_set=product_set,
            update_mask=None,
            project_id=PROJECT_ID_TEST,
            retry=None,
            timeout=None,
            metadata=None,
        )
        # Then
        self.assertEqual(result, MessageToDict(product_set))
        update_product_set_method.assert_called_once_with(
            product_set=ProductSet(name=productset_name),
            metadata=None,
            retry=None,
            timeout=None,
            update_mask=None,
        )

    @parameterized.expand([(None, None), (None, PRODUCTSET_ID_TEST), (LOC_ID_TEST, None)])
    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_update_productset_no_explicit_name_and_missing_params_for_constructed_name(
        self, location, product_set_id, get_conn
    ):
        # Given
        update_product_set_method = get_conn.return_value.update_product_set
        update_product_set_method.return_value = None
        product_set = ProductSet()
        # When
        with self.assertRaises(AirflowException) as cm:
            self.hook.update_product_set(
                location=location,
                product_set_id=product_set_id,
                product_set=product_set,
                update_mask=None,
                project_id=PROJECT_ID_TEST,
                retry=None,
                timeout=None,
                metadata=None,
            )
        err = cm.exception
        self.assertTrue(err)
        self.assertIn(
            "Unable to determine the ProductSet name. Please either set the name directly in the "
            "ProductSet object or provide the `location` and `productset_id` parameters.",
            str(err),
        )
        update_product_set_method.assert_not_called()

    @parameterized.expand([(None, None), (None, PRODUCTSET_ID_TEST), (LOC_ID_TEST, None)])
    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_update_productset_explicit_name_missing_params_for_constructed_name(
        self, location, product_set_id, get_conn
    ):
        # Given
        explicit_ps_name = ProductSearchClient.product_set_path(
            PROJECT_ID_TEST_2, LOC_ID_TEST_2, PRODUCTSET_ID_TEST_2
        )
        product_set = ProductSet(name=explicit_ps_name)
        update_product_set_method = get_conn.return_value.update_product_set
        update_product_set_method.return_value = product_set
        # When
        result = self.hook.update_product_set(
            location=location,
            product_set_id=product_set_id,
            product_set=product_set,
            update_mask=None,
            project_id=PROJECT_ID_TEST,
            retry=None,
            timeout=None,
            metadata=None,
        )
        # Then
        self.assertEqual(result, MessageToDict(product_set))
        update_product_set_method.assert_called_once_with(
            product_set=ProductSet(name=explicit_ps_name),
            metadata=None,
            retry=None,
            timeout=None,
            update_mask=None,
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_update_productset_explicit_name_different_from_constructed(self, get_conn):
        # Given
        update_product_set_method = get_conn.return_value.update_product_set
        update_product_set_method.return_value = None
        explicit_ps_name = ProductSearchClient.product_set_path(
            PROJECT_ID_TEST_2, LOC_ID_TEST_2, PRODUCTSET_ID_TEST_2
        )
        product_set = ProductSet(name=explicit_ps_name)
        template_ps_name = ProductSearchClient.product_set_path(
            PROJECT_ID_TEST, LOC_ID_TEST, PRODUCTSET_ID_TEST
        )
        # When
        # Location and product_set_id are passed in addition to a ProductSet with an explicit name,
        # but both names differ (constructed != explicit).
        # Should throw AirflowException in this case.
        with self.assertRaises(AirflowException) as cm:
            self.hook.update_product_set(
                location=LOC_ID_TEST,
                product_set_id=PRODUCTSET_ID_TEST,
                product_set=product_set,
                update_mask=None,
                project_id=PROJECT_ID_TEST,
                retry=None,
                timeout=None,
                metadata=None,
            )
        err = cm.exception
        # self.assertIn("The required parameter 'project_id' is missing", str(err))
        self.assertTrue(err)
        self.assertIn(
            "The ProductSet name provided in the object ({}) is different than the name "
            "created from the input parameters ({}). Please either: 1) Remove the ProductSet "
            "name, 2) Remove the location and productset_id parameters, 3) Unify the "
            "ProductSet name and input parameters.".format(explicit_ps_name, template_ps_name),
            str(err),
        )
        update_product_set_method.assert_not_called()

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_delete_productset(self, get_conn):
        # Given
        delete_product_set_method = get_conn.return_value.delete_product_set
        delete_product_set_method.return_value = None
        name = ProductSearchClient.product_set_path(PROJECT_ID_TEST, LOC_ID_TEST, PRODUCTSET_ID_TEST)
        # When
        response = self.hook.delete_product_set(
            location=LOC_ID_TEST, product_set_id=PRODUCTSET_ID_TEST, project_id=PROJECT_ID_TEST
        )
        # Then
        self.assertIsNone(response)
        delete_product_set_method.assert_called_once_with(name=name, retry=None, timeout=None, metadata=None)

    @mock.patch(
        'airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn',
        **{'return_value.create_reference_image.return_value': REFERENCE_IMAGE_TEST}
    )
    def test_create_reference_image_explicit_id(self, get_conn):
        # Given
        create_reference_image_method = get_conn.return_value.create_reference_image

        # When
        result = self.hook.create_reference_image(
            project_id=PROJECT_ID_TEST,
            location=LOC_ID_TEST,
            product_id=PRODUCT_ID_TEST,
            reference_image=REFERENCE_IMAGE_WITHOUT_ID_NAME,
            reference_image_id=REFERENCE_IMAGE_ID_TEST,
        )
        # Then
        # Product ID was provided explicitly in the method call above, should be returned from the method
        self.assertEqual(result, REFERENCE_IMAGE_ID_TEST)
        create_reference_image_method.assert_called_once_with(
            parent=PRODUCT_NAME,
            reference_image=REFERENCE_IMAGE_WITHOUT_ID_NAME,
            reference_image_id=REFERENCE_IMAGE_ID_TEST,
            retry=None,
            timeout=None,
            metadata=None,
        )

    @mock.patch(
        'airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn',
        **{'return_value.create_reference_image.return_value': REFERENCE_IMAGE_TEST}
    )
    def test_create_reference_image_autogenerated_id(self, get_conn):
        # Given
        create_reference_image_method = get_conn.return_value.create_reference_image

        # When
        result = self.hook.create_reference_image(
            project_id=PROJECT_ID_TEST,
            location=LOC_ID_TEST,
            product_id=PRODUCT_ID_TEST,
            reference_image=REFERENCE_IMAGE_TEST,
            reference_image_id=REFERENCE_IMAGE_ID_TEST,
        )
        # Then
        # Product ID was provided explicitly in the method call above, should be returned from the method
        self.assertEqual(result, REFERENCE_IMAGE_GEN_ID_TEST)
        create_reference_image_method.assert_called_once_with(
            parent=PRODUCT_NAME,
            reference_image=REFERENCE_IMAGE_TEST,
            reference_image_id=REFERENCE_IMAGE_ID_TEST,
            retry=None,
            timeout=None,
            metadata=None,
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_add_product_to_product_set(self, get_conn):
        # Given
        add_product_to_product_set_method = get_conn.return_value.add_product_to_product_set

        # When
        self.hook.add_product_to_product_set(
            product_set_id=PRODUCTSET_ID_TEST,
            product_id=PRODUCT_ID_TEST,
            location=LOC_ID_TEST,
            project_id=PROJECT_ID_TEST,
        )
        # Then
        # Product ID was provided explicitly in the method call above, should be returned from the method
        add_product_to_product_set_method.assert_called_once_with(
            name=PRODUCTSET_NAME_TEST, product=PRODUCT_NAME_TEST, retry=None, timeout=None, metadata=None
        )

    # remove_product_from_product_set
    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_remove_product_from_product_set(self, get_conn):
        # Given
        remove_product_from_product_set_method = get_conn.return_value.remove_product_from_product_set

        # When
        self.hook.remove_product_from_product_set(
            product_set_id=PRODUCTSET_ID_TEST,
            product_id=PRODUCT_ID_TEST,
            location=LOC_ID_TEST,
            project_id=PROJECT_ID_TEST,
        )
        # Then
        # Product ID was provided explicitly in the method call above, should be returned from the method
        remove_product_from_product_set_method.assert_called_once_with(
            name=PRODUCTSET_NAME_TEST, product=PRODUCT_NAME_TEST, retry=None, timeout=None, metadata=None
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.annotator_client')
    def test_annotate_image(self, annotator_client_mock):
        # Given
        annotate_image_method = annotator_client_mock.annotate_image

        # When
        self.hook.annotate_image(request=ANNOTATE_IMAGE_REQUEST)
        # Then
        # Product ID was provided explicitly in the method call above, should be returned from the method
        annotate_image_method.assert_called_once_with(
            request=ANNOTATE_IMAGE_REQUEST, retry=None, timeout=None
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_create_product_explicit_id(self, get_conn):
        # Given
        create_product_method = get_conn.return_value.create_product
        create_product_method.return_value = None
        parent = ProductSearchClient.location_path(PROJECT_ID_TEST, LOC_ID_TEST)
        product = Product()
        # When
        result = self.hook.create_product(
            location=LOC_ID_TEST, product_id=PRODUCT_ID_TEST, product=product, project_id=PROJECT_ID_TEST
        )
        # Then
        # Product ID was provided explicitly in the method call above, should be returned from the method
        self.assertEqual(result, PRODUCT_ID_TEST)
        create_product_method.assert_called_once_with(
            parent=parent,
            product=product,
            product_id=PRODUCT_ID_TEST,
            retry=None,
            timeout=None,
            metadata=None,
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_create_product_autogenerated_id(self, get_conn):
        # Given
        autogenerated_id = 'autogen-p-id'
        response_product = Product(
            name=ProductSearchClient.product_path(PROJECT_ID_TEST, LOC_ID_TEST, autogenerated_id)
        )
        create_product_method = get_conn.return_value.create_product
        create_product_method.return_value = response_product
        parent = ProductSearchClient.location_path(PROJECT_ID_TEST, LOC_ID_TEST)
        product = Product()
        # When
        result = self.hook.create_product(
            location=LOC_ID_TEST, product_id=None, product=product, project_id=PROJECT_ID_TEST
        )
        # Then
        # Product ID was not provided in the method call above. Should be extracted from the API response
        # and returned.
        self.assertEqual(result, autogenerated_id)
        create_product_method.assert_called_once_with(
            parent=parent, product=product, product_id=None, retry=None, timeout=None, metadata=None
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_create_product_autogenerated_id_wrong_name_in_response(self, get_conn):
        # Given
        wrong_name = 'wrong_name_not_a_correct_path'
        response_product = Product(name=wrong_name)
        create_product_method = get_conn.return_value.create_product
        create_product_method.return_value = response_product
        parent = ProductSearchClient.location_path(PROJECT_ID_TEST, LOC_ID_TEST)
        product = Product()
        # When
        with self.assertRaises(AirflowException) as cm:
            self.hook.create_product(
                location=LOC_ID_TEST, product_id=None, product=product, project_id=PROJECT_ID_TEST
            )
        # Then
        # API response was wrong (wrong name format) and thus ProductSet ID extraction should fail.
        err = cm.exception
        self.assertIn('Unable to get id from name', str(err))
        create_product_method.assert_called_once_with(
            parent=parent, product=product, product_id=None, retry=None, timeout=None, metadata=None
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_create_product_autogenerated_id_wrong_api_response(self, get_conn):
        # Given
        response_product = None
        create_product_method = get_conn.return_value.create_product
        create_product_method.return_value = response_product
        parent = ProductSearchClient.location_path(PROJECT_ID_TEST, LOC_ID_TEST)
        product = Product()
        # When
        with self.assertRaises(AirflowException) as cm:
            self.hook.create_product(
                location=LOC_ID_TEST, product_id=None, product=product, project_id=PROJECT_ID_TEST
            )
        # Then
        # API response was wrong (None) and thus ProductSet ID extraction should fail.
        err = cm.exception
        self.assertIn('Unable to get name from response...', str(err))
        create_product_method.assert_called_once_with(
            parent=parent, product=product, product_id=None, retry=None, timeout=None, metadata=None
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_update_product_no_explicit_name(self, get_conn):
        # Given
        product = Product()
        update_product_method = get_conn.return_value.update_product
        update_product_method.return_value = product
        product_name = ProductSearchClient.product_path(PROJECT_ID_TEST, LOC_ID_TEST, PRODUCT_ID_TEST)
        # When
        result = self.hook.update_product(
            location=LOC_ID_TEST,
            product_id=PRODUCT_ID_TEST,
            product=product,
            update_mask=None,
            project_id=PROJECT_ID_TEST,
            retry=None,
            timeout=None,
            metadata=None,
        )
        # Then
        self.assertEqual(result, MessageToDict(product))
        update_product_method.assert_called_once_with(
            product=Product(name=product_name), metadata=None, retry=None, timeout=None, update_mask=None
        )

    @parameterized.expand([(None, None), (None, PRODUCT_ID_TEST), (LOC_ID_TEST, None)])
    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_update_product_no_explicit_name_and_missing_params_for_constructed_name(
        self, location, product_id, get_conn
    ):
        # Given
        update_product_method = get_conn.return_value.update_product
        update_product_method.return_value = None
        product = Product()
        # When
        with self.assertRaises(AirflowException) as cm:
            self.hook.update_product(
                location=location,
                product_id=product_id,
                product=product,
                update_mask=None,
                project_id=PROJECT_ID_TEST,
                retry=None,
                timeout=None,
                metadata=None,
            )
        err = cm.exception
        self.assertTrue(err)
        self.assertIn(
            "Unable to determine the Product name. Please either set the name directly in the "
            "Product object or provide the `location` and `product_id` parameters.",
            str(err),
        )
        update_product_method.assert_not_called()

    @parameterized.expand([(None, None), (None, PRODUCT_ID_TEST), (LOC_ID_TEST, None)])
    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_update_product_explicit_name_missing_params_for_constructed_name(
        self, location, product_id, get_conn
    ):
        # Given
        explicit_p_name = ProductSearchClient.product_path(
            PROJECT_ID_TEST_2, LOC_ID_TEST_2, PRODUCT_ID_TEST_2
        )
        product = Product(name=explicit_p_name)
        update_product_method = get_conn.return_value.update_product
        update_product_method.return_value = product
        # When
        result = self.hook.update_product(
            location=location,
            product_id=product_id,
            product=product,
            update_mask=None,
            project_id=PROJECT_ID_TEST,
            retry=None,
            timeout=None,
            metadata=None,
        )
        # Then
        self.assertEqual(result, MessageToDict(product))
        update_product_method.assert_called_once_with(
            product=Product(name=explicit_p_name), metadata=None, retry=None, timeout=None, update_mask=None
        )

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_update_product_explicit_name_different_from_constructed(self, get_conn):
        # Given
        update_product_method = get_conn.return_value.update_product
        update_product_method.return_value = None
        explicit_p_name = ProductSearchClient.product_path(
            PROJECT_ID_TEST_2, LOC_ID_TEST_2, PRODUCT_ID_TEST_2
        )
        product = Product(name=explicit_p_name)
        template_p_name = ProductSearchClient.product_path(PROJECT_ID_TEST, LOC_ID_TEST, PRODUCT_ID_TEST)
        # When
        # Location and product_id are passed in addition to a Product with an explicit name,
        # but both names differ (constructed != explicit).
        # Should throw AirflowException in this case.
        with self.assertRaises(AirflowException) as cm:
            self.hook.update_product(
                location=LOC_ID_TEST,
                product_id=PRODUCT_ID_TEST,
                product=product,
                update_mask=None,
                project_id=PROJECT_ID_TEST,
                retry=None,
                timeout=None,
                metadata=None,
            )
        err = cm.exception
        self.assertTrue(err)
        self.assertIn(
            "The Product name provided in the object ({}) is different than the name created from the input "
            "parameters ({}). Please either: 1) Remove the Product name, 2) Remove the location and product_"
            "id parameters, 3) Unify the Product name and input parameters.".format(
                explicit_p_name, template_p_name
            ),
            str(err),
        )
        update_product_method.assert_not_called()

    @mock.patch('airflow.contrib.hooks.gcp_vision_hook.CloudVisionHook.get_conn')
    def test_delete_product(self, get_conn):
        # Given
        delete_product_method = get_conn.return_value.delete_product
        delete_product_method.return_value = None
        name = ProductSearchClient.product_path(PROJECT_ID_TEST, LOC_ID_TEST, PRODUCT_ID_TEST)
        # When
        response = self.hook.delete_product(
            location=LOC_ID_TEST, product_id=PRODUCT_ID_TEST, project_id=PROJECT_ID_TEST
        )
        # Then
        self.assertIsNone(response)
        delete_product_method.assert_called_once_with(name=name, retry=None, timeout=None, metadata=None)
