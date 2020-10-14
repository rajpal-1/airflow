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
import inspect
import os
import shutil
import textwrap
import unittest
import sys
from datetime import datetime, timezone
from enum import Enum
from tempfile import NamedTemporaryFile, mkdtemp
from unittest.mock import patch
from zipfile import ZipFile

from freezegun import freeze_time
from sqlalchemy import func

import airflow.example_dags
from airflow import models
from airflow.models import DagBag, DagModel
from airflow.models.serialized_dag import SerializedDagModel
from airflow.utils.dates import timezone as tz
from airflow.utils.session import create_session
from tests import cluster_policies
from tests.models import TEST_DAGS_FOLDER
from tests.test_utils import db
from tests.test_utils.asserts import assert_queries_count
from tests.test_utils.config import conf_vars

class DependencyPosition(Enum):
    BEFORE = 1
    AFTER = 2

class TestDagBag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.empty_dir = mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.empty_dir)

    def setUp(self) -> None:
        db.clear_db_dags()
        db.clear_db_serialized_dags()

    def tearDown(self) -> None:
        db.clear_db_dags()
        db.clear_db_serialized_dags()

    def test_get_existing_dag(self):
        """
        Test that we're able to parse some example DAGs and retrieve them
        """
        dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=True)

        some_expected_dag_ids = ["example_bash_operator",
                                 "example_branch_operator"]

        for dag_id in some_expected_dag_ids:
            dag = dagbag.get_dag(dag_id)

            self.assertIsNotNone(dag)
            self.assertEqual(dag_id, dag.dag_id)

        self.assertGreaterEqual(dagbag.size(), 7)

    def test_get_non_existing_dag(self):
        """
        test that retrieving a non existing dag id returns None without crashing
        """
        dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)

        non_existing_dag_id = "non_existing_dag_id"
        self.assertIsNone(dagbag.get_dag(non_existing_dag_id))

    def test_dont_load_example(self):
        """
        test that the example are not loaded
        """
        dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)

        self.assertEqual(dagbag.size(), 0)

    def test_safe_mode_heuristic_match(self):
        """With safe mode enabled, a file matching the discovery heuristics
        should be discovered.
        """
        with NamedTemporaryFile(dir=self.empty_dir, suffix=".py") as f:
            f.write(b"# airflow")
            f.write(b"# DAG")
            f.flush()

            with conf_vars({('core', 'dags_folder'): self.empty_dir}):
                dagbag = models.DagBag(include_examples=False, safe_mode=True)

            self.assertEqual(len(dagbag.dagbag_stats), 1)
            self.assertEqual(
                dagbag.dagbag_stats[0].file,
                "/{}".format(os.path.basename(f.name)))

    def test_safe_mode_heuristic_mismatch(self):
        """With safe mode enabled, a file not matching the discovery heuristics
        should not be discovered.
        """
        with NamedTemporaryFile(dir=self.empty_dir, suffix=".py"):
            with conf_vars({('core', 'dags_folder'): self.empty_dir}):
                dagbag = models.DagBag(include_examples=False, safe_mode=True)
            self.assertEqual(len(dagbag.dagbag_stats), 0)

    def test_safe_mode_disabled(self):
        """With safe mode disabled, an empty python file should be discovered.
        """
        with NamedTemporaryFile(dir=self.empty_dir, suffix=".py") as f:
            with conf_vars({('core', 'dags_folder'): self.empty_dir}):
                dagbag = models.DagBag(include_examples=False, safe_mode=False)
            self.assertEqual(len(dagbag.dagbag_stats), 1)
            self.assertEqual(
                dagbag.dagbag_stats[0].file,
                "/{}".format(os.path.basename(f.name)))

    def test_process_file_that_contains_multi_bytes_char(self):
        """
        test that we're able to parse file that contains multi-byte char
        """
        f = NamedTemporaryFile()
        f.write('\u3042'.encode())  # write multi-byte char (hiragana)
        f.flush()

        dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)
        self.assertEqual([], dagbag.process_file(f.name))

    def test_zip_skip_log(self):
        """
        test the loading of a DAG from within a zip file that skips another file because
        it doesn't have "airflow" and "DAG"
        """
        with self.assertLogs() as cm:
            test_zip_path = os.path.join(TEST_DAGS_FOLDER, "test_zip.zip")
            dagbag = models.DagBag(dag_folder=test_zip_path, include_examples=False)

            self.assertTrue(dagbag.has_logged)
            self.assertIn(
                f'INFO:airflow.models.dagbag.DagBag:File {test_zip_path}:file_no_airflow_dag.py '
                'assumed to contain no DAGs. Skipping.',
                cm.output
            )

    def test_zip(self):
        """
        test the loading of a DAG within a zip file that includes dependencies
        """
        dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)
        dagbag.process_file(os.path.join(TEST_DAGS_FOLDER, "test_zip.zip"))
        self.assertTrue(dagbag.get_dag("test_zip_dag"))

    def _generate_flat_test_dag_zip(self, path: str, dag_id: str, version_lib: str, dependency_position: DependencyPosition) -> str:
        """
        generate a DAG zip file with a dependency and an import
        """
        dagpack_name = os.path.join(path, "test-dag-%s-with-%s.zip" % (dag_id, version_lib))
        with ZipFile(dagpack_name, mode='w') as zip_file:
            if dependency_position == DependencyPosition.BEFORE:
                with zip_file.open("dag_dependency.py", mode='w') as dep:
                    dep.write(("version = '%s'\n" % version_lib).encode())

            with zip_file.open("dag.py", mode='w') as dag:
                dag.write(("from airflow.models import DAG\n").encode())
                dag.write(("from airflow.operators.dummy_operator import DummyOperator\n").encode())
                dag.write(("from airflow.utils.log.logging_mixin import LoggingMixin\n").encode())
                dag.write(("import dag_dependency\n").encode())
                dag.write(("from datetime import datetime\n").encode())
                dag.write(("\n").encode())
                dag.write(("log = LoggingMixin().log\n").encode())
                dag.write(("log.info('dag_dependency.version=%s' % dag_dependency.version)\n").encode())
                dag.write(("\n").encode())
                dag.write(("DEFAULT_DATE = datetime(2020, 1, 1)\n").encode())
                dag.write(("\n").encode())

                dag.write(("dag = DAG(dag_id='%s', description=dag_dependency.version, start_date=DEFAULT_DATE)\n" % dag_id).encode())
                dag.write(("task = DummyOperator(task_id='test_task', dag=dag)\n").encode())

            if dependency_position == DependencyPosition.AFTER:
                with zip_file.open("dag_dependency.py", mode='w') as dep:
                    dep.write(("version = '%s'\n" % version_lib).encode())
            # please note that we write dag_dependency.py exactly once within the zip archive.

        return dagpack_name

    def _test_dag_zip_isolate_flat_local_modules_inner(self, dependency_position: DependencyPosition):
        """
        test that in case multiple zips provide a library file, each DAG receives its local copy
        """
        test_dir = mkdtemp()
        try:
            dagpack_a = self._generate_flat_test_dag_zip(test_dir, "dag-a", "dep-version-a", dependency_position)
            dagpack_b = self._generate_flat_test_dag_zip(test_dir, "dag-b", "dep-version-b", dependency_position)
            dagpack_c = self._generate_flat_test_dag_zip(test_dir, "dag-c", "dep-version-c", dependency_position)

            dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)
            dagbag.process_file(dagpack_a)
            dagbag.process_file(dagpack_b)
            dagbag.process_file(dagpack_c)

            # if any of these asserts fail, then the corresponding dagback received a different "dag_dependency" module
            # than it expected to receive:
            dag_a = dagbag.get_dag('dag-a')
            assert dag_a.description == "dep-version-a"
            dag_b = dagbag.get_dag('dag-b')
            assert dag_b.description == "dep-version-b"
            dag_c = dagbag.get_dag('dag-c')
            assert dag_c.description == "dep-version-c"

        finally:
            shutil.rmtree(test_dir)

    def test_dag_zip_isolate_flat_local_modules_inner_dep_before(self):
        """
        test that in case multiple zips provide a library file, each DAG receives its local copy

        Case where the dependency modules are located in the zip file BEFORE the dag modules
        """
        self._test_dag_zip_isolate_flat_local_modules_inner(DependencyPosition.BEFORE)

    def test_dag_zip_isolate_flat_local_modules_inner_dep_after(self):
        """
        test that in case multiple zips provide a library file, each DAG receives its local copy

        Case where the dependency modules are located in the zip file AFTER the dag modules
        """
        self._test_dag_zip_isolate_flat_local_modules_inner(DependencyPosition.AFTER)


    def _test_dag_zip_flat_dependency_overrides_external_inner(self, dependency_position: DependencyPosition):
        test_dir = mkdtemp()
        original_modules = set(sys.modules.keys())
        original_path = [x for x in sys.path]
        try:
            with open(os.path.join(test_dir, "dag_dependency.py"), mode='w') as dep:
                dep.write("version = 'EXTERNAL-FLAT'\n")
            sys.path.insert(0, test_dir)
            import dag_dependency # this imports the file we just wrote here...
            del sys.path[:1] # this is critical to this test variant: we ensure here the test_dir isn't considered SYSTEM

            assert dag_dependency.version == "EXTERNAL-FLAT" # ... which we prove here

            dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)

            dagpack_a = self._generate_flat_test_dag_zip(test_dir, "dag-a", "dep-version-a", dependency_position)
            dagpack_b = self._generate_flat_test_dag_zip(test_dir, "dag-b", "dep-version-b", dependency_position)

            dagbag.process_file(dagpack_a)
            dagbag.process_file(dagpack_b)

            dag_a = dagbag.get_dag('dag-a')
            assert dag_a.description == "dep-version-a" # successfully overrides 'EXTERNAL-FLAT'
            dag_b = dagbag.get_dag('dag-b')
            assert dag_b.description == "dep-version-b" # successfully overrides 'EXTERNAL-FLAT'

        finally:
            extra_modules = set(sys.modules.keys()) - original_modules
            for extra_module in extra_modules:
                del sys.modules[extra_module]
            sys.path = original_path
            shutil.rmtree(test_dir)

    def test_dag_zip_flat_dependency_overrides_external_before(self):
        """
        test that in case a zip provides a dependency that is already loaded from "not the system" (i.e. somewhere
        that doesn't stay in sys.path beyond loading that module) then that exernally-provided dependency is ignored
        and the dag-provided dependency takes precedence

        This case handles the case where the dependency modules are in the zip file BEFORE the DAG module(s)
        """
        self._test_dag_zip_flat_dependency_overrides_external_inner(DependencyPosition.BEFORE)

    def test_dag_zip_flat_dependency_overrides_external_after(self):
        """
        test that in case a zip provides a dependency that is already loaded from "not the system" (i.e. somewhere
        that doesn't stay in sys.path beyond loading that module) then that exernally-provided dependency is ignored
        and the dag-provided dependency takes precedence

        This case handles the case where the dependency modules are in the zip file AFTER the DAG module(s)
        """
        self._test_dag_zip_flat_dependency_overrides_system_inner(DependencyPosition.AFTER)

    def _test_dag_zip_flat_dependency_overrides_system_inner(self, dependency_position: DependencyPosition):
        test_dir = mkdtemp()
        original_modules = set(sys.modules.keys())
        original_path = [x for x in sys.path]
        try:
            with open(os.path.join(test_dir, "dag_dependency.py"), mode='w') as dep:
                dep.write("version = 'SYSTEM-FLAT'\n")
            sys.path.insert(0, test_dir)
            import dag_dependency # this imports the file we just wrote here...
            assert dag_dependency.version == "SYSTEM-FLAT" # ... which we prove here

            assert "dag_dependency" in list(sys.modules.keys())

            # Note that "test_dir" stays in the path at the creation of the DagBag. It is therefore considered "system"

            dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)

            from airflow.utils.log.logging_mixin import LoggingMixin
            log = LoggingMixin().log

            log.warning("about to generate dagbags")

            dagpack_a = self._generate_flat_test_dag_zip(test_dir, "dag-a", "dep-version-a", dependency_position)
            dagpack_b = self._generate_flat_test_dag_zip(test_dir, "dag-b", "dep-version-b", dependency_position)

            log.warning("about to load dagpack A")
            dagbag.process_file(dagpack_a)
            log.warning("about to load dagpack B")
            dagbag.process_file(dagpack_b)

            dag_a = dagbag.get_dag('dag-a')
            assert dag_a.description == "dep-version-a" # overriding the system-provided module
            dag_b = dagbag.get_dag('dag-b')
            assert dag_b.description == "dep-version-b" # overriding the system-provided module

        finally:
            extra_modules = set(sys.modules.keys()) - original_modules
            for extra_module in extra_modules:
                del sys.modules[extra_module]
            sys.path = original_path
            shutil.rmtree(test_dir)

    def test_dag_zip_flat_dependency_overrides_system_before(self):
        """
        test that in case a zip provides a dependency that is already loaded (at the site-packages level) then
        the system-level dependency is ignored and the dag-provided dependency takes precedence

        This case handles the case where the dependency modules are in the zip file BEFORE the DAG module(s)
        """
        self._test_dag_zip_flat_dependency_overrides_system_inner(DependencyPosition.BEFORE)

    def test_dag_zip_flat_dependency_overrides_system_after(self):
        """
        test that in case a zip provides a dependency that is already loaded (at the site-packages level) then
        the system-level dependency is ignored and the dag-provided dependency takes precedence

        This case handles the case where the dependency modules are in the zip file AFTER the DAG module(s)
        """
        self._test_dag_zip_flat_dependency_overrides_system_inner(DependencyPosition.AFTER)

    def _generate_nested_test_dag_zip(self, path: str, dag_id: str, version_lib: str) -> str:
        """
        generate a DAG zip file with a dependency and an import
        """
        dagpack_name = os.path.join(path, "test-dag-%s-with-%s.zip" % (dag_id, version_lib))
        with ZipFile(dagpack_name, mode='w') as zip_file:
            with zip_file.open("dag_module/dag_dependency.py", mode='w') as dep:
                dep.write(("version = '%s'\n" % version_lib).encode())
            with zip_file.open("dag_module/__init__.py", mode='w') as init:
                init.write(("# empty\n").encode())

            with zip_file.open("dag.py", mode='w') as dag:
                dag.write(("from airflow.models import DAG\n").encode())
                dag.write(("from airflow.operators.dummy_operator import DummyOperator\n").encode())
                dag.write(("from dag_module.dag_dependency import version\n").encode())
                dag.write(("from datetime import datetime\n").encode())
                dag.write(("\n").encode())
                dag.write(("DEFAULT_DATE = datetime(2020, 1, 1)\n").encode())
                dag.write(("\n").encode())

                dag.write(("dag = DAG(dag_id='%s', description=version, start_date=DEFAULT_DATE)\n" % dag_id).encode())
                dag.write(("task = DummyOperator(task_id='test_task', dag=dag)\n").encode())

        return dagpack_name


    def test_dag_zip_isolate_nested_local_modules(self):
        """
        test that in case multiple zips provide a library file, each DAG receives its local copy
        """
        test_dir = mkdtemp()
        try:
            dagpack_a = self._generate_nested_test_dag_zip(test_dir, "dag-a", "dep-version-a")
            dagpack_b = self._generate_nested_test_dag_zip(test_dir, "dag-b", "dep-version-b")
            dagpack_c = self._generate_nested_test_dag_zip(test_dir, "dag-c", "dep-version-c")

            dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)
            dagbag.process_file(dagpack_a)
            dagbag.process_file(dagpack_b)
            dagbag.process_file(dagpack_c)

            # if any of these asserts fail, then the corresponding dagback received a different "dag_dependency" module
            # than it expected to receive:
            dag_a = dagbag.get_dag('dag-a')
            assert dag_a.description == "dep-version-a"
            dag_b = dagbag.get_dag('dag-b')
            assert dag_b.description == "dep-version-b"
            dag_c = dagbag.get_dag('dag-c')
            assert dag_c.description == "dep-version-c"

        finally:
            shutil.rmtree(test_dir)


    def test_dag_zip_nested_dependency_overrides_system(self):
        """
        test that in case a zip provides a dependency that is already loaded (at the site-packages level) then
        the system-level dependency is ignored and the dag-provided dependency takes precedence
        """
        test_dir = mkdtemp()
        original_modules = set(sys.modules.keys())
        original_path = sys.path
        try:
            os.mkdir(os.path.join(test_dir, "dag_module"))
            with open(os.path.join(test_dir, "dag_module", "dag_dependency.py"), mode='w') as dep:
                dep.write("version = 'SYSTEM-NESTED'\n")
            with open(os.path.join(test_dir, "dag_module", "__init__.py"), mode='w') as init:
                init.write("# empty\n")

            dagpack_a = self._generate_nested_test_dag_zip(test_dir, "dag-a", "dep-version-a")
            dagpack_b = self._generate_nested_test_dag_zip(test_dir, "dag-b", "dep-version-b")

            sys.path.insert(0, test_dir)
            import dag_module
            from dag_module.dag_dependency import version # this imports the file we just wrote here
            assert version == "SYSTEM-NESTED" # prove we imported our module into the 'system' path

            dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)
            dagbag.process_file(dagpack_a)
            dagbag.process_file(dagpack_b)

            dag_a = dagbag.get_dag('dag-a')
            assert dag_a.description == "dep-version-a" # successfully overrides 'SYSTEM'
            dag_b = dagbag.get_dag('dag-b')
            assert dag_b.description == "dep-version-b" # successfully overrides 'SYSTEM' and ignore the version in "dag-a"

        finally:
            extra_modules = set(sys.modules.keys()) - original_modules
            for extra_module in extra_modules:
                del sys.modules[extra_module]
            sys.path = original_path
            shutil.rmtree(test_dir)

    def test_process_file_cron_validity_check(self):
        """
        test if an invalid cron expression
        as schedule interval can be identified
        """
        invalid_dag_files = ["test_invalid_cron.py", "test_zip_invalid_cron.zip"]
        dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)

        self.assertEqual(len(dagbag.import_errors), 0)
        for file in invalid_dag_files:
            dagbag.process_file(os.path.join(TEST_DAGS_FOLDER, file))
        self.assertEqual(len(dagbag.import_errors), len(invalid_dag_files))
        self.assertEqual(len(dagbag.dags), 0)

    @patch.object(DagModel, 'get_current')
    def test_get_dag_without_refresh(self, mock_dagmodel):
        """
        Test that, once a DAG is loaded, it doesn't get refreshed again if it
        hasn't been expired.
        """
        dag_id = 'example_bash_operator'

        mock_dagmodel.return_value = DagModel()
        mock_dagmodel.return_value.last_expired = None
        mock_dagmodel.return_value.fileloc = 'foo'

        class _TestDagBag(models.DagBag):
            process_file_calls = 0

            def process_file(self, filepath, only_if_updated=True, safe_mode=True):
                if os.path.basename(filepath) == 'example_bash_operator.py':
                    _TestDagBag.process_file_calls += 1
                super().process_file(filepath, only_if_updated, safe_mode)

        dagbag = _TestDagBag(include_examples=True)
        dagbag.process_file_calls

        # Should not call process_file again, since it's already loaded during init.
        self.assertEqual(1, dagbag.process_file_calls)
        self.assertIsNotNone(dagbag.get_dag(dag_id))
        self.assertEqual(1, dagbag.process_file_calls)

    def test_get_dag_fileloc(self):
        """
        Test that fileloc is correctly set when we load example DAGs,
        specifically SubDAGs and packaged DAGs.
        """
        dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=True)
        dagbag.process_file(os.path.join(TEST_DAGS_FOLDER, "test_zip.zip"))

        expected = {
            'example_bash_operator': 'airflow/example_dags/example_bash_operator.py',
            'example_subdag_operator': 'airflow/example_dags/example_subdag_operator.py',
            'example_subdag_operator.section-1': 'airflow/example_dags/subdags/subdag.py',
            'test_zip_dag': 'dags/test_zip.zip/test_zip.py'
        }

        for dag_id, path in expected.items():
            dag = dagbag.get_dag(dag_id)
            self.assertTrue(dag.fileloc.endswith(path))

    @patch.object(DagModel, "get_current")
    def test_refresh_py_dag(self, mock_dagmodel):
        """
        Test that we can refresh an ordinary .py DAG
        """
        example_dags_folder = airflow.example_dags.__path__[0]

        dag_id = "example_bash_operator"
        fileloc = os.path.realpath(
            os.path.join(example_dags_folder, "example_bash_operator.py")
        )

        mock_dagmodel.return_value = DagModel()
        mock_dagmodel.return_value.last_expired = datetime.max.replace(
            tzinfo=timezone.utc
        )
        mock_dagmodel.return_value.fileloc = fileloc

        class _TestDagBag(DagBag):
            process_file_calls = 0

            def process_file(self, filepath, only_if_updated=True, safe_mode=True):
                if filepath == fileloc:
                    _TestDagBag.process_file_calls += 1
                return super().process_file(filepath, only_if_updated, safe_mode)

        dagbag = _TestDagBag(dag_folder=self.empty_dir, include_examples=True)

        self.assertEqual(1, dagbag.process_file_calls)
        dag = dagbag.get_dag(dag_id)
        self.assertIsNotNone(dag)
        self.assertEqual(dag_id, dag.dag_id)
        self.assertEqual(2, dagbag.process_file_calls)

    @patch.object(DagModel, "get_current")
    def test_refresh_packaged_dag(self, mock_dagmodel):
        """
        Test that we can refresh a packaged DAG
        """
        dag_id = "test_zip_dag"
        fileloc = os.path.realpath(
            os.path.join(TEST_DAGS_FOLDER, "test_zip.zip/test_zip.py")
        )

        mock_dagmodel.return_value = DagModel()
        mock_dagmodel.return_value.last_expired = datetime.max.replace(
            tzinfo=timezone.utc
        )
        mock_dagmodel.return_value.fileloc = fileloc

        class _TestDagBag(DagBag):
            process_file_calls = 0

            def process_file(self, filepath, only_if_updated=True, safe_mode=True):
                if filepath in fileloc:
                    _TestDagBag.process_file_calls += 1
                return super().process_file(filepath, only_if_updated, safe_mode)

        dagbag = _TestDagBag(dag_folder=os.path.realpath(TEST_DAGS_FOLDER), include_examples=False)

        self.assertEqual(1, dagbag.process_file_calls)
        dag = dagbag.get_dag(dag_id)
        self.assertIsNotNone(dag)
        self.assertEqual(dag_id, dag.dag_id)
        self.assertEqual(2, dagbag.process_file_calls)

    def process_dag(self, create_dag):
        """
        Helper method to process a file generated from the input create_dag function.
        """
        # write source to file
        source = textwrap.dedent(''.join(
            inspect.getsource(create_dag).splitlines(True)[1:-1]))
        f = NamedTemporaryFile()
        f.write(source.encode('utf8'))
        f.flush()

        dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)
        found_dags = dagbag.process_file(f.name)
        return dagbag, found_dags, f.name

    def validate_dags(self, expected_parent_dag, actual_found_dags, actual_dagbag,
                      should_be_found=True):
        expected_dag_ids = list(map(lambda dag: dag.dag_id, expected_parent_dag.subdags))
        expected_dag_ids.append(expected_parent_dag.dag_id)

        actual_found_dag_ids = list(map(lambda dag: dag.dag_id, actual_found_dags))

        for dag_id in expected_dag_ids:
            actual_dagbag.log.info('validating %s' % dag_id)
            self.assertEqual(
                dag_id in actual_found_dag_ids, should_be_found,
                'dag "%s" should %shave been found after processing dag "%s"' %
                (dag_id, '' if should_be_found else 'not ', expected_parent_dag.dag_id)
            )
            self.assertEqual(
                dag_id in actual_dagbag.dags, should_be_found,
                'dag "%s" should %sbe in dagbag.dags after processing dag "%s"' %
                (dag_id, '' if should_be_found else 'not ', expected_parent_dag.dag_id)
            )

    def test_load_subdags(self):
        # Define Dag to load
        def standard_subdag():
            import datetime  # pylint: disable=redefined-outer-name,reimported

            from airflow.models import DAG
            from airflow.operators.dummy_operator import DummyOperator
            from airflow.operators.subdag_operator import SubDagOperator
            dag_name = 'master'
            default_args = {
                'owner': 'owner1',
                'start_date': datetime.datetime(2016, 1, 1)
            }
            dag = DAG(
                dag_name,
                default_args=default_args)

            # master:
            #     A -> opSubDag_0
            #          master.opsubdag_0:
            #              -> subdag_0.task
            #     A -> opSubDag_1
            #          master.opsubdag_1:
            #              -> subdag_1.task

            with dag:
                def subdag_0():
                    subdag_0 = DAG('master.op_subdag_0', default_args=default_args)
                    DummyOperator(task_id='subdag_0.task', dag=subdag_0)
                    return subdag_0

                def subdag_1():
                    subdag_1 = DAG('master.op_subdag_1', default_args=default_args)
                    DummyOperator(task_id='subdag_1.task', dag=subdag_1)
                    return subdag_1

                op_subdag_0 = SubDagOperator(
                    task_id='op_subdag_0', dag=dag, subdag=subdag_0())
                op_subdag_1 = SubDagOperator(
                    task_id='op_subdag_1', dag=dag, subdag=subdag_1())

                op_a = DummyOperator(task_id='A')
                op_a.set_downstream(op_subdag_0)
                op_a.set_downstream(op_subdag_1)
            return dag

        test_dag = standard_subdag()
        # sanity check to make sure DAG.subdag is still functioning properly
        self.assertEqual(len(test_dag.subdags), 2)

        # Perform processing dag
        dagbag, found_dags, _ = self.process_dag(standard_subdag)

        # Validate correctness
        # all dags from test_dag should be listed
        self.validate_dags(test_dag, found_dags, dagbag)

        # Define Dag to load
        def nested_subdags():
            import datetime  # pylint: disable=redefined-outer-name,reimported

            from airflow.models import DAG
            from airflow.operators.dummy_operator import DummyOperator
            from airflow.operators.subdag_operator import SubDagOperator
            dag_name = 'master'
            default_args = {
                'owner': 'owner1',
                'start_date': datetime.datetime(2016, 1, 1)
            }
            dag = DAG(
                dag_name,
                default_args=default_args)

            # master:
            #     A -> op_subdag_0
            #          master.op_subdag_0:
            #              -> opSubDag_A
            #                 master.op_subdag_0.opSubdag_A:
            #                     -> subdag_a.task
            #              -> opSubdag_B
            #                 master.op_subdag_0.opSubdag_B:
            #                     -> subdag_b.task
            #     A -> op_subdag_1
            #          master.op_subdag_1:
            #              -> opSubdag_C
            #                 master.op_subdag_1.opSubdag_C:
            #                     -> subdag_c.task
            #              -> opSubDag_D
            #                 master.op_subdag_1.opSubdag_D:
            #                     -> subdag_d.task

            with dag:
                def subdag_a():
                    subdag_a = DAG(
                        'master.op_subdag_0.opSubdag_A', default_args=default_args)
                    DummyOperator(task_id='subdag_a.task', dag=subdag_a)
                    return subdag_a

                def subdag_b():
                    subdag_b = DAG(
                        'master.op_subdag_0.opSubdag_B', default_args=default_args)
                    DummyOperator(task_id='subdag_b.task', dag=subdag_b)
                    return subdag_b

                def subdag_c():
                    subdag_c = DAG(
                        'master.op_subdag_1.opSubdag_C', default_args=default_args)
                    DummyOperator(task_id='subdag_c.task', dag=subdag_c)
                    return subdag_c

                def subdag_d():
                    subdag_d = DAG(
                        'master.op_subdag_1.opSubdag_D', default_args=default_args)
                    DummyOperator(task_id='subdag_d.task', dag=subdag_d)
                    return subdag_d

                def subdag_0():
                    subdag_0 = DAG('master.op_subdag_0', default_args=default_args)
                    SubDagOperator(task_id='opSubdag_A', dag=subdag_0, subdag=subdag_a())
                    SubDagOperator(task_id='opSubdag_B', dag=subdag_0, subdag=subdag_b())
                    return subdag_0

                def subdag_1():
                    subdag_1 = DAG('master.op_subdag_1', default_args=default_args)
                    SubDagOperator(task_id='opSubdag_C', dag=subdag_1, subdag=subdag_c())
                    SubDagOperator(task_id='opSubdag_D', dag=subdag_1, subdag=subdag_d())
                    return subdag_1

                op_subdag_0 = SubDagOperator(
                    task_id='op_subdag_0', dag=dag, subdag=subdag_0())
                op_subdag_1 = SubDagOperator(
                    task_id='op_subdag_1', dag=dag, subdag=subdag_1())

                op_a = DummyOperator(task_id='A')
                op_a.set_downstream(op_subdag_0)
                op_a.set_downstream(op_subdag_1)

            return dag

        test_dag = nested_subdags()
        # sanity check to make sure DAG.subdag is still functioning properly
        self.assertEqual(len(test_dag.subdags), 6)

        # Perform processing dag
        dagbag, found_dags, _ = self.process_dag(nested_subdags)

        # Validate correctness
        # all dags from test_dag should be listed
        self.validate_dags(test_dag, found_dags, dagbag)

    def test_skip_cycle_dags(self):
        """
        Don't crash when loading an invalid (contains a cycle) DAG file.
        Don't load the dag into the DagBag either
        """

        # Define Dag to load
        def basic_cycle():
            import datetime  # pylint: disable=redefined-outer-name,reimported

            from airflow.models import DAG
            from airflow.operators.dummy_operator import DummyOperator
            dag_name = 'cycle_dag'
            default_args = {
                'owner': 'owner1',
                'start_date': datetime.datetime(2016, 1, 1)
            }
            dag = DAG(
                dag_name,
                default_args=default_args)

            # A -> A
            with dag:
                op_a = DummyOperator(task_id='A')
                op_a.set_downstream(op_a)

            return dag

        test_dag = basic_cycle()
        # sanity check to make sure DAG.subdag is still functioning properly
        self.assertEqual(len(test_dag.subdags), 0)

        # Perform processing dag
        dagbag, found_dags, file_path = self.process_dag(basic_cycle)

        # #Validate correctness
        # None of the dags should be found
        self.validate_dags(test_dag, found_dags, dagbag, should_be_found=False)
        self.assertIn(file_path, dagbag.import_errors)

        # Define Dag to load
        def nested_subdag_cycle():
            import datetime  # pylint: disable=redefined-outer-name,reimported

            from airflow.models import DAG
            from airflow.operators.dummy_operator import DummyOperator
            from airflow.operators.subdag_operator import SubDagOperator
            dag_name = 'nested_cycle'
            default_args = {
                'owner': 'owner1',
                'start_date': datetime.datetime(2016, 1, 1)
            }
            dag = DAG(
                dag_name,
                default_args=default_args)

            # cycle:
            #     A -> op_subdag_0
            #          cycle.op_subdag_0:
            #              -> opSubDag_A
            #                 cycle.op_subdag_0.opSubdag_A:
            #                     -> subdag_a.task
            #              -> opSubdag_B
            #                 cycle.op_subdag_0.opSubdag_B:
            #                     -> subdag_b.task
            #     A -> op_subdag_1
            #          cycle.op_subdag_1:
            #              -> opSubdag_C
            #                 cycle.op_subdag_1.opSubdag_C:
            #                     -> subdag_c.task -> subdag_c.task  >Invalid Loop<
            #              -> opSubDag_D
            #                 cycle.op_subdag_1.opSubdag_D:
            #                     -> subdag_d.task

            with dag:
                def subdag_a():
                    subdag_a = DAG(
                        'nested_cycle.op_subdag_0.opSubdag_A', default_args=default_args)
                    DummyOperator(task_id='subdag_a.task', dag=subdag_a)
                    return subdag_a

                def subdag_b():
                    subdag_b = DAG(
                        'nested_cycle.op_subdag_0.opSubdag_B', default_args=default_args)
                    DummyOperator(task_id='subdag_b.task', dag=subdag_b)
                    return subdag_b

                def subdag_c():
                    subdag_c = DAG(
                        'nested_cycle.op_subdag_1.opSubdag_C', default_args=default_args)
                    op_subdag_c_task = DummyOperator(
                        task_id='subdag_c.task', dag=subdag_c)
                    # introduce a loop in opSubdag_C
                    op_subdag_c_task.set_downstream(op_subdag_c_task)
                    return subdag_c

                def subdag_d():
                    subdag_d = DAG(
                        'nested_cycle.op_subdag_1.opSubdag_D', default_args=default_args)
                    DummyOperator(task_id='subdag_d.task', dag=subdag_d)
                    return subdag_d

                def subdag_0():
                    subdag_0 = DAG('nested_cycle.op_subdag_0', default_args=default_args)
                    SubDagOperator(task_id='opSubdag_A', dag=subdag_0, subdag=subdag_a())
                    SubDagOperator(task_id='opSubdag_B', dag=subdag_0, subdag=subdag_b())
                    return subdag_0

                def subdag_1():
                    subdag_1 = DAG('nested_cycle.op_subdag_1', default_args=default_args)
                    SubDagOperator(task_id='opSubdag_C', dag=subdag_1, subdag=subdag_c())
                    SubDagOperator(task_id='opSubdag_D', dag=subdag_1, subdag=subdag_d())
                    return subdag_1

                op_subdag_0 = SubDagOperator(
                    task_id='op_subdag_0', dag=dag, subdag=subdag_0())
                op_subdag_1 = SubDagOperator(
                    task_id='op_subdag_1', dag=dag, subdag=subdag_1())

                op_a = DummyOperator(task_id='A')
                op_a.set_downstream(op_subdag_0)
                op_a.set_downstream(op_subdag_1)

            return dag

        test_dag = nested_subdag_cycle()
        # sanity check to make sure DAG.subdag is still functioning properly
        self.assertEqual(len(test_dag.subdags), 6)

        # Perform processing dag
        dagbag, found_dags, file_path = self.process_dag(nested_subdag_cycle)

        # Validate correctness
        # None of the dags should be found
        self.validate_dags(test_dag, found_dags, dagbag, should_be_found=False)
        self.assertIn(file_path, dagbag.import_errors)

    def test_process_file_with_none(self):
        """
        test that process_file can handle Nones
        """
        dagbag = models.DagBag(dag_folder=self.empty_dir, include_examples=False)

        self.assertEqual([], dagbag.process_file(None))

    def test_deactivate_unknown_dags(self):
        """
        Test that dag_ids not passed into deactivate_unknown_dags
        are deactivated when function is invoked
        """
        dagbag = DagBag(include_examples=True)
        dag_id = "test_deactivate_unknown_dags"
        expected_active_dags = dagbag.dags.keys()

        model_before = DagModel(dag_id=dag_id, is_active=True)
        with create_session() as session:
            session.merge(model_before)

        models.DAG.deactivate_unknown_dags(expected_active_dags)

        after_model = DagModel.get_dagmodel(dag_id)
        self.assertTrue(model_before.is_active)
        self.assertFalse(after_model.is_active)

        # clean up
        with create_session() as session:
            session.query(DagModel).filter(DagModel.dag_id == 'test_deactivate_unknown_dags').delete()

    @patch("airflow.models.dagbag.settings.STORE_SERIALIZED_DAGS", True)
    def test_serialized_dags_are_written_to_db_on_sync(self):
        """
        Test that when dagbag.sync_to_db is called the DAGs are Serialized and written to DB
        even when dagbag.read_dags_from_db is False
        """
        with create_session() as session:
            serialized_dags_count = session.query(func.count(SerializedDagModel.dag_id)).scalar()
            self.assertEqual(serialized_dags_count, 0)

            dagbag = DagBag(
                dag_folder=os.path.join(TEST_DAGS_FOLDER, "test_example_bash_operator.py"),
                include_examples=False)
            dagbag.sync_to_db()

            self.assertFalse(dagbag.read_dags_from_db)

            new_serialized_dags_count = session.query(func.count(SerializedDagModel.dag_id)).scalar()
            self.assertEqual(new_serialized_dags_count, 1)

    @patch("airflow.models.dagbag.settings.STORE_SERIALIZED_DAGS", True)
    @patch("airflow.models.dagbag.settings.MIN_SERIALIZED_DAG_UPDATE_INTERVAL", 5)
    @patch("airflow.models.dagbag.settings.MIN_SERIALIZED_DAG_FETCH_INTERVAL", 5)
    def test_get_dag_with_dag_serialization(self):
        """
        Test that Serialized DAG is updated in DagBag when it is updated in
        Serialized DAG table after 'min_serialized_dag_fetch_interval' seconds are passed.
        """

        with freeze_time(tz.datetime(2020, 1, 5, 0, 0, 0)):
            example_bash_op_dag = DagBag(include_examples=True).dags.get("example_bash_operator")
            SerializedDagModel.write_dag(dag=example_bash_op_dag)

            dag_bag = DagBag(read_dags_from_db=True)
            ser_dag_1 = dag_bag.get_dag("example_bash_operator")
            ser_dag_1_update_time = dag_bag.dags_last_fetched["example_bash_operator"]
            self.assertEqual(example_bash_op_dag.tags, ser_dag_1.tags)
            self.assertEqual(ser_dag_1_update_time, tz.datetime(2020, 1, 5, 0, 0, 0))

        # Check that if min_serialized_dag_fetch_interval has not passed we do not fetch the DAG
        # from DB
        with freeze_time(tz.datetime(2020, 1, 5, 0, 0, 4)):
            with assert_queries_count(0):
                self.assertEqual(dag_bag.get_dag("example_bash_operator").tags, ["example"])

        # Make a change in the DAG and write Serialized DAG to the DB
        with freeze_time(tz.datetime(2020, 1, 5, 0, 0, 6)):
            example_bash_op_dag.tags += ["new_tag"]
            SerializedDagModel.write_dag(dag=example_bash_op_dag)

        # Since min_serialized_dag_fetch_interval is passed verify that calling 'dag_bag.get_dag'
        # fetches the Serialized DAG from DB
        with freeze_time(tz.datetime(2020, 1, 5, 0, 0, 8)):
            with assert_queries_count(2):
                updated_ser_dag_1 = dag_bag.get_dag("example_bash_operator")
                updated_ser_dag_1_update_time = dag_bag.dags_last_fetched["example_bash_operator"]

        self.assertCountEqual(updated_ser_dag_1.tags, ["example", "new_tag"])
        self.assertGreater(updated_ser_dag_1_update_time, ser_dag_1_update_time)

    def test_collect_dags_from_db(self):
        """DAGs are collected from Database"""
        example_dags_folder = airflow.example_dags.__path__[0]
        dagbag = DagBag(example_dags_folder)

        example_dags = dagbag.dags
        for dag in example_dags.values():
            SerializedDagModel.write_dag(dag)

        new_dagbag = DagBag(read_dags_from_db=True)
        self.assertEqual(len(new_dagbag.dags), 0)
        new_dagbag.collect_dags_from_db()
        new_dags = new_dagbag.dags
        self.assertEqual(len(example_dags), len(new_dags))
        for dag_id, dag in example_dags.items():
            serialized_dag = new_dags[dag_id]

            self.assertEqual(serialized_dag.dag_id, dag.dag_id)
            self.assertEqual(set(serialized_dag.task_dict), set(dag.task_dict))

    @patch("airflow.settings.policy", cluster_policies.cluster_policy)
    def test_cluster_policy_violation(self):
        """test that file processing results in import error when task does not
        obey cluster policy.
        """
        dag_file = os.path.join(TEST_DAGS_FOLDER, "test_missing_owner.py")

        dagbag = DagBag(dag_folder=dag_file,
                        include_smart_sensor=False,
                        include_examples=False)
        self.assertEqual(set(), set(dagbag.dag_ids))
        expected_import_errors = {
            dag_file: (
                f"""DAG policy violation (DAG ID: test_missing_owner, Path: {dag_file}):\n"""
                """Notices:\n"""
                """ * Task must have non-None non-default owner. Current value: airflow"""
            )
        }
        self.assertEqual(expected_import_errors, dagbag.import_errors)

    @patch("airflow.settings.policy", cluster_policies.cluster_policy)
    def test_cluster_policy_obeyed(self):
        """test that dag successfully imported without import errors when tasks
        obey cluster policy.
        """
        dag_file = os.path.join(TEST_DAGS_FOLDER,
                                "test_with_non_default_owner.py")

        dagbag = DagBag(dag_folder=dag_file,
                        include_examples=False,
                        include_smart_sensor=False)
        self.assertEqual({"test_with_non_default_owner"}, set(dagbag.dag_ids))

        self.assertEqual({}, dagbag.import_errors)
