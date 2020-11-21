#!/usr/bin/env python
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
import argparse
import fnmatch
import os
import re
import shlex
import shutil
import sys
from collections import defaultdict
from glob import glob
from subprocess import run
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, List, Optional, Tuple

from tabulate import tabulate

from docs.exts.docs_build import dev_index_generator, lint_checks  # pylint: disable=no-name-in-module
from docs.exts.docs_build.errors import (  # pylint: disable=no-name-in-module
    DocBuildError,
    display_errors_summary,
    parse_sphinx_warnings,
)
from docs.exts.docs_build.spelling_checks import (  # pylint: disable=no-name-in-module
    SpellingError,
    display_spelling_error_summary,
    parse_spelling_warnings,
)
from docs.exts.provider_yaml_utils import load_package_data  # pylint: disable=no-name-in-module

if __name__ != "__main__":
    raise Exception(
        "This file is intended to be executed as an executable program. You cannot use it as a module."
        "To run this script, run the ./build_docs.py command"
    )

ROOT_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
ROOT_PACKAGE_DIR = os.path.join(ROOT_PROJECT_DIR, "airflow")
DOCS_DIR = os.path.join(ROOT_PROJECT_DIR, "docs")
ALL_PROVIDER_YAMLS = load_package_data()

CHANNEL_INVITATION = """\
If you need help, write to #documentation channel on Airflow's Slack.
Channel link: https://apache-airflow.slack.com/archives/CJ1LVREHX
Invitation link: https://s.apache.org/airflow-slack\
"""


class AirflowDocsBuilder:
    """Documentation builder for Airflow."""

    def __init__(self, package_name: str):
        self.package_name = package_name

    @property
    def _doctree_dir(self) -> str:
        return f"{DOCS_DIR}/_doctrees/docs/{self.package_name}"

    @property
    def _out_dir(self) -> str:
        if self.package_name == 'provider-references':
            # Disable versioning for references
            return f"{DOCS_DIR}/_build/docs/{self.package_name}"
        else:
            return f"{DOCS_DIR}/_build/docs/{self.package_name}/latest"

    @property
    def _src_dir(self) -> str:
        # TODO(mik-laj):
        #  After migrating the content from the core to providers, we should move all documentation from .
        #  to /airflow/ to keep the directory structure more maintainable.
        if self.package_name == 'apache-airflow':
            return DOCS_DIR
        elif self.package_name.startswith('apache-airflow-providers-') or (
            self.package_name == 'provider-references'
        ):
            return f"{DOCS_DIR}/{self.package_name}"
        else:
            raise Exception(F"Unsupported package: {self.package_name}")

    def clean_files(self) -> None:
        """Cleanup all artifacts generated by previous builds."""
        api_dir = os.path.join(self._src_dir, "_api")

        shutil.rmtree(api_dir, ignore_errors=True)
        shutil.rmtree(self._out_dir, ignore_errors=True)
        os.makedirs(api_dir, exist_ok=True)
        os.makedirs(self._out_dir, exist_ok=True)

        print(f"Recreated content of the {shlex.quote(self._out_dir)} and {shlex.quote(api_dir)} folders")

    def check_spelling(self):
        """Checks spelling."""
        spelling_errors = []
        with TemporaryDirectory() as tmp_dir:
            build_cmd = [
                "sphinx-build",
                "-W",  # turn warnings into errors
                "-T",  # show full traceback on exception
                "-b",  # builder to use
                "spelling",
                "-c",
                DOCS_DIR,
                "-d",  # path for the cached environment and doctree files
                self._doctree_dir,
                self._src_dir,  # path to documentation source files
                tmp_dir,
            ]
            print("Executing cmd: ", " ".join([shlex.quote(c) for c in build_cmd]))
            env = os.environ.copy()
            env['AIRFLOW_PACKAGE_NAME'] = self.package_name
            completed_proc = run(  # pylint: disable=subprocess-run-check
                build_cmd, cwd=self._src_dir, env=env
            )
            if completed_proc.returncode != 0:
                spelling_errors.append(
                    SpellingError(
                        file_path=None,
                        line_no=None,
                        spelling=None,
                        suggestion=None,
                        context_line=None,
                        message=(
                            f"Sphinx spellcheck returned non-zero exit status: {completed_proc.returncode}."
                        ),
                    )
                )
                warning_text = ""
                for filepath in glob(f"{tmp_dir}/**/*.spelling", recursive=True):
                    with open(filepath) as speeling_file:
                        warning_text += speeling_file.read()

                spelling_errors.extend(parse_spelling_warnings(warning_text, self._src_dir))
        return spelling_errors

    def build_sphinx_docs(self) -> List[DocBuildError]:
        """Build Sphinx documentation"""
        build_errors = []
        with NamedTemporaryFile() as tmp_file:
            build_cmd = [
                "sphinx-build",
                "-T",  # show full traceback on exception
                "--color",  # do emit colored output
                "-b",  # builder to use
                "html",
                "-d",  # path for the cached environment and doctree files
                self._doctree_dir,
                "-c",
                DOCS_DIR,
                "-w",  # write warnings (and errors) to given file
                tmp_file.name,
                self._src_dir,  # path to documentation source files
                self._out_dir,  # path to output directory
            ]
            print("Executing cmd: ", " ".join([shlex.quote(c) for c in build_cmd]))
            env = os.environ.copy()
            env['AIRFLOW_PACKAGE_NAME'] = self.package_name
            completed_proc = run(  # pylint: disable=subprocess-run-check
                build_cmd, cwd=self._src_dir, env=env
            )
            if completed_proc.returncode != 0:
                build_errors.append(
                    DocBuildError(
                        file_path=None,
                        line_no=None,
                        message=f"Sphinx returned non-zero exit status: {completed_proc.returncode}.",
                    )
                )
            tmp_file.seek(0)
            warning_text = tmp_file.read().decode()
            # Remove 7-bit C1 ANSI escape sequences
            warning_text = re.sub(r"\x1B[@-_][0-?]*[ -/]*[@-~]", "", warning_text)
            build_errors.extend(parse_sphinx_warnings(warning_text, self._src_dir))
        return build_errors


def get_available_packages():
    """Get list of all available packages to build."""
    provider_package_names = [provider['package-name'] for provider in ALL_PROVIDER_YAMLS]
    return ["apache-airflow", "provider-references", *provider_package_names]


def _get_parser():
    available_packages_list = " * " + "\n * ".join(get_available_packages())
    parser = argparse.ArgumentParser(
        description='Builds documentation and runs spell checking',
        epilog=f"List of supported documentation packages:\n{available_packages_list}" "",
    )
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.add_argument(
        '--disable-checks', dest='disable_checks', action='store_true', help='Disables extra checks'
    )
    parser.add_argument(
        "--package-filter",
        action="append",
        help=(
            "Filter specifying for which packages the documentation is to be built. Wildcard are supported."
        ),
    )
    parser.add_argument('--docs-only', dest='docs_only', action='store_true', help='Only build documentation')
    parser.add_argument(
        '--spellcheck-only', dest='spellcheck_only', action='store_true', help='Only perform spellchecking'
    )
    return parser


def build_docs_for_packages(
    current_packages: List[str], docs_only: bool, spellcheck_only: bool
) -> Tuple[Dict[str, List[DocBuildError]], Dict[str, List[SpellingError]]]:
    """Builds documentation for single package and returns errors"""
    all_build_errors: Dict[str, List[DocBuildError]] = defaultdict(list)
    all_spelling_errors: Dict[str, List[SpellingError]] = defaultdict(list)
    for package_name in current_packages:
        print("#" * 20, package_name, "#" * 20)
        builder = AirflowDocsBuilder(package_name=package_name)
        builder.clean_files()
        if not docs_only:
            spelling_errors = builder.check_spelling()
            if spelling_errors:
                all_spelling_errors[package_name].extend(spelling_errors)

        if not spellcheck_only:
            docs_errors = builder.build_sphinx_docs()
            if docs_errors:
                all_build_errors[package_name].extend(docs_errors)

    return all_build_errors, all_spelling_errors


def display_packages_summary(
    build_errors: Dict[str, List[DocBuildError]], spelling_errors: Dict[str, List[SpellingError]]
):
    """Displays a summary that contains information on the number of errors in each packages"""
    packages_names = {*build_errors.keys(), *spelling_errors.keys()}
    tabular_data = [
        {
            "Package name": package_name,
            "Count of doc build errors": len(build_errors.get(package_name, [])),
            "Count of spelling errors": len(spelling_errors.get(package_name, [])),
        }
        for package_name in sorted(packages_names, key=lambda k: k or '')
    ]
    print("#" * 20, "Packages errors summary", "#" * 20)
    print(tabulate(tabular_data=tabular_data, headers="keys"))
    print("#" * 50)


def print_build_errors_and_exit(
    message: str,
    build_errors: Dict[str, List[DocBuildError]],
    spelling_errors: Dict[str, List[SpellingError]],
) -> None:
    """Prints build errors and exists."""
    if build_errors or spelling_errors:
        if build_errors:
            display_errors_summary(build_errors)
            print()
        if spelling_errors:
            display_spelling_error_summary(spelling_errors)
            print()
        print(message)
        display_packages_summary(build_errors, spelling_errors)
        print()
        print(CHANNEL_INVITATION)
        sys.exit(1)


def main():
    """Main code"""
    args = _get_parser().parse_args()
    available_packages = get_available_packages()
    print("Available packages: ", available_packages)

    docs_only = args.docs_only
    spellcheck_only = args.spellcheck_only
    disable_checks = args.disable_checks
    package_filters = args.package_filter

    print("Current package filters: ", package_filters)
    current_packages = [p for p in available_packages if any(fnmatch.fnmatch(p, f) for f in package_filters)]
    print(f"Documentation will be built for {len(current_packages)} package(s): {current_packages}")

    all_build_errors: Dict[Optional[str], List[DocBuildError]] = {}
    all_spelling_errors: Dict[Optional[str], List[SpellingError]] = {}
    package_build_errors, package_spelling_errors = build_docs_for_packages(
        current_packages=current_packages,
        docs_only=docs_only,
        spellcheck_only=spellcheck_only,
    )
    if package_build_errors:
        all_build_errors.update(package_build_errors)
    if package_spelling_errors:
        all_spelling_errors.update(package_spelling_errors)
    to_retry_packages = [
        package_name
        for package_name, errors in package_build_errors.items()
        if any(
            'failed to reach any of the inventories with the following issues' in e.message for e in errors
        )
    ]
    if to_retry_packages:
        for package_name in to_retry_packages:
            if package_name in all_build_errors:
                del all_build_errors[package_name]
            if package_name in all_spelling_errors:
                del all_spelling_errors[package_name]

        package_build_errors, package_spelling_errors = build_docs_for_packages(
            current_packages=to_retry_packages,
            docs_only=docs_only,
            spellcheck_only=spellcheck_only,
        )
        if package_build_errors:
            all_build_errors.update(package_build_errors)
        if package_spelling_errors:
            all_spelling_errors.update(package_spelling_errors)

    if not disable_checks:
        general_errors = []
        general_errors.extend(lint_checks.check_guide_links_in_operator_descriptions())
        general_errors.extend(lint_checks.check_enforce_code_block())
        general_errors.extend(lint_checks.check_exampleinclude_for_example_dags())
        if general_errors:
            all_build_errors[None] = general_errors

    dev_index_generator.generate_index(f"{DOCS_DIR}/_build/index.html")
    print_build_errors_and_exit(
        "The documentation has errors.",
        all_build_errors,
        all_spelling_errors,
    )


main()
