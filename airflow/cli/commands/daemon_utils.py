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

import signal
from argparse import Namespace
from typing import Callable

from daemon import daemon
from daemon.pidfile import TimeoutPIDLockFile

from airflow import settings
from airflow.utils.cli import setup_locations, setup_logging, sigint_handler, sigquit_handler
from airflow.utils.process_utils import check_if_pidfile_process_is_running


def run_command_with_daemon_mode(
    args: Namespace,
    process_name: str,
    callback: Callable,
    should_setup_logging: bool = False,
    umask: str = settings.DAEMON_UMASK,
    pid_file: str = None,
):
    """
    Parameters
    ----------
    args : Namespace
        The set of arguments passed to the original CLI command
    process_name : str
        Process name used in naming log and PID files for the daemon
    callback : Callable
        The actual command to run with or without daemon context
    should_setup_logging : bool
        If true, then a log file handler for the daemon process will be created
    umask : str
        File access creation mask ("umask") to set for the process on daemon start
    pid_file : str
        If specified, this file path us used to store daemon process PID.
        If not specified, a file path is generated with the default pattern.
    """
    if args.daemon:
        pid, stdout, stderr, log_file = setup_locations(
            process=process_name, pid=args.pid, stdout=args.stdout, stderr=args.stderr, log=args.log_file
        )
        if pid_file:
            pid = pid_file

        # Check if the process is already running; if not but a pidfile exists, clean it up
        check_if_pidfile_process_is_running(pid_file=pid, process_name=process_name)

        files_preserve = []
        if should_setup_logging:
            handle = setup_logging(log_file)
            files_preserve.append(handle)
        with open(stdout, "a") as stdout_handle, open(stderr, "a") as stderr_handle:
            stdout_handle.truncate(0)
            stderr_handle.truncate(0)

            ctx = daemon.DaemonContext(
                pidfile=TimeoutPIDLockFile(pid, -1),
                files_preserve=files_preserve,
                stdout=stdout_handle,
                stderr=stderr_handle,
                umask=int(umask, 8),
            )

            with ctx:
                callback()
    else:
        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGTERM, sigint_handler)
        signal.signal(signal.SIGQUIT, sigquit_handler)
        callback()
