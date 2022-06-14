/*!
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

type RunState = 'success' | 'running' | 'queued' | 'failed' | 'no_status' | '';

type TaskState = RunState
| 'removed'
| 'scheduled'
| 'shutdown'
| 'restarting'
| 'up_for_retry'
| 'up_for_reschedule'
| 'upstream_failed'
| 'skipped'
| 'sensing'
| 'deferred';

interface DagRun {
  runId: string;
  runType: 'manual' | 'backfill' | 'scheduled';
  state: RunState;
  executionDate: string;
  dataIntervalStart: string;
  dataIntervalEnd: string;
  startDate: string;
  endDate: string;
}

interface GridTaskInstance {
  runId: string;
  taskId: string;
  startDate: string;
  endDate?: string;
  state: TaskState;
}

interface GridTask {
  id: string | null;
  label: string | null;
  instances: GridTaskInstance[];
  tooltip?: string;
  children?: GridTask[];
  extraLinks?: string[];
  isMapped?: boolean;
}

export type {
  DagRun,
  RunState,
  TaskState,
  GridTaskInstance,
  GridTask,
};
