/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.models.backend.cuda;

/**
 * Pipeline stage a device operation belongs to.
 *
 * <p>Gate G2 requires a run to report which work ran on device <em>per format and per stage</em>,
 * so an accelerator that is present but inert is visible as such. A single global call counter
 * cannot do that: it cannot distinguish "attention never ran" from "attention ran and did not
 * help".
 */
public enum CudaStage {
  /** Multi-token projection during prompt ingestion. */
  PREFILL_PROJECTION,
  /** Single-token projection during generation. */
  DECODE_PROJECTION,
  /** Single-token grouped-query attention during generation. */
  DECODE_ATTENTION;
}
