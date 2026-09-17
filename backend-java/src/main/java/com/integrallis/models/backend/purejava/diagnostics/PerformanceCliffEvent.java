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
package com.integrallis.models.backend.purejava.diagnostics;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;

/**
 * JFR event committed the first time a {@link PerformanceCliff} is reported in a process.
 *
 * <p>Enable it with {@code -XX:StartFlightRecording} (it is enabled by default in the {@code
 * default} and {@code profile} settings, as every event without an explicit {@code @Enabled(false)}
 * is) or programmatically by its {@link #NAME}. The stack trace points at the branch that took the
 * slower path.
 */
@Name(PerformanceCliffEvent.NAME)
@Label("Performance Cliff")
@Category("Models")
@Description("A fast path was not taken; reported once per process per reason")
@StackTrace(true)
public final class PerformanceCliffEvent extends jdk.jfr.Event {

  /** Fully qualified JFR event name. */
  public static final String NAME = "com.integrallis.models.PerformanceCliff";

  @Label("Reason")
  @Description("Stable PerformanceCliff identifier")
  String reason;

  @Label("Detail")
  @Description("Branch-specific context such as the architecture, tensor types, or kernel")
  String detail;

  PerformanceCliffEvent(String reason, String detail) {
    this.reason = reason;
    this.detail = detail;
  }
}
