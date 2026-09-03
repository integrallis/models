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
package com.integrallis.models.router;

import java.util.List;

/** Raised after every eligible model in a fleet fails. */
public final class RoutingExecutionException extends RuntimeException {
  private static final long serialVersionUID = 1L;

  private final transient List<RoutingAttempt> attempts;

  RoutingExecutionException(List<RoutingAttempt> attempts, Throwable cause) {
    super("all " + attempts.size() + " routed model attempts failed", cause);
    this.attempts = List.copyOf(attempts);
  }

  public List<RoutingAttempt> attempts() {
    return attempts;
  }
}
