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
package com.integrallis.models.backend.purejava.cact;

/** Indicates that a `.cact` artifact violates its binary format contract. */
public final class MalformedCactException extends IllegalArgumentException {

  private static final long serialVersionUID = 1L;

  /** Creates a malformed-artifact exception with a stable diagnostic. */
  public MalformedCactException(String message) {
    super("Malformed CACT: " + message);
  }

  /** Creates a malformed-artifact exception with its underlying cause. */
  public MalformedCactException(String message, Throwable cause) {
    super("Malformed CACT: " + message, cause);
  }
}
