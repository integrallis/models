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
package com.integrallis.models.backend.purejava.cache;

import java.util.concurrent.atomic.AtomicLong;

/** Hands out the non-zero identities that distinguish one sequence's KV storage from another's. */
final class SequenceIdentities {
  private static final AtomicLong NEXT = new AtomicLong();

  private SequenceIdentities() {}

  static long next() {
    return NEXT.incrementAndGet();
  }
}
