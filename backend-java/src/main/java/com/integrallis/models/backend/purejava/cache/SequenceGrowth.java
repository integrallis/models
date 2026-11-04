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

/**
 * How far a sequence-sized buffer grows when it has to hold one more position.
 *
 * <p>One definition, used by every cache and every score buffer in every architecture. The caches
 * here grew lazily while four architectures' own caches allocated their session's ceiling at open,
 * and that divergence is what let #357 -- Qwen3.5 4B exhausting an 8 GB heap before reading a token
 * -- survive after the shared caches had already been made lazy. A growth rule that differs per
 * architecture is one nobody can reason about, so there is no second copy of this arithmetic.
 *
 * <p>The ceiling is the session's capacity, which defaults to the model's declared context. A
 * declared context states what the model can attend to, and a caller is entitled to all of it; what
 * a session must not do is allocate it before a sequence reaches it.
 */
public final class SequenceGrowth {

  /**
   * Positions the first allocation covers.
   *
   * <p>Small on purpose: the common session is a short prompt and a short answer, and doubling
   * reaches any length in a logarithmic number of reallocations.
   */
  public static final int INITIAL_CAPACITY = 16;

  private SequenceGrowth() {}

  /**
   * The capacity to reallocate at, given what is already held and what is needed.
   *
   * <p>Doubling, clamped to {@code ceiling}, so the total copying across a sequence that advances
   * one position at a time stays linear in its length rather than quadratic.
   *
   * @param allocated positions currently held; 0 when nothing has been allocated yet
   * @param required positions that must be held after this growth
   * @param ceiling the session's capacity, which the result never exceeds
   * @return a capacity at least {@code required}, or {@code allocated} when that already covers it
   * @throws IllegalArgumentException if {@code required} exceeds {@code ceiling}
   */
  public static int nextCapacity(int allocated, int required, int ceiling) {
    if (required > ceiling) {
      throw new IllegalArgumentException(
          "required positions exceed the sequence ceiling " + ceiling + ": " + required);
    }
    if (required <= allocated) {
      return allocated;
    }
    int grown = allocated > 0 ? allocated : Math.min(ceiling, INITIAL_CAPACITY);
    while (grown < required) {
      // Halving the comparison rather than doubling it: at a declared context of 1,024,000 a
      // doubled int cannot overflow, but the ceiling is caller-supplied and this costs nothing.
      grown = grown <= ceiling / 2 ? grown * 2 : ceiling;
    }
    return grown;
  }
}
