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

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.function.Predicate;

/** Shared cancellation classification for provider-neutral blocking and streaming adapters. */
public final class RoutingCancellation {
  private RoutingCancellation() {}

  /**
   * Whether a failure or its cause is a standard cancellation or interruption signal.
   *
   * <p>SDK wrappers preserve these signals through their cause chains. Cyclic cause chains are
   * tolerated. Ordinary timeouts and suppressed failures do not cancel a request. This method does
   * not modify the calling thread's interrupt flag; an asynchronous callback may run on a different
   * thread from the interrupted operation.
   */
  public static boolean isCancellation(Throwable failure) {
    return contains(
        failure,
        cause -> cause instanceof CancellationException || cause instanceof InterruptedException);
  }

  static boolean isInterruption(Throwable failure) {
    return contains(failure, cause -> cause instanceof InterruptedException);
  }

  private static boolean contains(Throwable failure, Predicate<Throwable> predicate) {
    Set<Throwable> seen = Collections.newSetFromMap(new IdentityHashMap<>());
    for (Throwable cause = failure; cause != null && seen.add(cause); cause = cause.getCause()) {
      if (predicate.test(cause)) return true;
    }
    return false;
  }
}
