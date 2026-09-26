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

import static org.assertj.core.api.Assertions.*;

import java.net.SocketTimeoutException;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletionException;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RoutingCancellationTest {
  @Test
  void detectsWrappedInterruptionWithoutChangingCallbackThread() {
    var failure = new CompletionException(new InterruptedException("cancelled elsewhere"));
    assertThat(RoutingCancellation.isCancellation(failure)).isTrue();
    assertThat(Thread.currentThread().isInterrupted()).isFalse();
  }

  @Test
  void doesNotTreatSocketTimeoutAsCancellation() {
    assertThat(RoutingCancellation.isCancellation(new SocketTimeoutException("provider timeout")))
        .isFalse();
  }

  @Test
  void suppressedCancellationDoesNotReplaceThePrimaryFailure() {
    var failure = new IllegalStateException("provider failure");
    failure.addSuppressed(new CancellationException("cleanup cancelled"));
    assertThat(RoutingCancellation.isCancellation(failure)).isFalse();
  }

  @Test
  void cyclicProviderCauseChainTerminates() {
    var first = new IllegalStateException("first");
    var second = new IllegalArgumentException("second", first);
    first.initCause(second);
    assertThat(RoutingCancellation.isCancellation(first)).isFalse();
    assertThat(RoutingCancellation.isInterruption(first)).isFalse();
  }
}
