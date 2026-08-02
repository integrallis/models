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
package com.integrallis.models.backend.purejava.gemma4;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gemma4.Gemma4TensorLayout.ExpertWeights;
import com.integrallis.models.backend.purejava.gemma4.Gemma4TensorLayout.TensorSlice;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Gemma4MappedExpertsTest {

  @Test
  void leasesExpertWeightsAsZeroCopySlicesOfTheMappedGguf() throws Exception {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment source = arena.allocate(64, 64);
      for (int index = 0; index < source.byteSize(); index++) {
        source.set(ValueLayout.JAVA_BYTE, index, (byte) index);
      }

      Gemma4MappedExperts experts =
          new Gemma4MappedExperts(source, 1, 3, Gemma4MappedExpertsTest::weights);
      try (Gemma4Experts.Lease lease = experts.acquire(0, 1, 4)) {
        assertThat(lease.gateUp().address()).isEqualTo(source.address() + 8);
        assertThat(lease.down().address()).isEqualTo(source.address() + 12);
        assertThat(lease.gateUp().get(ValueLayout.JAVA_BYTE, 0)).isEqualTo((byte) 8);
        assertThat(lease.down().get(ValueLayout.JAVA_BYTE, 0)).isEqualTo((byte) 12);
      }
      assertThat(experts.concurrentLeasesPerLayer()).isEqualTo(3);

      experts.close();
      assertThatThrownBy(() -> experts.acquire(0, 0))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("closed");
    }
  }

  @Test
  void precomputesStableExpertViewsAtLoadTime() throws Exception {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment source = arena.allocate(64, 64);
      AtomicInteger resolutions = new AtomicInteger();
      Gemma4Experts.ExpertResolver resolver =
          (layer, expert) -> {
            resolutions.incrementAndGet();
            return weights(layer, expert);
          };

      Gemma4MappedExperts experts = new Gemma4MappedExperts(source, 1, 3, resolver);
      assertThat(resolutions).hasValue(3);

      Gemma4Experts.Lease first = experts.acquire(0, 2);
      Gemma4Experts.Lease second = experts.acquire(0, 2, 8);

      assertThat(first).isSameAs(second);
      assertThat(resolutions).hasValue(3);
    }
  }

  private static ExpertWeights weights(int layer, int expert) {
    long offset = expert * 8L;
    return new ExpertWeights(
        layer,
        expert,
        new TensorSlice("gate-up-" + expert, GgufTensorType.Q4_K, offset, 4, new long[] {32, 32}),
        new TensorSlice("down-" + expert, GgufTensorType.Q8_0, offset + 4, 4, new long[] {32, 32}));
  }
}
