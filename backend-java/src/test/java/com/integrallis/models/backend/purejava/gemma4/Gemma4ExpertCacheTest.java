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

import com.integrallis.models.backend.purejava.gemma4.Gemma4ExpertCache.CachePolicy;
import com.integrallis.models.backend.purejava.gemma4.Gemma4TensorLayout.ExpertWeights;
import com.integrallis.models.backend.purejava.gemma4.Gemma4TensorLayout.TensorSlice;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Gemma4ExpertCacheTest {

  @Test
  void reusesAHitWithoutReadingOrAllocatingAnotherSlot() throws IOException {
    CountingReader reader = new CountingReader(sequence(64));
    try (Gemma4ExpertCache cache = cache(reader, 2, CachePolicy.LFU)) {
      long firstAddress;
      try (Gemma4ExpertCache.Lease first = cache.acquire(0, 1)) {
        firstAddress = first.gateUp().address();
      }
      try (Gemma4ExpertCache.Lease second = cache.acquire(0, 1)) {
        assertThat(second.gateUp().address()).isEqualTo(firstAddress);
      }

      assertThat(reader.calls()).isEqualTo(2);
      assertThat(cache.stats()).isEqualTo(new Gemma4ExpertCache.Stats(1, 1, 8, 0, 0));
      assertThat(cache.allocatedSlots()).isEqualTo(1);
      assertThat(cache.allocatedBytes()).isEqualTo(8);
    }
  }

  @Test
  void lfuEvictsTheLessFrequentlyUsedExpertAndBreaksTiesByRecency() throws IOException {
    CountingReader reader = new CountingReader(sequence(64));
    try (Gemma4ExpertCache cache = cache(reader, 2, CachePolicy.LFU)) {
      acquireAndRelease(cache, 0);
      acquireAndRelease(cache, 1);
      acquireAndRelease(cache, 0);

      acquireAndRelease(cache, 2);
      acquireAndRelease(cache, 0);
      acquireAndRelease(cache, 1);

      assertThat(cache.stats().hits()).isEqualTo(2);
      assertThat(cache.stats().misses()).isEqualTo(4);
      assertThat(cache.stats().evictions()).isEqualTo(2);
      assertThat(reader.calls()).isEqualTo(8);
      assertThat(cache.allocatedSlots()).isEqualTo(2);
      assertThat(cache.allocatedBytes()).isEqualTo(16);
    }
  }

  @Test
  void weightedAcquirePreservesAnExpertUsedBySeveralPromptTokens() throws IOException {
    CountingReader reader = new CountingReader(sequence(64));
    try (Gemma4ExpertCache cache = cache(reader, 2, CachePolicy.LFU)) {
      try (Gemma4ExpertCache.Lease ignored = cache.acquire(0, 0, 5)) {
        assertThat(ignored.expert()).isZero();
      }
      try (Gemma4ExpertCache.Lease ignored = cache.acquire(0, 1, 1)) {
        assertThat(ignored.expert()).isEqualTo(1);
      }
      try (Gemma4ExpertCache.Lease ignored = cache.acquire(0, 2, 1)) {
        assertThat(ignored.expert()).isEqualTo(2);
      }
      try (Gemma4ExpertCache.Lease frequent = cache.acquire(0, 0)) {
        assertThat(frequent.expert()).isZero();
      }

      assertThat(cache.stats().hits()).isEqualTo(1);
      assertThat(cache.stats().misses()).isEqualTo(3);
      assertThat(cache.stats().evictions()).isEqualTo(1);
    }
  }

  @Test
  void aLiveLeasePreventsTheOnlySlotFromBeingReused() throws Exception {
    CountingReader reader = new CountingReader(sequence(64));
    try (Gemma4ExpertCache cache = cache(reader, 1, CachePolicy.LFU);
        var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      Gemma4ExpertCache.Lease first = cache.acquire(0, 0);
      CountDownLatch started = new CountDownLatch(1);
      Future<Integer> waiting =
          executor.submit(
              () -> {
                started.countDown();
                try (Gemma4ExpertCache.Lease second = cache.acquire(0, 1)) {
                  return second.expert();
                }
              });

      assertThat(started.await(1, TimeUnit.SECONDS)).isTrue();
      Thread.sleep(50);
      assertThat(waiting.isDone()).isFalse();

      first.close();

      assertThat(waiting.get(1, TimeUnit.SECONDS)).isEqualTo(1);
      assertThat(cache.stats().waits()).isPositive();
      assertThat(cache.stats().evictions()).isEqualTo(1);
    }
  }

  @Test
  void aFailedLoadIsNeverPublishedAsACacheHit() throws IOException {
    AtomicInteger attempts = new AtomicInteger();
    byte[] source = sequence(64);
    Gemma4ExpertLoader.PositionalReader reader =
        (destination, position) -> {
          if (attempts.getAndIncrement() == 0) {
            throw new IOException("injected read failure");
          }
          int count = destination.remaining();
          destination.put(source, Math.toIntExact(position), count);
          return count;
        };

    try (Gemma4ExpertCache cache = cache(reader, 1, CachePolicy.LFU)) {
      org.assertj.core.api.Assertions.assertThatThrownBy(() -> cache.acquire(0, 0))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("injected read failure");

      try (Gemma4ExpertCache.Lease recovered = cache.acquire(0, 0)) {
        assertThat(recovered.expert()).isZero();
      }

      assertThat(cache.stats().hits()).isZero();
      assertThat(cache.stats().misses()).isEqualTo(1);
    }
  }

  private static void acquireAndRelease(Gemma4ExpertCache cache, int expert) throws IOException {
    try (Gemma4ExpertCache.Lease ignored = cache.acquire(0, expert)) {
      assertThat(ignored.expert()).isEqualTo(expert);
    }
  }

  private static Gemma4ExpertCache cache(
      Gemma4ExpertLoader.PositionalReader reader, int slots, CachePolicy policy) {
    return new Gemma4ExpertCache(
        new Gemma4ExpertLoader(reader), 1, 3, slots, Gemma4ExpertCacheTest::weights, policy);
  }

  private static ExpertWeights weights(int layer, int expert) {
    long offset = expert * 8L;
    return new ExpertWeights(
        layer,
        expert,
        new TensorSlice("gate-up-" + expert, GgufTensorType.Q4_K, offset, 4, new long[] {32, 32}),
        new TensorSlice("down-" + expert, GgufTensorType.Q8_0, offset + 4, 4, new long[] {32, 32}));
  }

  private static byte[] sequence(int size) {
    byte[] result = new byte[size];
    for (int index = 0; index < size; index++) {
      result[index] = (byte) index;
    }
    return result;
  }

  private static final class CountingReader implements Gemma4ExpertLoader.PositionalReader {
    private final byte[] source;
    private final AtomicInteger calls = new AtomicInteger();

    private CountingReader(byte[] source) {
      this.source = source;
    }

    @Override
    public int read(ByteBuffer destination, long filePosition) {
      calls.incrementAndGet();
      int count = destination.remaining();
      destination.put(source, Math.toIntExact(filePosition), count);
      return count;
    }

    private int calls() {
      return calls.get();
    }
  }
}
