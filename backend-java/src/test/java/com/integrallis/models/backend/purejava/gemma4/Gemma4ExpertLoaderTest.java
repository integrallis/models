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
import java.io.EOFException;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class Gemma4ExpertLoaderTest {

  @Test
  void loadsBothExpertRangesIntoOneCallerOwnedSlot(@TempDir Path tempDir) throws IOException {
    byte[] source = sequence(96);
    Path file = tempDir.resolve("model.gguf");
    Files.write(file, source);
    ExpertWeights weights = weights(17, 11, 43, 13);

    try (Arena arena = Arena.ofConfined();
        Gemma4ExpertLoader loader = Gemma4ExpertLoader.open(file)) {
      MemorySegment slot = arena.allocate(weights.totalBytes(), 64);

      Gemma4ExpertLoader.LoadedExpert loaded = loader.loadInto(weights, slot);

      assertThat(loaded.gateUp().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE))
          .containsExactly(Arrays.copyOfRange(source, 17, 28));
      assertThat(loaded.down().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE))
          .containsExactly(Arrays.copyOfRange(source, 43, 56));
      assertThat(loaded.storage()).isEqualTo(slot);
    }
  }

  @Test
  void retriesShortReadsUntilBothRangesAreComplete() throws IOException {
    byte[] source = sequence(96);
    AtomicInteger calls = new AtomicInteger();
    Gemma4ExpertLoader.PositionalReader reader =
        (destination, filePosition) -> {
          calls.incrementAndGet();
          int count = Math.min(3, destination.remaining());
          destination.put(source, Math.toIntExact(filePosition), count);
          return count;
        };
    ExpertWeights weights = weights(17, 11, 43, 13);

    try (Arena arena = Arena.ofConfined();
        Gemma4ExpertLoader loader = new Gemma4ExpertLoader(reader)) {
      Gemma4ExpertLoader.LoadedExpert loaded =
          loader.loadInto(weights, arena.allocate(weights.totalBytes(), 64));

      assertThat(loaded.gateUp().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE))
          .containsExactly(Arrays.copyOfRange(source, 17, 28));
      assertThat(loaded.down().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE))
          .containsExactly(Arrays.copyOfRange(source, 43, 56));
      assertThat(calls).hasValue(9);
    }
  }

  @Test
  void reportsTheIncompleteRangeWhenTheSourceEndsEarly() throws IOException {
    Gemma4ExpertLoader.PositionalReader reader = (destination, filePosition) -> -1;
    ExpertWeights weights = weights(17, 11, 43, 13);

    try (Arena arena = Arena.ofConfined();
        Gemma4ExpertLoader loader = new Gemma4ExpertLoader(reader)) {
      MemorySegment slot = arena.allocate(weights.totalBytes(), 64);

      assertThatThrownBy(() -> loader.loadInto(weights, slot))
          .isInstanceOf(EOFException.class)
          .hasMessageContaining("blk.0.ffn_gate_up_exps.weight")
          .hasMessageContaining("11 bytes");
    }
  }

  @Test
  void rejectsSlotsThatCannotHoldTheCompleteExpert() throws IOException {
    ExpertWeights weights = weights(17, 11, 43, 13);

    try (Arena arena = Arena.ofConfined();
        Gemma4ExpertLoader loader = new Gemma4ExpertLoader((destination, position) -> -1)) {
      MemorySegment shortSlot = arena.allocate(weights.totalBytes() - 1, 64);

      assertThatThrownBy(() -> loader.loadInto(weights, shortSlot))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("24 bytes")
          .hasMessageContaining("23");
    }
  }

  @Test
  void closesTheOwnedReader() throws IOException {
    AtomicBoolean closed = new AtomicBoolean();
    Gemma4ExpertLoader.PositionalReader reader =
        new Gemma4ExpertLoader.PositionalReader() {
          @Override
          public int read(ByteBuffer destination, long filePosition) {
            return -1;
          }

          @Override
          public void close() {
            closed.set(true);
          }
        };

    new Gemma4ExpertLoader(reader).close();

    assertThat(closed).isTrue();
  }

  private static ExpertWeights weights(
      long gateOffset, long gateBytes, long downOffset, long downBytes) {
    return new ExpertWeights(
        0,
        7,
        new TensorSlice(
            "blk.0.ffn_gate_up_exps.weight",
            GgufTensorType.Q4_K,
            gateOffset,
            gateBytes,
            new long[] {32, 32}),
        new TensorSlice(
            "blk.0.ffn_down_exps.weight",
            GgufTensorType.Q8_0,
            downOffset,
            downBytes,
            new long[] {32, 32}));
  }

  private static byte[] sequence(int size) {
    byte[] result = new byte[size];
    for (int index = 0; index < size; index++) {
      result[index] = (byte) index;
    }
    return result;
  }
}
