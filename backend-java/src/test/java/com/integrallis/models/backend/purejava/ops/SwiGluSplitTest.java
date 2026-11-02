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
package com.integrallis.models.backend.purejava.ops;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.SplittableRandom;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The scalar SwiGLU's parallel split is bit-identical to the serial loop, checked by a SHA-256 of
 * the whole output array rather than by sampled lanes. The sizes span one worker up to the
 * machine's processor count (the worker count is {@code min(processors, size / 1024)}).
 */
class SwiGluSplitTest {

  static int[] sizes() {
    return new int[] {
      1023, 1024, 2049, 3 * 1024 + 37, 896, 14_336, 18_944, 12 * 1024 + 13, 100_003
    };
  }

  @ParameterizedTest
  @MethodSource("sizes")
  void scalarSplitMatchesTheSerialLoop(int size) throws Exception {
    float[] gate = random(size, 17L * size);
    float[] up = random(size, 19L * size);
    float[] expected = new float[size];
    for (int i = 0; i < size; i++) {
      float x = gate[i];
      expected[i] = x / (1.0f + (float) Math.exp(-x)) * up[i];
    }
    float[] actual = new float[size];
    TensorOps.swiGlu(actual, 0, gate, 0, up, 0, size);
    assertThat(sha256(actual)).as("size %d", size).isEqualTo(sha256(expected));
  }

  private static float[] random(int n, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[] values = new float[n];
    for (int i = 0; i < n; i++) {
      values[i] = (float) random.nextDouble(-12.0, 12.0);
    }
    return values;
  }

  private static String sha256(float[] values) throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES);
    for (float v : values) {
      bytes.putInt(Float.floatToRawIntBits(v));
    }
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes.array()));
  }
}
