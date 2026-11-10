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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class NativeVisionArithmeticTest {
  private static final Path FIXTURES = Path.of("src/main/rust/model-kernels/testdata");

  @Test
  void publicSingleAndGroupedProjectionsMatchIndependentGraphFixtures() throws Exception {
    try (var kernel = RustGgufBatchedMatrixKernel.open(library(), 3);
        var arena = Arena.ofConfined()) {
      assumeTrue(kernel.configureImagePrefill(), "Reference graph arithmetic needs AVX2/FMA/F16C");
      for (var type :
          new GgufTensorType[] {GgufTensorType.Q4_K, GgufTensorType.Q5_K, GgufTensorType.Q6_K}) {
        boolean q4 = type == GgufTensorType.Q4_K;
        Path directory =
            FIXTURES.resolve(
                q4
                    ? "q4-repacked/768"
                    : "k-quant-graph/" + (type == GgufTensorType.Q5_K ? "q5" : "q6") + "/768");
        byte[] packed = Files.readAllBytes(directory.resolve(q4 ? "weights.q4k" : "weights.bin"));
        var weights = arena.allocate(packed.length);
        weights.copyFrom(MemorySegment.ofArray(packed));
        float[] inputs = floats(directory.resolve("input.f32"));
        for (int batch : new int[] {1, 3, 8, 17}) {
          float[] input = Arrays.copyOf(inputs, batch * 768);
          float[] expected =
              floats(
                  directory.resolve(q4 ? batch + "-expected.f32" : "expected-" + batch + ".f32"));
          float[] actual = new float[batch * 8];
          kernel.multiply(actual, input, weights, type, batch, 8, 768);
          assertBits(actual, expected);
          float[] second = new float[actual.length];
          Arrays.fill(actual, Float.NaN);
          kernel.multiplyDual(
              actual, weights, type, 8, second, weights, type, 8, input, batch, 768);
          assertBits(actual, expected);
          assertBits(second, expected);
        }
      }
      assertThat(kernel.preservesPrunedBatchRow(GgufTensorType.Q4_K, 8, 7)).isFalse();
      assertThat(kernel.preservesPrunedBatchRow(GgufTensorType.Q4_K, 9, 8)).isTrue();
      assertThat(kernel.preservesPrunedBatchRow(GgufTensorType.Q5_K, 8, 7)).isFalse();
      assertThat(kernel.preservesPrunedBatchRow(GgufTensorType.Q6_K, 7, 6)).isTrue();
      assertThat(kernel.preservesPrunedBatchRow(GgufTensorType.Q8_0, 8, 7)).isTrue();
    }
    try (var ordinary = RustGgufBatchedMatrixKernel.open(library(), 3)) {
      assertThat(ordinary.preservesPrunedBatchRow(GgufTensorType.Q4_K, 8, 7)).isTrue();
    }
  }

  @Test
  void rotaryBoundaryPreservesOffsetsAndValidatesAliasingAndOwnership() {
    var dense = NativeDebertaKernels.load(library(), 2);
    var attention = new NativeCausalAttention(dense);
    float[] cosine = {7, 7, 7, 7}, sine = {8, 8, 8, 8};
    try {
      assertThat(attention.rotaryFactors(0, 2, 10000, .25f, 1, null, cosine, sine, 1)).isTrue();
      assertThat(cosine).containsExactly(7, 1, 1, 7);
      assertThat(sine).containsExactly(8, 0, 0, 8);
      assertThat(
              attention.rotaryFactors(20, 2, 10000, .25f, 2, new float[] {1, 2}, cosine, sine, 1))
          .isTrue();
      assertThat(cosine[1]).isCloseTo(2f * (float) Math.cos(5), within(1e-6f));
      assertThat(sine[2]).isCloseTo(2f * (float) Math.sin(.025), within(1e-6f));
      assertThatThrownBy(() -> attention.rotaryFactors(0, 2, 10000, 1, 1, null, cosine, cosine, 0))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> attention.rotaryFactors(0, 2, 10000, 1, 1, cosine, cosine, sine, 0))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(
              () -> attention.rotaryFactors(0, 2, 10000, 1, 1, new float[1], cosine, sine, 0))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> attention.rotaryFactors(0, 2, 10000, 1, 1, null, cosine, sine, 3))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> attention.rotaryFactors(-1, 2, 10000, 1, 1, null, cosine, sine, 0))
          .isInstanceOf(IllegalStateException.class);
    } finally {
      attention.close();
    }
    assertThatThrownBy(() -> attention.rotaryFactors(0, 2, 10000, 1, 1, null, cosine, sine, 0))
        .isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(() -> dense.rotaryFactors(0, 2, 10000, 1, 1, null, cosine, sine, 0))
        .isInstanceOf(IllegalStateException.class);
  }

  private static Path library() {
    return Path.of(System.getProperty("models.native.kernels.library"));
  }

  private static float[] floats(Path path) throws Exception {
    var buffer =
        ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    float[] values = new float[buffer.remaining()];
    buffer.get(values);
    return values;
  }

  private static void assertBits(float[] actual, float[] expected) {
    assertThat(actual).hasSameSizeAs(expected);
    for (int i = 0; i < actual.length; i++) {
      assertThat(Float.floatToRawIntBits(actual[i]))
          .as("output %s", i)
          .isEqualTo(Float.floatToRawIntBits(expected[i]));
    }
  }
}
