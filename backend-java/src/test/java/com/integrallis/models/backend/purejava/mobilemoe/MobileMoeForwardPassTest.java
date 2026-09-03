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
package com.integrallis.models.backend.purejava.mobilemoe;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.safetensors.SyntheticSafetensorsBuilder;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.junit.jupiter.api.parallel.Resources;

@Tag("unit")
@ResourceLock(Resources.SYSTEM_PROPERTIES)
class MobileMoeForwardPassTest {

  private static final int HIDDEN = 32;
  private static final int VOCAB = 4;
  private static final int EXPERTS = 2;

  @Test
  void completeToyGraphOwnsSequentialAndBatchedState(@TempDir Path directory) throws IOException {
    withProperties(
        null,
        () -> {
          try (Arena arena = Arena.ofConfined()) {
            MobileMoeForwardPass graph = graph(directory, arena);
            MobileMoeForwardPass.Session first = graph.openSession();
            MobileMoeForwardPass.Session second = graph.openSession();

            float[] firstToken = graph.forward(first, 0, 0);
            assertThat(firstToken).hasSize(VOCAB);
            assertThat(allFinite(firstToken)).isTrue();
            assertThat(graph.forward(second, 0, 0)).containsExactly(firstToken);
            assertThat(graph.forwardTransient(first, 1, 1)).isNotSameAs(firstToken);
            assertThat(first.checkpoint()).isEqualTo(2);

            graph.rewind(first, 1);
            float[] continuation = graph.forward(first, 1, 1);
            graph.reset(first);
            graph.advance(first, 0, 0);
            assertThat(graph.forward(first, 1, 1)).containsExactly(continuation);

            MobileMoeForwardPass.Session sequential = graph.openSession();
            graph.advance(sequential, 0, 0);
            float[] expected = graph.forward(sequential, 1, 1);
            MobileMoeForwardPass.Session batched = graph.openSession();
            assertThat(graph.prefill(batched, new int[] {0, 1}, 0))
                .containsExactly(expected, within(1.0e-5f));
            assertThat(batched.checkpoint()).isEqualTo(2);
            assertThat(graph.config()).isEqualTo(config());
            assertThat(graph.prefillBatchSize()).isEqualTo(8);
            assertThat(graph.runtimeWeightLayout()).isEqualTo("q8");

            assertThatThrownBy(() -> graph.forward(batched, 1, 3))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("sequential");
            assertThatThrownBy(() -> graph.forward(batched, VOCAB, 2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("vocabulary");
            assertThatThrownBy(() -> graph.prefill(graph.openSession(), new int[0], 0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("empty");
          }
        });
  }

  @Test
  void compactPackedLayoutExecutesWithoutPreparedWeights(@TempDir Path directory)
      throws IOException {
    withProperties(
        "packed-int4",
        () -> {
          try (Arena arena = Arena.ofConfined()) {
            MobileMoeForwardPass graph = graph(directory, arena);
            MobileMoeForwardPass.Session sequential = graph.openSession();
            graph.advance(sequential, 2, 0);
            float[] expected = graph.forward(sequential, 3, 1);

            MobileMoeForwardPass.Session batched = graph.openSession();
            assertThat(graph.prefill(batched, new int[] {2, 3}, 0))
                .containsExactly(expected, within(5.0e-4f));
            assertThat(graph.runtimeWeightLayout()).isEqualTo("packed-int4");
          }
        });
  }

  private static MobileMoeForwardPass graph(Path directory, Arena arena) throws IOException {
    Path artifact = Files.write(directory.resolve("model.safetensors"), tensors().build());
    SafetensorsTensorSource source =
        new SafetensorsTensorSource(SafetensorsBundle.open(artifact, arena));
    return MobileMoeForwardPass.load(config(), source, 8, arena);
  }

  private static SyntheticSafetensorsBuilder tensors() {
    SyntheticSafetensorsBuilder builder = new SyntheticSafetensorsBuilder();
    linear(builder, "model.embed_tokens", VOCAB, HIDDEN, 1);
    builder.add("model.norm.weight", "BF16", new long[] {HIDDEN}, bf16(ones(HIDDEN)));
    String layer = "model.layers.0.";
    builder.add(layer + "input_layernorm.weight", "BF16", new long[] {HIDDEN}, bf16(ones(HIDDEN)));
    linear(builder, layer + "self_attn.q_proj", HIDDEN, HIDDEN, 2);
    linear(builder, layer + "self_attn.k_proj", HIDDEN, HIDDEN, 3);
    linear(builder, layer + "self_attn.v_proj", HIDDEN, HIDDEN, 4);
    linear(builder, layer + "self_attn.o_proj", HIDDEN, HIDDEN, 5);
    builder.add(
        layer + "post_attention_layernorm.weight", "BF16", new long[] {HIDDEN}, bf16(ones(HIDDEN)));
    builder.add(
        layer + "feed_forward.router.weight", "F32", new long[] {EXPERTS, HIDDEN}, f32(router()));
    builder.add(
        layer + "feed_forward.expert_bias",
        "BF16",
        new long[] {EXPERTS},
        bf16(new float[] {0.125f, -0.125f}));
    builder.add(
        layer + "feed_forward.experts.gate_up_qweight",
        "U8",
        new long[] {EXPERTS, HIDDEN, HIDDEN},
        packed(EXPERTS * HIDDEN * HIDDEN, 6));
    builder.add(
        layer + "feed_forward.experts.gate_up_scale",
        "F16",
        new long[] {EXPERTS, HIDDEN, 2},
        fp16(EXPERTS * HIDDEN * 2, 0.015625f));
    builder.add(
        layer + "feed_forward.experts.down_qweight",
        "U8",
        new long[] {EXPERTS, HIDDEN, HIDDEN / 2},
        packed(EXPERTS * HIDDEN * HIDDEN / 2, 7));
    builder.add(
        layer + "feed_forward.experts.down_scale",
        "F16",
        new long[] {EXPERTS, HIDDEN, 1},
        fp16(EXPERTS * HIDDEN, 0.015625f));
    linear(builder, layer + "feed_forward.shared_expert.gate_proj", HIDDEN, HIDDEN, 8);
    linear(builder, layer + "feed_forward.shared_expert.up_proj", HIDDEN, HIDDEN, 9);
    linear(builder, layer + "feed_forward.shared_expert.down_proj", HIDDEN, HIDDEN, 10);
    return builder;
  }

  private static void linear(
      SyntheticSafetensorsBuilder builder, String name, int rows, int columns, int seed) {
    builder.add(
        name + ".qweight", "U8", new long[] {rows, columns / 2L}, packed(rows * columns / 2, seed));
    builder.add(
        name + ".weight_scale",
        "F16",
        new long[] {rows, columns / 32L},
        fp16(rows * columns / 32, 0.015625f));
  }

  private static MobileMoeHuggingFaceConfig config() {
    return new MobileMoeHuggingFaceConfig(
        List.of("MobileMoEForCausalLM"),
        HIDDEN,
        HIDDEN,
        HIDDEN,
        1,
        1,
        1,
        HIDDEN,
        VOCAB,
        8,
        1.0e-5f,
        10_000.0f,
        new MobileMoeHuggingFaceConfig.RopeScaling("llama3", 1.0f, 8, 1.0f, 4.0f),
        EXPERTS,
        1,
        1,
        List.of(1),
        List.of("full_attention"),
        true,
        true,
        false,
        1.0f,
        true,
        Optional.of(
            new MobileMoeHuggingFaceConfig.Quantization(
                "mobilemoe-int4-g32",
                32,
                32,
                -8,
                7,
                true,
                true,
                "float16",
                "in (last dim of [out, in])",
                "out (last dim of [E, in, out])")));
  }

  private static int[] packed(int bytes, int seed) {
    int[] packed = new int[bytes];
    for (int index = 0; index < bytes; index++) {
      int low = (index + seed) % 7 - 3;
      int high = (index * 3 + seed) % 7 - 3;
      packed[index] = (low & 0x0f) | ((high & 0x0f) << 4);
    }
    return packed;
  }

  private static float[] router() {
    float[] router = new float[EXPERTS * HIDDEN];
    for (int index = 0; index < HIDDEN; index++) {
      router[index] = (index + 1) * 0.001f;
      router[HIDDEN + index] = -(index + 1) * 0.001f;
    }
    return router;
  }

  private static float[] ones(int count) {
    float[] values = new float[count];
    Arrays.fill(values, 1.0f);
    return values;
  }

  private static boolean allFinite(float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        return false;
      }
    }
    return true;
  }

  private static int[] bf16(float[] values) {
    ByteBuffer bytes =
        ByteBuffer.allocate(values.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      bytes.putShort((short) (Float.floatToRawIntBits(value) >>> Short.SIZE));
    }
    return unsigned(bytes.array());
  }

  private static int[] fp16(int count, float value) {
    ByteBuffer bytes = ByteBuffer.allocate(count * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < count; index++) {
      bytes.putShort(Float.floatToFloat16(value));
    }
    return unsigned(bytes.array());
  }

  private static int[] f32(float[] values) {
    ByteBuffer bytes =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      bytes.putFloat(value);
    }
    return unsigned(bytes.array());
  }

  private static int[] unsigned(byte[] bytes) {
    int[] values = new int[bytes.length];
    for (int index = 0; index < bytes.length; index++) {
      values[index] = Byte.toUnsignedInt(bytes[index]);
    }
    return values;
  }

  private static void withProperties(String layout, ThrowingRunnable action) throws IOException {
    String previousLayout = System.getProperty("models.mobilemoe.runtimeLayout");
    String previousBatch = System.getProperty("models.mobilemoe.prefillBatchSize");
    try {
      if (layout == null) {
        System.clearProperty("models.mobilemoe.runtimeLayout");
      } else {
        System.setProperty("models.mobilemoe.runtimeLayout", layout);
      }
      System.clearProperty("models.mobilemoe.prefillBatchSize");
      action.run();
    } finally {
      restore("models.mobilemoe.runtimeLayout", previousLayout);
      restore("models.mobilemoe.prefillBatchSize", previousBatch);
    }
  }

  private static void restore(String name, String value) {
    if (value == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, value);
    }
  }

  @FunctionalInterface
  private interface ThrowingRunnable {
    void run() throws IOException;
  }
}
