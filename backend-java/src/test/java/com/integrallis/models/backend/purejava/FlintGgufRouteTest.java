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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.flint.FlintConfig;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Exercises Flint through the public backend, including rewind past an overwritten KV window. */
@Tag("unit")
class FlintGgufRouteTest {
  static final int DIM = 8;
  static final int VOCAB = 16;
  static final int LAYERS = 3;
  static final int HEADS = 4;
  static final int KV_HEADS = 2;
  static final int HEAD_DIM = 4;
  static final int HIDDEN = 12;
  static final int EXPERT_HIDDEN = 4;
  static final int SHARED_HIDDEN = 8;
  static final int EXPERTS = 4;
  static final int USED = 2;

  @Test
  void loadsFlintAndReplaysAnOverwrittenSlidingWindow(@TempDir Path directory) throws Exception {
    Path model = write(directory);
    try (PureJavaBackend backend = PureJavaBackend.load(model);
        PureJavaBackend fresh = PureJavaBackend.load(model)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("flint");
      assertThat(backend.metadata().embeddingDim()).isEqualTo(DIM);
      assertThat(backend.metadata().numHeads()).isEqualTo(HEADS);
      float[] logits = backend.prefill(new int[] {3, 4, 5, 6, 7, 8}, 0);
      assertThat(logits).hasSize(VOCAB);
      for (float logit : logits) {
        assertThat(Float.isFinite(logit)).isTrue();
      }
      backend.rewind(2);
      assertThat(backend.prefill(new int[] {9, 10}, 2))
          .containsExactly(fresh.prefill(new int[] {3, 4, 9, 10}, 0));
      backend.reset();
      fresh.reset();
      assertThat(backend.prefill(new int[] {10, 9, 8}, 0))
          .containsExactly(fresh.prefill(new int[] {10, 9, 8}, 0));
    }
  }

  @Test
  void prefillMatchesSequentialDecodeAndKeepsReturnedLogitsStable(@TempDir Path directory)
      throws Exception {
    Path model = write(directory);
    try (PureJavaBackend backend = PureJavaBackend.load(model);
        PureJavaBackend fresh = PureJavaBackend.load(model)) {
      float[] first = backend.forward(3, 0);
      float[] retained = first.clone();
      backend.forward(4, 1);
      assertThat(backend.forward(5, 2)).containsExactly(fresh.prefill(new int[] {3, 4, 5}, 0));
      assertThat(first).containsExactly(retained);
      assertThatThrownBy(() -> backend.forward(6, 1)).isInstanceOf(IllegalArgumentException.class);
    }
  }

  /** Writes the numerical fixture for the independent llama.cpp oracle. */
  public static void main(String[] args) throws Exception {
    Path directory = Path.of(args[0]);
    Files.createDirectories(directory);
    System.out.println(write(directory));
  }

  /** Full vectors from llama.cpp 0893ec98f5eafad36a43a72cd1c3723d72b0d03a, F32 KV. */
  @Test
  void matchesIndependentLlamaCppLogitsAcrossDenseMoeAndSlidingLayers(@TempDir Path directory)
      throws Exception {
    float[][] expected = {
      {
        0.0703419968f,
        0.619555056f,
        0.0295845997f,
        -0.199276879f,
        -0.87206471f,
        0.0722146779f,
        0.621427655f,
        0.0314572677f,
        0.308103144f,
        -0.870192051f,
        0.0740873516f,
        0.623300254f,
        0.5201689f,
        0.309975803f,
        -0.868319452f,
        0.0759599879f
      },
      {
        0.303573132f,
        0.146121353f,
        0.106762148f,
        -0.16657117f,
        -0.879462659f,
        0.279025048f,
        0.12157312f,
        0.0822140276f,
        0.530928612f,
        -0.904010653f,
        0.254476964f,
        0.0970250294f,
        0.260107964f,
        0.506380498f,
        -0.928558826f,
        0.229928806f
      },
      {
        0.498356909f,
        -0.244475633f,
        -0.198618069f,
        0.0857558846f,
        -0.454812825f,
        0.466821522f,
        -0.27601102f,
        -0.230153441f,
        0.73145771f,
        -0.486348182f,
        0.435286134f,
        -0.307546437f,
        -0.768576384f,
        0.699922264f,
        -0.517883599f,
        0.403750688f
      },
      {
        0.485931307f,
        -0.0554041155f,
        -0.855186582f,
        0.00339169591f,
        0.853360772f,
        0.463283598f,
        -0.078051731f,
        -0.87783432f,
        -0.40556699f,
        0.830713093f,
        0.44063583f,
        -0.100699507f,
        -1.34117365f,
        -0.428214699f,
        0.808065295f,
        0.417988092f
      },
      {
        0.0999066457f,
        0.0680702031f,
        -0.532876074f,
        -0.0186746232f,
        1.03231287f,
        0.104951002f,
        0.0731145665f,
        -0.527831733f,
        -0.815384746f,
        1.03735721f,
        0.109995313f,
        0.0781588927f,
        -0.366875708f,
        -0.810340405f,
        1.04240155f,
        0.115039676f
      },
      {
        -0.548439205f,
        0.209002361f,
        0.363402963f,
        0.0285096951f,
        0.210419998f,
        -0.525118232f,
        0.232323259f,
        0.386723906f,
        -0.437739164f,
        0.233740896f,
        -0.501797318f,
        0.255644262f,
        1.32724059f,
        -0.414418191f,
        0.257061899f,
        -0.478476375f
      }
    };
    try (PureJavaBackend backend = PureJavaBackend.load(write(directory))) {
      for (int position = 0; position < expected.length; position++) {
        float[] actual = backend.forward(position + 3, position);
        for (int token = 0; token < VOCAB; token++) {
          assertThat(actual[token])
              .as("position %s, token %s", position, token)
              .isCloseTo(expected[position][token], within(2e-5f));
        }
      }
    }
  }

  @Test
  void independentSessionsDoNotShareMutableState(@TempDir Path directory) throws Exception {
    try (PureJavaBackend backend = PureJavaBackend.load(write(directory));
        var first = backend.openSession();
        var second = backend.openSession()) {
      float[] a = backend.prefill(first, new int[] {3, 4, 5}, 0);
      backend.prefill(second, new int[] {7, 8}, 0);
      backend.reset(second);
      assertThat(backend.prefill(second, new int[] {3, 4, 5}, 0)).containsExactly(a);
      assertThat(backend.forward(first, 6, 3)).containsExactly(backend.forward(second, 6, 3));
    }
  }

  @Test
  void rejectsMetadataThatWouldSilentlyChangeTheGraph(@TempDir Path directory) throws Exception {
    try (var arena = Arena.ofConfined()) {
      var file = GgufParser.parse(write(directory), arena);
      var cases =
          Map.<String, GgufMetadataValue>of(
              "flint.expert_gating_func", new GgufMetadataValue.Uint32Value(1),
              "flint.attention.head_count", new GgufMetadataValue.Uint32Value(1 << 30),
              "flint.attention.head_count_kv", new GgufMetadataValue.Uint32Value(3),
              "flint.rope.dimension_count", new GgufMetadataValue.Uint32Value(2),
              "flint.expert_shared_count", new GgufMetadataValue.Uint32Value(1),
              "flint.leading_dense_block_count", new GgufMetadataValue.Uint32Value(LAYERS),
              "flint.expert_weights_scale", new GgufMetadataValue.Float32Value(Float.NaN),
              "flint.rope.scaling.type", new GgufMetadataValue.StringValue("yarn"));
      for (var entry : cases.entrySet()) {
        var metadata = new HashMap<>(file.metadata().entries());
        metadata.put(entry.getKey(), entry.getValue());
        assertThatThrownBy(() -> FlintConfig.fromMetadata(new GgufMetadata(metadata)))
            .as(entry.getKey())
            .isInstanceOf(IllegalArgumentException.class);
      }
      var metadata = new HashMap<>(file.metadata().entries());
      metadata.remove("flint.attention.rope_pattern");
      assertThatThrownBy(() -> FlintConfig.fromMetadata(new GgufMetadata(metadata)))
          .hasMessageContaining("rope_pattern");
    }
  }

  static Path write(Path directory) throws Exception {
    List<String> tokens = new ArrayList<>(List.of("<unk>", "<s>", "</s>"));
    for (int index = 3; index < VOCAB; index++) {
      tokens.add("t" + index);
    }
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "flint")
            .addString("general.name", "Flint numerical fixture")
            .addUint32("flint.block_count", LAYERS)
            .addUint32("flint.embedding_length", DIM)
            .addUint32("flint.attention.head_count", HEADS)
            .addUint32("flint.attention.head_count_kv", KV_HEADS)
            .addUint32("flint.attention.key_length", HEAD_DIM)
            .addUint32("flint.attention.value_length", HEAD_DIM)
            .addUint32("flint.rope.dimension_count", HEAD_DIM)
            .addUint32("flint.context_length", 32)
            .addUint32("flint.feed_forward_length", HIDDEN)
            .addUint32("flint.leading_dense_block_count", 1)
            .addUint32("flint.expert_feed_forward_length", EXPERT_HIDDEN)
            .addUint32("flint.expert_shared_feed_forward_length", SHARED_HIDDEN)
            .addUint32("flint.expert_shared_count", 2)
            .addUint32("flint.expert_count", EXPERTS)
            .addUint32("flint.expert_used_count", USED)
            .addUint32("flint.expert_gating_func", 2)
            .addBool("flint.expert_weights_norm", true)
            .addFloat32("flint.expert_weights_scale", 1.0f)
            .addUint32("flint.attention.sliding_window", 2)
            .addBoolArray("flint.attention.sliding_window_pattern", List.of(false, true, false))
            .addBoolArray("flint.attention.rope_pattern", List.of(false, true, false))
            .addFloat32("flint.rope.freq_base", 10000.0f)
            .addFloat32("flint.attention.layer_norm_rms_epsilon", 1.0e-6f)
            .addString("tokenizer.ggml.model", "llama")
            .addStringArray("tokenizer.ggml.tokens", tokens)
            .addFloat32Array("tokenizer.ggml.scores", java.util.Collections.nCopies(VOCAB, 0.0f))
            .addInt32Array(
                "tokenizer.ggml.token_type",
                java.util.stream.IntStream.range(0, VOCAB)
                    .map(i -> i == 0 ? 2 : i < 3 ? 3 : 1)
                    .boxed()
                    .toList())
            .addUint32("tokenizer.ggml.bos_token_id", 1)
            .addUint32("tokenizer.ggml.eos_token_id", 2)
            .addUint32("tokenizer.ggml.unknown_token_id", 0);
    tensor(builder, "token_embd.weight", 1, false, DIM, VOCAB);
    tensor(builder, "output.weight", 2, false, DIM, VOCAB);
    tensor(builder, "output_norm.weight", 3, true, DIM);
    for (int layer = 0; layer < LAYERS; layer++) {
      String p = "blk." + layer + ".";
      int seed = 11 + 19 * layer;
      tensor(builder, p + "attn_norm.weight", seed++, true, DIM);
      tensor(builder, p + "attn_q.weight", seed++, false, DIM, HEADS * HEAD_DIM);
      tensor(builder, p + "attn_k.weight", seed++, false, DIM, KV_HEADS * HEAD_DIM);
      tensor(builder, p + "attn_v.weight", seed++, false, DIM, KV_HEADS * HEAD_DIM);
      tensor(builder, p + "attn_output.weight", seed++, false, HEADS * HEAD_DIM, DIM);
      tensor(builder, p + "attn_q_norm.weight", seed++, true, HEAD_DIM);
      tensor(builder, p + "attn_k_norm.weight", seed++, true, HEAD_DIM);
      tensor(builder, p + "attn_sinks.weight", seed++, false, HEADS);
      tensor(builder, p + "ffn_norm.weight", seed++, true, DIM);
      if (layer == 0) {
        tensor(builder, p + "ffn_gate.weight", seed++, false, DIM, HIDDEN);
        tensor(builder, p + "ffn_up.weight", seed++, false, DIM, HIDDEN);
        tensor(builder, p + "ffn_down.weight", seed++, false, HIDDEN, DIM);
      } else {
        tensor(builder, p + "ffn_gate_inp.weight", seed++, false, DIM, EXPERTS);
        tensor(builder, p + "exp_probs_b.bias", seed++, false, EXPERTS);
        tensor(builder, p + "ffn_gate_exps.weight", seed++, false, DIM, EXPERT_HIDDEN, EXPERTS);
        tensor(builder, p + "ffn_up_exps.weight", seed++, false, DIM, EXPERT_HIDDEN, EXPERTS);
        tensor(builder, p + "ffn_down_exps.weight", seed++, false, EXPERT_HIDDEN, DIM, EXPERTS);
        tensor(builder, p + "ffn_gate_shexp.weight", seed++, false, DIM, SHARED_HIDDEN);
        tensor(builder, p + "ffn_up_shexp.weight", seed++, false, DIM, SHARED_HIDDEN);
        tensor(builder, p + "ffn_down_shexp.weight", seed++, false, SHARED_HIDDEN, DIM);
      }
    }
    Path model = directory.resolve("flint-f32.gguf");
    Files.write(model, builder.build());
    return model;
  }

  private static void tensor(
      SyntheticGgufBuilder builder, String name, int seed, boolean norm, long... shape) {
    int count = 1;
    for (long dimension : shape) {
      count = Math.multiplyExact(count, Math.toIntExact(dimension));
    }
    ByteBuffer bytes = ByteBuffer.allocate(count * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < count; i++) {
      bytes.putFloat((norm ? 1.0f : 0.0f) + (((seed * 17 + i * 7) % 31) - 15) * 0.025f);
    }
    builder.addTensor(name, GgufTensorType.F32, shape, bytes.array());
  }
}
