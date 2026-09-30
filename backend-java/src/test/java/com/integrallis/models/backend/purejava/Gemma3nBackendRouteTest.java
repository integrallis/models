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

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A Gemma 3n file reaching the decoder through the backend's own dispatch.
 *
 * <p>The third surface. A layer shape has to be taught the graph, the planner, and the adapter, and
 * the graph tests cover only the first -- on 2026-09-29 one fleet run produced four failures that
 * were all in the other two. So this goes through {@link PureJavaBackend#load}, which is the path
 * the fleet takes, rather than constructing the graph directly as the other Gemma 3n tests do.
 *
 * <p>A separate fixture from {@code Gemma3nToyModel}: that one builds a {@code GgufFile} in memory
 * with no metadata, and loading through the backend needs real metadata and a tokenizer on disk.
 */
@Tag("unit")
class Gemma3nBackendRouteTest {

  private static final int DIM = 8;
  private static final int LAYERS = 4;
  private static final int HEADS = 2;
  private static final int KV_HEADS = 1;
  private static final int HEAD_DIM = 4;
  private static final int HIDDEN = 6;
  private static final int VOCAB = 8;
  private static final int ALTUP = 3;
  private static final int PER_LAYER_DIM = 2;
  private static final int LAUREL_RANK = 2;

  @Test
  void theBackendLoadsAGemma3nFileUnderItsOwnArchitecture(@TempDir Path directory)
      throws Exception {
    Path model = write(directory);

    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("gemma3n");
      assertThat(backend.metadata().numLayers()).isEqualTo(LAYERS);
      assertThat(backend.metadata().embeddingDim()).isEqualTo(DIM);

      float[] logits = backend.prefill(new int[] {1, 2, 3}, 0);

      assertThat(logits).hasSize(VOCAB);
      for (float logit : logits) {
        assertThat(Float.isFinite(logit)).isTrue();
      }
    }
  }

  /**
   * And a rewind through the backend, which is what the RAG harness does before its first question.
   */
  @Test
  void theBackendCanRewindAGemma3nSequence(@TempDir Path directory) throws Exception {
    Path model = write(directory);

    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      backend.prefill(new int[] {1, 2, 3, 4}, 0);
      backend.rewind(2);
      float[] afterRewind = backend.prefill(new int[] {3, 4}, 2).clone();

      try (PureJavaBackend fresh = PureJavaBackend.load(model)) {
        assertThat(afterRewind).containsExactly(fresh.prefill(new int[] {1, 2, 3, 4}, 0));
      }
    }
  }

  private static Path write(Path directory) throws Exception {
    List<Boolean> sliding = List.of(true, true, false, true);
    List<Float> sparsity = new ArrayList<>();
    for (int layer = 0; layer < LAYERS; layer++) {
      // Negative infinity past the first layer, exactly as the published file carries it.
      sparsity.add(layer == 0 ? 1.0f : Float.NEGATIVE_INFINITY);
    }
    List<String> tokens = new ArrayList<>();
    for (int index = 0; index < VOCAB; index++) {
      tokens.add("t" + index);
    }
    List<Float> scores = new ArrayList<>();
    for (int index = 0; index < VOCAB; index++) {
      scores.add(0.0f);
    }

    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "gemma3n")
            .addString("general.name", "Toy Gemma 3n")
            .addUint32("gemma3n.block_count", LAYERS)
            .addUint32("gemma3n.embedding_length", DIM)
            .addUint32("gemma3n.attention.head_count", HEADS)
            .addUint32("gemma3n.attention.head_count_kv", KV_HEADS)
            .addUint32("gemma3n.attention.key_length", HEAD_DIM)
            .addUint32("gemma3n.attention.value_length", HEAD_DIM)
            .addUint32("gemma3n.context_length", 16)
            .addUint32("gemma3n.feed_forward_length", HIDDEN)
            .addUint32("gemma3n.embedding_length_per_layer_input", PER_LAYER_DIM)
            .addUint32("gemma3n.altup.num_inputs", ALTUP)
            .addUint32("gemma3n.altup.active_idx", 0)
            .addUint32("gemma3n.attention.sliding_window", 2)
            .addFloat32("gemma3n.rope.freq_base", 10_000.0f)
            .addFloat32("gemma3n.attention.layer_norm_rms_epsilon", 1.0e-6f)
            // A float, as published: one sharing layer over four.
            .addFloat32("gemma3n.attention.shared_kv_layers", 1.0f)
            .addBoolArray("gemma3n.attention.sliding_window_pattern", sliding)
            .addFloat32Array("gemma3n.activation_sparsity_scale", sparsity)
            .addStringArray("tokenizer.ggml.tokens", tokens)
            .addFloat32Array("tokenizer.ggml.scores", scores)
            .addUint32("tokenizer.ggml.bos_token_id", 0)
            .addUint32("tokenizer.ggml.eos_token_id", 1);

    matrix(builder, "token_embd.weight", VOCAB, DIM, 3);
    matrix(builder, "per_layer_token_embd.weight", VOCAB, PER_LAYER_DIM * LAYERS, 5);
    matrix(builder, "per_layer_model_proj.weight", PER_LAYER_DIM * LAYERS, DIM, 7);
    vector(builder, "per_layer_proj_norm.weight", PER_LAYER_DIM, 9);
    stacked(builder, "altup_proj.weight", DIM, ALTUP - 1, 11);
    stacked(builder, "altup_unembd_proj.weight", DIM, ALTUP - 1, 13);
    vector(builder, "output_norm.weight", DIM, 15);

    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      boolean ownsKv = layer < LAYERS - 1;
      vector(builder, prefix + "attn_norm.weight", DIM, 20 + layer);
      matrix(builder, prefix + "attn_q.weight", HEADS * HEAD_DIM, DIM, 30 + layer);
      if (ownsKv) {
        matrix(builder, prefix + "attn_k.weight", KV_HEADS * HEAD_DIM, DIM, 40 + layer);
        matrix(builder, prefix + "attn_v.weight", KV_HEADS * HEAD_DIM, DIM, 50 + layer);
      }
      matrix(builder, prefix + "attn_output.weight", DIM, HEADS * HEAD_DIM, 60 + layer);
      vector(builder, prefix + "attn_q_norm.weight", HEAD_DIM, 70 + layer);
      vector(builder, prefix + "attn_k_norm.weight", HEAD_DIM, 80 + layer);
      vector(builder, prefix + "post_attention_norm.weight", DIM, 90 + layer);
      vector(builder, prefix + "ffn_norm.weight", DIM, 100 + layer);
      matrix(builder, prefix + "ffn_gate.weight", HIDDEN, DIM, 110 + layer);
      matrix(builder, prefix + "ffn_up.weight", HIDDEN, DIM, 120 + layer);
      matrix(builder, prefix + "ffn_down.weight", DIM, HIDDEN, 130 + layer);
      vector(builder, prefix + "post_ffw_norm.weight", DIM, 140 + layer);
      matrix(builder, prefix + "inp_gate.weight", PER_LAYER_DIM, DIM, 150 + layer);
      matrix(builder, prefix + "proj.weight", DIM, PER_LAYER_DIM, 160 + layer);
      vector(builder, prefix + "post_norm.weight", DIM, 170 + layer);
      matrix(builder, prefix + "altup_router.weight", ALTUP, DIM, 180 + layer);
      vector(builder, prefix + "altup_router_norm.weight", DIM, 190 + layer);
      matrix(builder, prefix + "altup_predict_coef.weight", ALTUP * ALTUP, ALTUP, 200 + layer);
      matrix(builder, prefix + "altup_correct_coef.weight", ALTUP, ALTUP, 210 + layer);
      vector(builder, prefix + "altup_correct_scale.weight", DIM, 220 + layer);
      matrix(builder, prefix + "laurel_l.weight", LAUREL_RANK, DIM, 230 + layer);
      matrix(builder, prefix + "laurel_r.weight", DIM, LAUREL_RANK, 240 + layer);
      vector(builder, prefix + "laurel_post_norm.weight", DIM, 250 + layer);
    }

    Path model = directory.resolve("toy-gemma3n.gguf");
    Files.write(model, builder.build());
    return model;
  }

  private static void matrix(
      SyntheticGgufBuilder builder, String name, int rows, int columns, int seed) {
    builder.addTensor(
        name, GgufTensorType.F32, new long[] {columns, rows}, bytes(values(rows * columns, seed)));
  }

  /** A stacked {@code [dim, dim, slices]} tensor, which is how the AltUp projections are stored. */
  private static void stacked(
      SyntheticGgufBuilder builder, String name, int dim, int slices, int seed) {
    builder.addTensor(
        name,
        GgufTensorType.F32,
        new long[] {dim, dim, slices},
        bytes(values(dim * dim * slices, seed)));
  }

  private static void vector(SyntheticGgufBuilder builder, String name, int length, int seed) {
    float[] norm = new float[length];
    for (int index = 0; index < length; index++) {
      norm[index] = 0.85f + ((seed + index * 3) % 7) * 0.04f;
    }
    builder.addTensor(name, GgufTensorType.F32, new long[] {length}, bytes(norm));
  }

  private static float[] values(int count, int seed) {
    float[] result = new float[count];
    for (int index = 0; index < count; index++) {
      result[index] = (((seed + index * 7) % 13) - 6) * 0.05f;
    }
    return result;
  }

  private static byte[] bytes(float[] values) {
    ByteBuffer buffer =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      buffer.putFloat(value);
    }
    return buffer.array();
  }
}
