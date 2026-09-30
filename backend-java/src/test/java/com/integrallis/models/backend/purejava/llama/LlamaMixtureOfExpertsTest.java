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
package com.integrallis.models.backend.purejava.llama;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The routed feed-forward of a mixture-of-experts decoder.
 *
 * <p>The central test builds two nano models that are <b>bit-identical outside the feed-forward</b>
 * -- same architecture id, same attention weights, same norms, drawn once and handed to both
 * builders -- where one is dense and the other holds {@link #EXPERTS} copies of that same dense
 * feed-forward. Because the routing weights sum to one, a token routed to any subset of identical
 * experts must produce exactly what the dense model produces. That single equality pins the
 * renormalization, the per-expert SwiGLU, the weighted accumulation, and the expert slicing at
 * once, and it fails loudly for the error that matters most: weights that do not sum to one are a
 * silent per-token scale error, not a crash.
 *
 * <p>The routed model deliberately declares a {@code feed_forward_length} that differs from its
 * {@code expert_feed_forward_length}. A routed model carries both, and Qwen3-30B-A3B's differ by a
 * factor of eight, so a path that reached for the dense width would be shaped wrongly rather than
 * merely slower.
 */
@Tag("unit")
class LlamaMixtureOfExpertsTest {

  private static final int DIM = 32;
  private static final int HEADS = 2;
  private static final int KV_HEADS = 1;
  private static final int HEAD_DIM = DIM / HEADS;
  private static final int KV_DIM = KV_HEADS * HEAD_DIM;
  private static final int VOCAB_SIZE = 32;
  private static final int LAYERS = 2;
  private static final int CONTEXT = 64;

  /** The width one expert's feed-forward has, and the dense model's only feed-forward width. */
  private static final int EXPERT_HIDDEN = 32;

  /**
   * The routed model's dense {@code feed_forward_length}, which nothing should ever read. Different
   * from {@link #EXPERT_HIDDEN} on purpose.
   */
  private static final int UNUSED_DENSE_HIDDEN = 64;

  private static final int EXPERTS = 4;
  private static final int USED = 2;

  private static final String ARCH = "qwen3moe";

  @Test
  void aRoutedLayerCarriesExpertsInsteadOfADenseFeedForward() {
    GgufFile file =
        build(sharedTensors(new Random(31), GgufTensorType.F32), true, GgufTensorType.F32);
    LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());

    assertThat(config.architecture()).isEqualTo(DecoderArchitecture.QWEN3MOE);
    assertThat(config.usesMixtureOfExperts()).isTrue();
    assertThat(config.numExperts()).isEqualTo(EXPERTS);
    assertThat(config.numExpertsUsed()).isEqualTo(USED);
    assertThat(config.expertHiddenDim()).isEqualTo(EXPERT_HIDDEN);
    assertThat(config.hiddenDim())
        .describedAs("the dense width is carried but must not be the expert width")
        .isEqualTo(UNUSED_DENSE_HIDDEN)
        .isNotEqualTo(config.expertHiddenDim());

    // Loading at all is the assertion: a routed layer publishes no dense ffn_gate, which is also
    // how
    // the Phi-3 fused gate+up split is detected, so without a mixture-of-experts guard this throws
    // claiming a malformed ffn_up.
    LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
    for (int layer = 0; layer < LAYERS; layer++) {
      LlamaWeights.LayerWeights lw = weights.layer(layer);
      assertThat(lw.moe()).describedAs("layer %s must be routed", layer).isNotNull();
      assertThat(lw.ffnGate()).isNull();
      assertThat(lw.ffnUp()).isNull();
      assertThat(lw.ffnDown()).isNull();
      assertThat(lw.moe().gate()).hasSize(EXPERTS);
      assertThat(lw.moe().up()).hasSize(EXPERTS);
      assertThat(lw.moe().down()).hasSize(EXPERTS);
    }
  }

  /**
   * Planning topology on a routed model, which is where loading actually broke.
   *
   * <p>Every Qwen3-30B-A3B in the fleet failed to load with a NullPointerException out of {@code
   * ModelTopology.threadShareable}: the tensor <i>types</i> were made routing-aware, but the pass
   * below them that probes the mapped <i>segments</i> for thread shareability still named {@code
   * ffnGate}/{@code ffnUp}/{@code ffnDown}, which a routed layer does not have. Nothing caught it
   * because the routed tests exercised the graph and never the planner.
   */
  @Test
  void planningTopologyReadsARoutedLayerWithoutTouchingItsAbsentDenseTensors() {
    Map<String, byte[]> shared = sharedTensors(new Random(77), GgufTensorType.F32);
    // Thread-shareable on purpose. With a confined fixture the probe rejects the attention output
    // first and returns before it ever looks at the feed-forward, so the defect this pins -- a null
    // dense ffn_gate on a routed layer -- is unreachable and the test would pass either way.
    GgufFile routedFile = build(shared, true, GgufTensorType.F32, true, true);
    GgufFile denseFile = build(shared, false, GgufTensorType.F32, true, true);
    LlamaConfig routedConfig = LlamaConfig.fromMetadata(routedFile.metadata());
    LlamaConfig denseConfig = LlamaConfig.fromMetadata(denseFile.metadata());

    // Reaching past this line at all is the regression: the shareability probe used to dereference
    // the dense feed-forward segments, which a routed layer does not have.
    ModelTopology routed =
        ModelTopology.from(ARCH, routedConfig, LlamaWeights.fromGgufFile(routedFile, routedConfig));
    ModelTopology dense =
        ModelTopology.from(ARCH, denseConfig, LlamaWeights.fromGgufFile(denseFile, denseConfig));

    assertThat(routed.architecture()).isEqualTo(ARCH);
    assertThat(routed.layers()).hasSize(LAYERS);
    for (ModelTopology.LayerTopology layer : routed.layers()) {
      // The expert tensors' types, since those are the weights a routed layer multiplies.
      assertThat(layer.gate()).isEqualTo(GgufTensorType.F32);
      assertThat(layer.up()).isEqualTo(GgufTensorType.F32);
      assertThat(layer.down()).isEqualTo(GgufTensorType.F32);
    }
    // Whether these fixtures' segments are shareable at all depends on how they were mapped, which
    // is not what this is about -- but routing must not change the answer, because the expert
    // segments are slices of tensors from the same file as the dense ones.
    assertThat(routed.threadShareableProjectionWeights())
        .describedAs("routing must not change the shareability verdict")
        .isEqualTo(dense.threadShareableProjectionWeights())
        // And it must be true, or the probe stopped early and never read the expert segments.
        .isTrue();
  }

  @Test
  void identicalExpertsReproduceTheDenseFeedForward() {
    Map<String, byte[]> shared = sharedTensors(new Random(1234), GgufTensorType.F32);
    GgufFile dense = build(shared, false, GgufTensorType.F32);
    GgufFile routed = build(shared, true, GgufTensorType.F32);

    LlamaConfig denseConfig = LlamaConfig.fromMetadata(dense.metadata());
    LlamaConfig routedConfig = LlamaConfig.fromMetadata(routed.metadata());
    assertThat(denseConfig.usesMixtureOfExperts()).isFalse();
    assertThat(routedConfig.usesMixtureOfExperts()).isTrue();

    int[] tokens = {3, 5, 7};
    float[] fromDense = runSequence(newForwardPass(dense, denseConfig), tokens);
    float[] fromRouted = runSequence(newForwardPass(routed, routedConfig), tokens);

    assertThat(fromRouted).hasSameSizeAs(fromDense);
    for (int index = 0; index < fromDense.length; index++) {
      assertThat(fromRouted[index])
          .describedAs(
              "logit %s: routing %s identical experts must weigh them to exactly one", index, USED)
          .isEqualTo(fromDense[index], within(1.0e-5f));
    }
  }

  @Test
  void batchedPrefillAgreesWithTokenByTokenDecodeOnARoutedModel() {
    // Q8_0 projections, because F32 is not a batched-matmul type: built on F32 weights this test
    // compared the serial path against itself and passed with the batched routed branch deleted.
    Map<String, byte[]> shared = sharedTensors(new Random(99), GgufTensorType.Q8_0);
    GgufFile routed = build(shared, true, GgufTensorType.Q8_0);
    LlamaConfig config = LlamaConfig.fromMetadata(routed.metadata());

    int[] tokens = {2, 4, 6, 8};
    LlamaForwardPass batchedPass = newForwardPass(routed, config);
    assertThat(batchedPass.usesBatchedPrefill())
        .describedAs("without batching engaged this test cannot see the batched routed branch")
        .isTrue();
    float[] serial = runSequence(newForwardPass(routed, config), tokens);
    float[] batched = batchedPass.prefill(tokens, 0);

    // The routed branch had to be added at the batched site as well as the single-token one. Gemma
    // 4
    // taught this the hard way: guarding one of the two leaves the other reaching for a tensor that
    // is not there.
    assertThat(batched).hasSameSizeAs(serial);
    for (int index = 0; index < serial.length; index++) {
      assertThat(batched[index]).isEqualTo(serial[index], within(1.0e-4f));
    }
  }

  @Test
  void expertSlicesFollowTheStackingOrderTheFileUses() {
    // The dense-equivalence test gives every expert the same weights, so it cannot see a transposed
    // or interleaved expert layout: wrong slicing would still return the right numbers. This pins
    // the
    // layout on its own, with each expert's block filled with its own index. GGUF lists dimensions
    // fastest-varying first, so ffn_gate_exps [DIM, EXPERT_HIDDEN, EXPERTS] stores expert e as one
    // contiguous EXPERT_HIDDEN x DIM block at e * EXPERT_HIDDEN * DIM. Straddling that boundary on
    // a
    // quantized tensor would read a neighbour's scale bytes as weights -- plausible numbers, no
    // error.
    Map<String, byte[]> shared = sharedTensors(new Random(5), GgufTensorType.F32);
    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      shared.put(prefix + "gate", markedF32(EXPERT_HIDDEN * DIM, layer + 1));
      shared.put(prefix + "up", markedF32(EXPERT_HIDDEN * DIM, layer + 1));
      shared.put(prefix + "down", markedF32(DIM * EXPERT_HIDDEN, layer + 1));
    }
    GgufFile file = buildDistinctExperts(shared);
    LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
    LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);

    for (int layer = 0; layer < LAYERS; layer++) {
      LlamaWeights.MoeWeights moe = weights.layer(layer).moe();
      for (int expert = 0; expert < EXPERTS; expert++) {
        assertThat(moe.gate()[expert].byteSize())
            .describedAs("layer %s expert %s gate slice size", layer, expert)
            .isEqualTo((long) EXPERT_HIDDEN * DIM * Float.BYTES);
        assertThat(moe.gate()[expert].get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0))
            .describedAs("layer %s gate slice must start at expert %s's own block", layer, expert)
            .isEqualTo(mark(layer + 1, expert));
        assertThat(moe.down()[expert].byteSize())
            .isEqualTo((long) DIM * EXPERT_HIDDEN * Float.BYTES);
        assertThat(moe.down()[expert].get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0))
            .describedAs("layer %s down slice must start at expert %s's own block", layer, expert)
            .isEqualTo(mark(layer + 1, expert));
      }
    }
  }

  @Test
  void twoRoutedSessionsBatchedTogetherAgreeWithEachRunAlone() {
    // The independent-session batch is a third feed-forward site, reached only through the
    // multi-session API, and it was silently unexercised until this test existed: deleting its
    // routed branch broke nothing.
    Map<String, byte[]> shared = sharedTensors(new Random(7), GgufTensorType.Q8_0);
    GgufFile routed = build(shared, true, GgufTensorType.Q8_0);
    LlamaConfig config = LlamaConfig.fromMetadata(routed.metadata());

    int[] firstPrompt = {2, 4, 6};
    int[] secondPrompt = {8, 10};

    LlamaForwardPass firstAlone = newForwardPass(routed, config);
    firstAlone.prefill(firstPrompt, 0);
    float[] firstExpected = firstAlone.forward(12, firstPrompt.length).clone();

    LlamaForwardPass secondAlone = newForwardPass(routed, config);
    secondAlone.prefill(secondPrompt, 0);
    float[] secondExpected = secondAlone.forward(14, secondPrompt.length).clone();

    LlamaForwardPass batched = newForwardPass(routed, config);
    LlamaForwardPass.Session first = batched.openSession();
    LlamaForwardPass.Session second = batched.openSession();
    batched.prefill(first, firstPrompt, 0);
    batched.prefill(second, secondPrompt, 0);
    LogitBatch actual =
        batched.forwardBatchTransient(
            new LlamaForwardPass.Session[] {first, second}, new int[] {12, 14});

    assertThat(actual.tokenCount()).isEqualTo(2);
    assertThat(actual.copyRow(0)).containsExactly(firstExpected, within(1.0e-4f));
    assertThat(actual.copyRow(1)).containsExactly(secondExpected, within(1.0e-4f));
  }

  @Test
  void selectionTakesTheHighestLogitsInDescendingOrder() {
    float[] logits = {0.1f, 2.5f, -1.0f, 0.9f, 2.4f, 0.0f};
    int[] selected = new int[3];
    float[] weights = new float[3];

    LlamaForwardPass.selectExperts(logits, logits.length, 3, selected, weights);

    assertThat(selected).containsExactly(1, 4, 3);
    assertThat(sum(weights)).isEqualTo(1.0f, within(1.0e-6f));
    assertThat(weights[0]).isGreaterThan(weights[1]);
    assertThat(weights[1]).isGreaterThan(weights[2]);
  }

  @Test
  void routingWeightsAreASoftmaxOverOnlyTheSelectedLogits() {
    float[] logits = {3.0f, 1.0f, -5.0f, 2.0f};
    int[] selected = new int[2];
    float[] weights = new float[2];

    LlamaForwardPass.selectExperts(logits, logits.length, 2, selected, weights);

    // Softmaxing the two selected logits is what renormalizing a full-width softmax comes to, and
    // it
    // is the arithmetic norm_topk_prob asks for. The unselected -5 must not appear in the
    // denominator: including it would pull both weights below their true share.
    float[] expected = {3.0f, 2.0f};
    TensorOps.softmax(expected, 0, 2);
    assertThat(selected).containsExactly(0, 3);
    assertThat(weights[0]).isEqualTo(expected[0], within(1.0e-6f));
    assertThat(weights[1]).isEqualTo(expected[1], within(1.0e-6f));
  }

  @Test
  void selectingEveryExpertIsAPlainSoftmax() {
    float[] logits = {0.5f, -0.25f, 1.5f, 0.0f};
    int[] selected = new int[logits.length];
    float[] weights = new float[logits.length];
    float[] expected = logits.clone();
    TensorOps.softmax(expected, 0, expected.length);

    LlamaForwardPass.selectExperts(logits, logits.length, logits.length, selected, weights);

    assertThat(selected).containsExactlyInAnyOrder(0, 1, 2, 3);
    for (int slot = 0; slot < selected.length; slot++) {
      assertThat(weights[slot]).isEqualTo(expected[selected[slot]], within(1.0e-6f));
    }
  }

  @Test
  void tiedLogitsKeepTheLowerExpertIndex() {
    float[] logits = {1.0f, 2.0f, 2.0f, 2.0f, 0.5f};
    int[] selected = new int[2];
    float[] weights = new float[2];

    LlamaForwardPass.selectExperts(logits, logits.length, 2, selected, weights);

    // Three experts tie for first. Which two run changes the answer, so the tie-break is part of
    // the
    // contract rather than an accident of the comparison that happened to be written.
    assertThat(selected).containsExactly(1, 2);
    assertThat(weights[0]).isEqualTo(0.5f, within(1.0e-6f));
    assertThat(weights[1]).isEqualTo(0.5f, within(1.0e-6f));
  }

  private static float sum(float[] values) {
    float total = 0.0f;
    for (float value : values) {
      total += value;
    }
    return total;
  }

  private static LlamaForwardPass newForwardPass(GgufFile file, LlamaConfig config) {
    return new LlamaForwardPass(
        config,
        LlamaWeights.fromGgufFile(file, config),
        new KvCache(
            config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
  }

  private static float[] runSequence(LlamaForwardPass forwardPass, int[] tokens) {
    float[] logits = null;
    for (int position = 0; position < tokens.length; position++) {
      logits = forwardPass.forward(tokens[position], position);
    }
    return logits.clone();
  }

  /**
   * Every tensor the dense and routed models share, drawn once so the two files differ in nothing
   * but how the feed-forward is stored. The {@code gate}/{@code up}/{@code down} entries are one
   * expert's worth, which the dense model uses as its only feed-forward and the routed model
   * repeats for every expert.
   */
  private static Map<String, byte[]> sharedTensors(Random rng, GgufTensorType type) {
    Map<String, byte[]> tensors = new LinkedHashMap<>();
    // Embeddings, norms and the router stay F32 exactly as a real routed model publishes them: only
    // the projections are quantized, and only they decide whether the batched path is eligible.
    tensors.put("token_embd.weight", randomF32(rng, VOCAB_SIZE * DIM));
    tensors.put("output_norm.weight", onesF32(DIM));
    tensors.put("output.weight", randomF32(rng, VOCAB_SIZE * DIM));
    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      tensors.put(prefix + "attn_norm.weight", onesF32(DIM));
      tensors.put(prefix + "attn_q.weight", projection(rng, type, DIM * DIM));
      tensors.put(prefix + "attn_k.weight", projection(rng, type, KV_DIM * DIM));
      tensors.put(prefix + "attn_v.weight", projection(rng, type, KV_DIM * DIM));
      tensors.put(prefix + "attn_q_norm.weight", onesF32(HEAD_DIM));
      tensors.put(prefix + "attn_k_norm.weight", onesF32(HEAD_DIM));
      tensors.put(prefix + "attn_output.weight", projection(rng, type, DIM * DIM));
      tensors.put(prefix + "ffn_norm.weight", onesF32(DIM));
      tensors.put(prefix + "gate", projection(rng, type, EXPERT_HIDDEN * DIM));
      tensors.put(prefix + "up", projection(rng, type, EXPERT_HIDDEN * DIM));
      tensors.put(prefix + "down", projection(rng, type, DIM * EXPERT_HIDDEN));
      tensors.put(prefix + "router", randomF32(rng, EXPERTS * DIM));
    }
    return tensors;
  }

  /** Random weights of one projection type. Q8_0 is the batched-eligible type used here. */
  private static byte[] projection(Random rng, GgufTensorType type, int valueCount) {
    if (type == GgufTensorType.F32) {
      return randomF32(rng, valueCount);
    }
    if (type != GgufTensorType.Q8_0) {
      throw new IllegalArgumentException("unsupported projection type: " + type);
    }
    if (valueCount % 32 != 0) {
      throw new IllegalArgumentException("Q8_0 value count must be a multiple of 32");
    }
    byte[] data = new byte[(valueCount / 32) * 34];
    rng.nextBytes(data);
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < valueCount / 32; block++) {
      buffer.putShort(block * 34, Float.floatToFloat16(0.01f));
    }
    return data;
  }

  private static GgufFile build(Map<String, byte[]> shared, boolean routed, GgufTensorType type) {
    return build(shared, routed, type, true, false);
  }

  private static GgufFile build(
      Map<String, byte[]> shared, boolean routed, GgufTensorType type, boolean repeatExperts) {
    return build(shared, routed, type, repeatExperts, false);
  }

  /**
   * @param shareableArena allocate the fixture where any thread may read it, as a mapped model file
   *     is. Whether segments are thread-shareable decides how far the planner's shareability probe
   *     walks a layer, so a confined fixture cannot reach the part of it that reads a layer's
   *     feed-forward at all.
   */
  private static GgufFile build(
      Map<String, byte[]> shared,
      boolean routed,
      GgufTensorType type,
      boolean repeatExperts,
      boolean shareableArena) {
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", ARCH)
            .addUint32(ARCH + ".embedding_length", DIM)
            .addUint32(ARCH + ".block_count", LAYERS)
            .addUint32(ARCH + ".attention.head_count", HEADS)
            .addUint32(ARCH + ".attention.head_count_kv", KV_HEADS)
            .addUint32(ARCH + ".vocab_size", VOCAB_SIZE)
            .addUint32(ARCH + ".context_length", CONTEXT)
            .addUint32(ARCH + ".feed_forward_length", routed ? UNUSED_DENSE_HIDDEN : EXPERT_HIDDEN);
    if (routed) {
      builder
          .addUint32(ARCH + ".expert_count", EXPERTS)
          .addUint32(ARCH + ".expert_used_count", USED)
          .addUint32(ARCH + ".expert_feed_forward_length", EXPERT_HIDDEN);
    }

    builder.addTensor(
        "token_embd.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        shared.get("token_embd.weight"));
    builder.addTensor(
        "output_norm.weight",
        GgufTensorType.F32,
        new long[] {DIM},
        shared.get("output_norm.weight"));
    builder.addTensor(
        "output.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        shared.get("output.weight"));

    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      builder.addTensor(
          prefix + "attn_norm.weight",
          GgufTensorType.F32,
          new long[] {DIM},
          shared.get(prefix + "attn_norm.weight"));
      builder.addTensor(
          prefix + "attn_q.weight",
          type,
          new long[] {DIM, DIM},
          shared.get(prefix + "attn_q.weight"));
      builder.addTensor(
          prefix + "attn_k.weight",
          type,
          new long[] {DIM, KV_DIM},
          shared.get(prefix + "attn_k.weight"));
      builder.addTensor(
          prefix + "attn_v.weight",
          type,
          new long[] {DIM, KV_DIM},
          shared.get(prefix + "attn_v.weight"));
      builder.addTensor(
          prefix + "attn_q_norm.weight",
          GgufTensorType.F32,
          new long[] {HEAD_DIM},
          shared.get(prefix + "attn_q_norm.weight"));
      builder.addTensor(
          prefix + "attn_k_norm.weight",
          GgufTensorType.F32,
          new long[] {HEAD_DIM},
          shared.get(prefix + "attn_k_norm.weight"));
      builder.addTensor(
          prefix + "attn_output.weight",
          type,
          new long[] {DIM, DIM},
          shared.get(prefix + "attn_output.weight"));
      builder.addTensor(
          prefix + "ffn_norm.weight",
          GgufTensorType.F32,
          new long[] {DIM},
          shared.get(prefix + "ffn_norm.weight"));

      byte[] gate = shared.get(prefix + "gate");
      byte[] up = shared.get(prefix + "up");
      byte[] down = shared.get(prefix + "down");
      if (routed) {
        builder.addTensor(
            prefix + "ffn_gate_inp.weight",
            GgufTensorType.F32,
            new long[] {DIM, EXPERTS},
            shared.get(prefix + "router"));
        builder.addTensor(
            prefix + "ffn_gate_exps.weight",
            type,
            new long[] {DIM, EXPERT_HIDDEN, EXPERTS},
            repeatExperts ? repeat(gate, EXPERTS) : gate);
        builder.addTensor(
            prefix + "ffn_up_exps.weight",
            type,
            new long[] {DIM, EXPERT_HIDDEN, EXPERTS},
            repeatExperts ? repeat(up, EXPERTS) : up);
        builder.addTensor(
            prefix + "ffn_down_exps.weight",
            type,
            new long[] {EXPERT_HIDDEN, DIM, EXPERTS},
            repeatExperts ? repeat(down, EXPERTS) : down);
      } else {
        builder.addTensor(prefix + "ffn_gate.weight", type, new long[] {DIM, EXPERT_HIDDEN}, gate);
        builder.addTensor(prefix + "ffn_up.weight", type, new long[] {DIM, EXPERT_HIDDEN}, up);
        builder.addTensor(prefix + "ffn_down.weight", type, new long[] {EXPERT_HIDDEN, DIM}, down);
      }
    }

    byte[] data = builder.build();
    // Confined by default: these fixtures are read on the test thread only.
    MemorySegment segment =
        (shareableArena ? Arena.ofAuto() : Arena.ofConfined()).allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return GgufParser.parseSegment(segment);
  }

  /**
   * One expert's weights repeated, which is what makes the routed model comparable to the dense.
   */
  private static byte[] repeat(byte[] block, int times) {
    byte[] all = new byte[block.length * times];
    for (int copy = 0; copy < times; copy++) {
      System.arraycopy(block, 0, all, copy * block.length, block.length);
    }
    return all;
  }

  /** The value filling expert {@code expert}'s block of a tensor tagged {@code tag}. */
  private static float mark(int tag, int expert) {
    return tag * 1000.0f + expert;
  }

  /** One expert's block, every element set to that expert's marker, for all {@link #EXPERTS}. */
  private static byte[] markedF32(int valuesPerExpert, int tag) {
    byte[] data = new byte[valuesPerExpert * EXPERTS * 4];
    ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int expert = 0; expert < EXPERTS; expert++) {
      for (int index = 0; index < valuesPerExpert; index++) {
        buf.putFloat((expert * valuesPerExpert + index) * 4, mark(tag, expert));
      }
    }
    return data;
  }

  /**
   * A routed model whose expert blocks are already stacked, so {@link #repeat} must not be applied.
   */
  private static GgufFile buildDistinctExperts(Map<String, byte[]> shared) {
    return build(shared, true, GgufTensorType.F32, false);
  }

  private static byte[] randomF32(Random rng, int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < count; i++) {
      buf.putFloat(i * 4, (rng.nextFloat() - 0.5f) * 0.1f);
    }
    return data;
  }

  private static byte[] onesF32(int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < count; i++) {
      buf.putFloat(i * 4, 1.0f);
    }
    return data;
  }
}
