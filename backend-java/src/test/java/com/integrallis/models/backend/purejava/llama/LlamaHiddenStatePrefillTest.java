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

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import com.integrallis.models.backend.purejava.plan.ExecutionPlanner;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import com.integrallis.models.backend.purejava.plan.PureJavaExecutionPlan;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.GgufQ6BatchedKernel;
import com.integrallis.vectors.core.GgufQ8BlockMajorKernel;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The hidden-state prefill must reach the batched path, and must not change a single observable
 * thing by doing so.
 *
 * <p>{@code prefill} has consulted {@code batchedPrefill} for a long time; {@code
 * prefillHiddenState} never did, so every hidden-state consumer -- embeddings through {@code
 * GgufEmbeddingBackend}, and any decision head reading a state -- ran the whole prompt one token at
 * a time. Measured on a real 3B checkpoint that is a fourteenfold difference, because a
 * single-token forward streams the entire weight set to produce one row.
 *
 * <p>Taking the batched path is asserted through a counter rather than inferred from timing, and it
 * is asserted in both directions: a prompt with nothing to batch must leave the counter alone. A
 * branch that were never taken would otherwise look exactly like a branch that did not matter.
 */
@Tag("unit")
class LlamaHiddenStatePrefillTest {

  private static final int DIM = 32;
  private static final int HEADS = 2;
  private static final int KV_HEADS = 1;
  private static final int HIDDEN_DIM = 64;
  private static final int VOCAB_SIZE = 32;
  private static final int LAYERS = 2;
  private static final int CONTEXT = 64;
  private static final float SIMD_REDUCTION_TOLERANCE = 2.0e-7f;

  private static final int[] PROMPT = {5, 7, 11, 13, 17, 19, 23, 29};

  @Test
  void theHiddenStateMatchesRunningTheSamePromptOneTokenAtATime() {
    Fixture fixture = new Fixture();
    float[] expected = sequentialHiddenState(fixture.pass(), PROMPT);
    float[] actual = fixture.newPass().prefillHiddenState(PROMPT, 0);

    assertThat(actual).hasSize(DIM);
    assertThat(actual).containsExactly(expected, within(SIMD_REDUCTION_TOLERANCE));
  }

  @Test
  void theBatchedPathIsTakenWhenThereIsSomethingToBatch() {
    LlamaForwardPass pass = new Fixture().pass();

    pass.prefillHiddenState(PROMPT, 0);

    assertThat(pass.batchedHiddenStatePrefills()).isEqualTo(1);
  }

  @Test
  void aPromptWithNothingToBatchLeavesTheCounterAlone() {
    LlamaForwardPass single = new Fixture().pass();
    single.prefillHiddenState(new int[] {5}, 0);
    assertThat(single.batchedHiddenStatePrefills()).isZero();

    LlamaForwardPass pair = new Fixture().pass();
    pair.prefillHiddenState(new int[] {5, 7}, 0);
    assertThat(pair.batchedHiddenStatePrefills()).isZero();
  }

  @Test
  void theKeyValueCacheEndsInTheSameStateAsRunningOneTokenAtATime() {
    Fixture baseline = new Fixture();
    sequentialHiddenState(baseline.pass(), PROMPT);
    float[] expectedKeys = baseline.cache().keyBuffer().clone();
    float[] expectedValues = baseline.cache().valueBuffer().clone();

    Fixture batched = new Fixture();
    batched.pass().prefillHiddenState(PROMPT, 0);

    assertThat(batched.cache().keyBuffer())
        .containsExactly(expectedKeys, within(SIMD_REDUCTION_TOLERANCE));
    assertThat(batched.cache().valueBuffer())
        .containsExactly(expectedValues, within(SIMD_REDUCTION_TOLERANCE));
  }

  @Test
  void generationContinuesFromTheBatchedPrefillExactlyAsItWouldHaveDone() {
    Fixture baseline = new Fixture();
    sequentialHiddenState(baseline.pass(), PROMPT);
    float[] expectedNext = baseline.pass().forward(31, PROMPT.length);

    Fixture batched = new Fixture();
    batched.pass().prefillHiddenState(PROMPT, 0);
    float[] actualNext = batched.pass().forward(31, PROMPT.length);

    assertThat(actualNext).containsExactly(expectedNext, within(SIMD_REDUCTION_TOLERANCE));
  }

  @Test
  void theSessionRouteMatchesOneTokenAtATimeAndTakesTheBatchedPathToo() {
    Fixture baseline = new Fixture();
    float[] expected = sequentialHiddenState(baseline.pass(), PROMPT);

    Fixture batched = new Fixture();
    LlamaForwardPass pass = batched.pass();
    LlamaForwardPass.Session session = pass.openSession();
    float[] actual = pass.prefillHiddenState(session, PROMPT, 0);

    assertThat(actual).containsExactly(expected, within(SIMD_REDUCTION_TOLERANCE));
    assertThat(pass.batchedHiddenStatePrefills()).isEqualTo(1);
    assertThat(session.checkpoint()).isEqualTo(PROMPT.length);
  }

  private static float[] sequentialHiddenState(LlamaForwardPass pass, int[] tokens) {
    float[] hidden = null;
    for (int position = 0; position < tokens.length; position++) {
      hidden = pass.prefillHiddenState(new int[] {tokens[position]}, position);
    }
    return hidden == null ? new float[0] : hidden.clone();
  }

  /** One nano model, one cache, one forward pass, built fresh so runs cannot contaminate. */
  private static final class Fixture {
    private final LlamaConfig config;
    private final LlamaWeights weights;
    private final KvCache cache;
    private final LlamaForwardPass pass;

    Fixture() {
      GgufFile file = buildNanoModel(new Random(42));
      this.config = LlamaConfig.fromMetadata(file.metadata());
      this.weights = LlamaWeights.fromGgufFile(file, config);
      this.cache =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      this.pass = new LlamaForwardPass(config, weights, cache, batchedPlan(config, weights));
    }

    LlamaForwardPass pass() {
      return pass;
    }

    KvCache cache() {
      return cache;
    }

    LlamaForwardPass newPass() {
      return new LlamaForwardPass(
          config,
          weights,
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()),
          batchedPlan(config, weights));
    }
  }

  /**
   * A plan pinned to a batch size of four. The default plan consults the host fingerprint, so a
   * test relying on it would assert one thing on this workstation and another on a benchmark host.
   */
  private static PureJavaExecutionPlan batchedPlan(LlamaConfig config, LlamaWeights weights) {
    return ExecutionPlanner.plan(
        RuntimeFingerprint.capture(),
        ModelTopology.from("llama", config, weights),
        new PureJavaPlanConfiguration(
            true,
            true,
            GgufQ4Kernel.WIDENED,
            GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
            4,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            GgufQ8BlockMajorKernel.SCATTERED,
            false,
            PureJavaPlanConfiguration.MODEL_MAXIMUM_CONTEXT));
  }

  private static GgufFile buildNanoModel(Random rng) {
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addUint32("llama.embedding_length", DIM)
            .addUint32("llama.block_count", LAYERS)
            .addUint32("llama.attention.head_count", HEADS)
            .addUint32("llama.attention.head_count_kv", KV_HEADS)
            .addUint32("llama.vocab_size", VOCAB_SIZE)
            .addUint32("llama.context_length", CONTEXT)
            .addUint32("llama.feed_forward_length", HIDDEN_DIM);
    int headDim = DIM / HEADS;
    builder.addTensor(
        "token_embd.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        randomF32(rng, VOCAB_SIZE * DIM));
    builder.addTensor("output_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
    builder.addTensor(
        "output.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        randomF32(rng, VOCAB_SIZE * DIM));
    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      builder.addTensor(
          prefix + "attn_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
      builder.addTensor(
          prefix + "attn_q.weight",
          GgufTensorType.Q8_0,
          new long[] {DIM, DIM},
          randomQ8(rng, DIM * DIM));
      builder.addTensor(
          prefix + "attn_k.weight",
          GgufTensorType.Q8_0,
          new long[] {DIM, KV_HEADS * headDim},
          randomQ8(rng, KV_HEADS * headDim * DIM));
      builder.addTensor(
          prefix + "attn_v.weight",
          GgufTensorType.Q8_0,
          new long[] {DIM, KV_HEADS * headDim},
          randomQ8(rng, KV_HEADS * headDim * DIM));
      builder.addTensor(
          prefix + "attn_output.weight",
          GgufTensorType.Q8_0,
          new long[] {DIM, DIM},
          randomQ8(rng, DIM * DIM));
      builder.addTensor(
          prefix + "ffn_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
      builder.addTensor(
          prefix + "ffn_gate.weight",
          GgufTensorType.Q8_0,
          new long[] {DIM, HIDDEN_DIM},
          randomQ8(rng, HIDDEN_DIM * DIM));
      builder.addTensor(
          prefix + "ffn_up.weight",
          GgufTensorType.Q8_0,
          new long[] {DIM, HIDDEN_DIM},
          randomQ8(rng, HIDDEN_DIM * DIM));
      builder.addTensor(
          prefix + "ffn_down.weight",
          GgufTensorType.Q8_0,
          new long[] {HIDDEN_DIM, DIM},
          randomQ8(rng, DIM * HIDDEN_DIM));
    }
    byte[] data = builder.build();
    MemorySegment segment = Arena.ofConfined().allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return GgufParser.parseSegment(segment);
  }

  private static byte[] randomF32(Random rng, int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < count; index++) {
      buffer.putFloat(index * 4, (rng.nextFloat() - 0.5f) * 0.1f);
    }
    return data;
  }

  private static byte[] randomQ8(Random rng, int valueCount) {
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

  private static byte[] onesF32(int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < count; index++) {
      buffer.putFloat(index * 4, 1.0f);
    }
    return data;
  }
}
