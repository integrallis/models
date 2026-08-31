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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.plan.ExecutionPlanner;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import com.integrallis.models.backend.purejava.plan.PureJavaExecutionPlan;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.GgufQ6BatchedKernel;
import com.integrallis.vectors.core.GgufQ8BlockMajorKernel;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntFunction;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class LlamaForwardPassTest {

  // Nano model: 2 layers, dim=16, heads=2, kvHeads=1, hiddenDim=32, vocabSize=32
  private static final int DIM = 16;
  private static final int HEADS = 2;
  private static final int KV_HEADS = 1;
  private static final int HIDDEN_DIM = 32;
  private static final int VOCAB_SIZE = 32;
  private static final int LAYERS = 2;
  private static final int CONTEXT = 64;
  private static final float SIMD_REDUCTION_TOLERANCE = 2.0e-7f;

  private GgufFile buildNanoModel(Random rng) {
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addUint32("llama.embedding_length", DIM)
            .addUint32("llama.block_count", LAYERS)
            .addUint32("llama.attention.head_count", HEADS)
            .addUint32("llama.attention.head_count_kv", KV_HEADS)
            .addUint32("llama.vocab_size", VOCAB_SIZE)
            .addUint32("llama.context_length", CONTEXT)
            .addUint32("llama.feed_forward_length", HIDDEN_DIM);

    // token_embd.weight: [vocab_size x dim]
    builder.addTensor(
        "token_embd.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        randomF32(rng, VOCAB_SIZE * DIM));
    // output_norm.weight: [dim]
    builder.addTensor("output_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
    // output.weight: [vocab_size x dim]
    builder.addTensor(
        "output.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        randomF32(rng, VOCAB_SIZE * DIM));

    for (int l = 0; l < LAYERS; l++) {
      String prefix = "blk." + l + ".";
      builder.addTensor(
          prefix + "attn_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
      builder.addTensor(
          prefix + "attn_q.weight",
          GgufTensorType.F32,
          new long[] {DIM, DIM},
          randomF32(rng, DIM * DIM));
      builder.addTensor(
          prefix + "attn_k.weight",
          GgufTensorType.F32,
          new long[] {DIM, KV_HEADS * (DIM / HEADS)},
          randomF32(rng, KV_HEADS * (DIM / HEADS) * DIM));
      builder.addTensor(
          prefix + "attn_v.weight",
          GgufTensorType.F32,
          new long[] {DIM, KV_HEADS * (DIM / HEADS)},
          randomF32(rng, KV_HEADS * (DIM / HEADS) * DIM));
      builder.addTensor(
          prefix + "attn_output.weight",
          GgufTensorType.F32,
          new long[] {DIM, DIM},
          randomF32(rng, DIM * DIM));
      builder.addTensor(
          prefix + "ffn_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
      builder.addTensor(
          prefix + "ffn_gate.weight",
          GgufTensorType.F32,
          new long[] {DIM, HIDDEN_DIM},
          randomF32(rng, HIDDEN_DIM * DIM));
      builder.addTensor(
          prefix + "ffn_up.weight",
          GgufTensorType.F32,
          new long[] {DIM, HIDDEN_DIM},
          randomF32(rng, HIDDEN_DIM * DIM));
      builder.addTensor(
          prefix + "ffn_down.weight",
          GgufTensorType.F32,
          new long[] {HIDDEN_DIM, DIM},
          randomF32(rng, DIM * HIDDEN_DIM));
    }

    byte[] data = builder.build();
    var segment = Arena.ofConfined().allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return GgufParser.parseSegment(segment);
  }

  private GgufFile buildQ4NanoModel(Random rng) {
    return buildQuantizedNanoModel(
        rng, GgufTensorType.Q4_0, 32, 64, valueCount -> randomQ4(rng, valueCount));
  }

  private static GgufFile copyToSharedArena(GgufFile source, Arena arena) {
    MemorySegment segment = arena.allocate(source.fileSegment().byteSize());
    segment.copyFrom(source.fileSegment());
    return GgufParser.parseSegment(segment);
  }

  private GgufFile buildQ8NanoModel(Random rng) {
    return buildQuantizedNanoModel(
        rng, GgufTensorType.Q8_0, 32, 64, valueCount -> randomQ8(rng, valueCount));
  }

  private GgufFile buildQ5NanoModel(Random rng) {
    return buildQuantizedNanoModel(
        rng, GgufTensorType.Q5_0, 32, 64, valueCount -> randomQ5(rng, valueCount), true);
  }

  private GgufFile buildQ4KNanoModel(Random rng) {
    return buildQuantizedNanoModel(
        rng, GgufTensorType.Q4_K, 256, 256, valueCount -> randomQ4K(rng, valueCount));
  }

  private GgufFile buildMixedQ4KQ6KNanoModel(Random rng) {
    return buildQuantizedNanoModel(
        rng,
        GgufTensorType.Q4_K,
        256,
        256,
        valueCount -> randomQ4K(rng, valueCount),
        GgufTensorType.Q6_K,
        valueCount -> randomQ6K(rng, valueCount));
  }

  private GgufFile buildMixedQ4KQ6KQuantizedOutputNanoModel(Random rng) {
    return buildQuantizedNanoModel(
        rng,
        GgufTensorType.Q4_K,
        256,
        256,
        valueCount -> randomQ4K(rng, valueCount),
        GgufTensorType.Q6_K,
        valueCount -> randomQ6K(rng, valueCount),
        false,
        true);
  }

  private GgufFile buildQ4QuantizedOutputNanoModel(Random rng) {
    return buildQuantizedNanoModel(
        rng,
        GgufTensorType.Q4_0,
        256,
        256,
        valueCount -> randomQ4(rng, valueCount),
        GgufTensorType.Q4_0,
        valueCount -> randomQ4(rng, valueCount),
        false,
        true);
  }

  private GgufFile buildMixedQ5KQ6KNanoModel(Random rng) {
    return buildQuantizedNanoModel(
        rng,
        GgufTensorType.Q5_K,
        256,
        256,
        valueCount -> randomQ5K(rng, valueCount),
        GgufTensorType.Q6_K,
        valueCount -> randomQ6K(rng, valueCount));
  }

  private GgufFile buildQuantizedNanoModel(
      Random rng,
      GgufTensorType quantizedType,
      int dim,
      int hiddenDim,
      IntFunction<byte[]> quantizedData) {
    return buildQuantizedNanoModel(rng, quantizedType, dim, hiddenDim, quantizedData, false);
  }

  private GgufFile buildQuantizedNanoModel(
      Random rng,
      GgufTensorType quantizedType,
      int dim,
      int hiddenDim,
      IntFunction<byte[]> quantizedData,
      boolean quantizedEmbedding) {
    return buildQuantizedNanoModel(
        rng,
        quantizedType,
        dim,
        hiddenDim,
        quantizedData,
        quantizedType,
        quantizedData,
        quantizedEmbedding,
        false);
  }

  private GgufFile buildQuantizedNanoModel(
      Random rng,
      GgufTensorType quantizedType,
      int dim,
      int hiddenDim,
      IntFunction<byte[]> quantizedData,
      GgufTensorType secondaryType,
      IntFunction<byte[]> secondaryData) {
    return buildQuantizedNanoModel(
        rng, quantizedType, dim, hiddenDim, quantizedData, secondaryType, secondaryData, false);
  }

  private GgufFile buildQuantizedNanoModel(
      Random rng,
      GgufTensorType quantizedType,
      int dim,
      int hiddenDim,
      IntFunction<byte[]> quantizedData,
      GgufTensorType secondaryType,
      IntFunction<byte[]> secondaryData,
      boolean quantizedEmbedding) {
    return buildQuantizedNanoModel(
        rng,
        quantizedType,
        dim,
        hiddenDim,
        quantizedData,
        secondaryType,
        secondaryData,
        quantizedEmbedding,
        false);
  }

  private GgufFile buildQuantizedNanoModel(
      Random rng,
      GgufTensorType quantizedType,
      int dim,
      int hiddenDim,
      IntFunction<byte[]> quantizedData,
      GgufTensorType secondaryType,
      IntFunction<byte[]> secondaryData,
      boolean quantizedEmbedding,
      boolean quantizedOutput) {
    int headDim = dim / HEADS;
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "qwen3")
            .addUint32("llama.embedding_length", dim)
            .addUint32("llama.block_count", LAYERS)
            .addUint32("llama.attention.head_count", HEADS)
            .addUint32("llama.attention.head_count_kv", KV_HEADS)
            .addUint32("llama.vocab_size", VOCAB_SIZE)
            .addUint32("llama.context_length", CONTEXT)
            .addUint32("llama.feed_forward_length", hiddenDim)
            .addTensor(
                "token_embd.weight",
                quantizedEmbedding ? quantizedType : GgufTensorType.F32,
                new long[] {dim, VOCAB_SIZE},
                quantizedEmbedding
                    ? quantizedData.apply(VOCAB_SIZE * dim)
                    : randomF32(rng, VOCAB_SIZE * dim))
            .addTensor("output_norm.weight", GgufTensorType.F32, new long[] {dim}, onesF32(dim))
            .addTensor(
                "output.weight",
                quantizedOutput ? secondaryType : GgufTensorType.F32,
                new long[] {dim, VOCAB_SIZE},
                quantizedOutput
                    ? secondaryData.apply(VOCAB_SIZE * dim)
                    : randomF32(rng, VOCAB_SIZE * dim));

    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      builder
          .addTensor(
              prefix + "attn_norm.weight", GgufTensorType.F32, new long[] {dim}, onesF32(dim))
          .addTensor(
              prefix + "attn_q.weight",
              quantizedType,
              new long[] {dim, dim},
              quantizedData.apply(dim * dim))
          .addTensor(
              prefix + "attn_q.bias", GgufTensorType.F32, new long[] {dim}, randomF32(rng, dim))
          .addTensor(
              prefix + "attn_q_norm.weight",
              GgufTensorType.F32,
              new long[] {headDim},
              onesF32(headDim))
          .addTensor(
              prefix + "attn_k.weight",
              quantizedType,
              new long[] {dim, headDim},
              quantizedData.apply(headDim * dim))
          .addTensor(
              prefix + "attn_k.bias",
              GgufTensorType.F32,
              new long[] {headDim},
              randomF32(rng, headDim))
          .addTensor(
              prefix + "attn_k_norm.weight",
              GgufTensorType.F32,
              new long[] {headDim},
              onesF32(headDim))
          .addTensor(
              prefix + "attn_v.weight",
              secondaryType,
              new long[] {dim, headDim},
              secondaryData.apply(headDim * dim))
          .addTensor(
              prefix + "attn_v.bias",
              GgufTensorType.F32,
              new long[] {headDim},
              randomF32(rng, headDim))
          .addTensor(
              prefix + "attn_output.weight",
              secondaryType,
              new long[] {dim, dim},
              secondaryData.apply(dim * dim))
          .addTensor(prefix + "ffn_norm.weight", GgufTensorType.F32, new long[] {dim}, onesF32(dim))
          .addTensor(
              prefix + "ffn_gate.weight",
              quantizedType,
              new long[] {dim, hiddenDim},
              quantizedData.apply(hiddenDim * dim))
          .addTensor(
              prefix + "ffn_up.weight",
              quantizedType,
              new long[] {dim, hiddenDim},
              quantizedData.apply(hiddenDim * dim))
          .addTensor(
              prefix + "ffn_down.weight",
              secondaryType,
              new long[] {hiddenDim, dim},
              secondaryData.apply(dim * hiddenDim));
    }

    byte[] data = builder.build();
    MemorySegment segment = Arena.ofConfined().allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return GgufParser.parseSegment(segment);
  }

  @Nested
  class NanoModel {

    @Test
    void rejectsAnExecutionPlanForDifferentWeights() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      ModelTopology.LayerTopology q4Layer =
          new ModelTopology.LayerTopology(
              GgufTensorType.Q4_0,
              GgufTensorType.Q4_0,
              GgufTensorType.Q4_0,
              GgufTensorType.Q4_0,
              GgufTensorType.Q4_0,
              GgufTensorType.Q4_0,
              GgufTensorType.Q4_0);
      var mismatchedPlan =
          ExecutionPlanner.plan(
              RuntimeFingerprint.capture(),
              new ModelTopology(
                  "llama",
                  config.queryDim(),
                  config.keyDim(),
                  config.valueDim(),
                  java.util.Collections.nCopies(config.numLayers(), q4Layer),
                  false),
              PureJavaPlanConfiguration.defaults());

      assertThatThrownBy(
              () ->
                  new LlamaForwardPass(
                      config,
                      weights,
                      new KvCache(
                          config.numLayers(),
                          config.contextLength(),
                          config.keyDim(),
                          config.valueDim()),
                      mismatchedPlan))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("execution plan topology");
    }

    @Test
    void producesLogitsOfVocabSize() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      KvCache cache =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);

      float[] logits = forwardPass.forward(1, 0);

      assertThat(logits).hasSize(VOCAB_SIZE);
    }

    @Test
    void deterministicForSameInput() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);

      KvCache cache1 =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass pass1 = new LlamaForwardPass(config, weights, cache1);
      float[] logits1 = pass1.forward(5, 0);

      KvCache cache2 =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass pass2 = new LlamaForwardPass(config, weights, cache2);
      float[] logits2 = pass2.forward(5, 0);

      assertThat(logits1).containsExactly(logits2);
    }

    @Test
    void prefillMatchesTokenAtATimeForwardPass() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13};

      LlamaForwardPass sequential =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] expected = runSequence(sequential, tokens);

      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] actual = batched.prefill(tokens, 0);

      assertThat(actual).containsExactly(expected, within(SIMD_REDUCTION_TOLERANCE));
      assertThat(batched.forward(17, tokens.length)).hasSize(VOCAB_SIZE);
    }

    @Test
    void hiddenStateHasEmbeddingWidthNotVocabularyWidth() {
      // The whole point of the embedding path: skip the vocabulary projection, which is both the
      // widest matmul in the pass and meaningless for a sentence vector.
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaForwardPass forwardPass =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));

      float[] hidden = forwardPass.hiddenState(1, 0);

      assertThat(hidden).hasSize(DIM);
      assertThat(DIM).isNotEqualTo(VOCAB_SIZE);
    }

    @Test
    void hiddenStateIsDeterministicAndInputDependent() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);

      float[] first = freshPass(config, weights).hiddenState(5, 0);
      float[] repeat = freshPass(config, weights).hiddenState(5, 0);
      float[] other = freshPass(config, weights).hiddenState(9, 0);

      assertThat(first).containsExactly(repeat);
      assertThat(first).isNotEqualTo(other);
    }

    @Test
    void hiddenStateRunsTheFinalLayerRatherThanTheKeyValueOnlyShortcut() {
      // Prefill skips the last layer's attention output and FFN when logits are discarded. The
      // embedding path needs exactly that discarded activation, so it must not take the shortcut:
      // if it did, this would come back null or all-zero.
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);

      float[] hidden = freshPass(config, weights).prefillHiddenState(new int[] {5, 7, 11}, 0);

      assertThat(hidden).hasSize(DIM);
      assertThat(hidden).isNotNull();
      boolean allZero = true;
      for (float value : hidden) {
        assertThat(Float.isFinite(value)).isTrue();
        if (value != 0.0f) {
          allZero = false;
        }
      }
      assertThat(allZero).isFalse();
    }

    @Test
    void prefillHiddenStateMatchesSteppingTokenAtATime() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13};

      LlamaForwardPass stepwise = freshPass(config, weights);
      float[] expected = null;
      for (int index = 0; index < tokens.length; index++) {
        expected = stepwise.hiddenState(tokens[index], index);
      }

      float[] actual = freshPass(config, weights).prefillHiddenState(tokens, 0);

      assertThat(actual).containsExactly(expected);
    }

    @Test
    void hiddenStateLeavesTheSequenceUsableForGeneration() {
      // Embedding a prompt must advance cache state exactly like an ordinary prefill, so a caller
      // can keep generating afterwards.
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11};

      LlamaForwardPass forwardPass = freshPass(config, weights);
      forwardPass.prefillHiddenState(tokens, 0);

      assertThat(forwardPass.forward(17, tokens.length)).hasSize(VOCAB_SIZE);
    }

    @Test
    void logitsPathIsUnchangedByTheEmbeddingPath() {
      // Regression guard for the shared internal switch: asking for a hidden state must not alter
      // what the logits path produces for the same input.
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13};

      float[] before = freshPass(config, weights).prefill(tokens, 0);
      freshPass(config, weights).prefillHiddenState(tokens, 0);
      float[] after = freshPass(config, weights).prefill(tokens, 0);

      assertThat(after).containsExactly(before);
    }

    private LlamaForwardPass freshPass(LlamaConfig config, LlamaWeights weights) {
      return new LlamaForwardPass(
          config,
          weights,
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
    }

    @Test
    void q4PrefillUsesBatchedKernelAndPreservesAutoregressiveState() {
      GgufFile file = buildQ4NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13, 17, 19, 23, 29};

      LlamaForwardPass sequential =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] expected = runSequence(sequential, tokens);

      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] actual = batched.prefill(tokens, 0);

      assertThat(batched.usesBatchedPrefill()).isTrue();
      assertThat(batched.usesGroupedBatchedPrefill()).isTrue();
      assertThat(actual).containsExactly(expected);
      assertThat(batched.forward(3, tokens.length))
          .containsExactly(sequential.forward(3, tokens.length));
    }

    @Test
    void injectedQ4KernelPreservesPrefillAndAutoregressiveState() {
      GgufFile file = buildQ4NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13, 17, 19, 23, 29};
      PureJavaExecutionPlan plan =
          ExecutionPlanner.plan(
              RuntimeFingerprint.capture(),
              ModelTopology.from("qwen3", config, weights),
              new PureJavaPlanConfiguration(
                  false,
                  false,
                  GgufQ4Kernel.WIDENED,
                  GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
                  32,
                  true,
                  true,
                  false,
                  false,
                  false,
                  false,
                  false,
                  GgufQ8BlockMajorKernel.SCATTERED,
                  false));
      LlamaForwardPass baseline =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()),
              plan);
      float[] expected = baseline.prefill(tokens, 0).clone();
      int nextToken = argmax(expected);
      float[] expectedNext = baseline.forward(nextToken, tokens.length);
      AtomicInteger invocations = new AtomicInteger();
      GgufBatchedMatrixKernel kernel =
          new GgufBatchedMatrixKernel() {
            @Override
            public boolean supports(GgufTensorType type) {
              return type == GgufTensorType.Q4_0;
            }

            @Override
            public void multiply(
                float[] output,
                float[] input,
                MemorySegment weights,
                GgufTensorType type,
                int batchSize,
                int rows,
                int cols) {
              invocations.incrementAndGet();
              TensorOps.ggufBatchedMatmul(
                  output,
                  input,
                  weights,
                  type,
                  batchSize,
                  rows,
                  cols,
                  new byte[batchSize * cols],
                  new float[batchSize * cols / 32],
                  new int[batchSize * cols / 4],
                  new short[batchSize * cols / 16],
                  new float[batchSize * rows * 8],
                  GgufQ4Kernel.WIDENED);
            }
          };
      LlamaForwardPass accelerated =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()),
              plan,
              kernel);

      float[] actual = accelerated.prefill(tokens, 0);

      assertThat(invocations).hasValueGreaterThan(0);
      assertThat(actual).containsExactly(expected);
      assertThat(accelerated.forward(nextToken, tokens.length)).containsExactly(expectedNext);
    }

    @Test
    void injectedAttentionKernelPreservesPrefillAndFallsBackForDecode() {
      GgufFile file = buildQ4NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13, 17, 19, 23, 29};
      PureJavaExecutionPlan plan = executionPlan(config, weights, tokens.length, false);
      LlamaForwardPass baseline =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()),
              plan);
      float[] expected = baseline.prefill(tokens, 0).clone();
      int nextToken = argmax(expected);
      float[] expectedNext = baseline.forward(nextToken, tokens.length);
      AtomicInteger invocations = new AtomicInteger();
      AtomicInteger resets = new AtomicInteger();
      float[][] keyCaches = new float[LAYERS][CONTEXT * config.keyDim()];
      float[][] valueCaches = new float[LAYERS][CONTEXT * config.valueDim()];
      BatchedCausalAttentionKernel attentionKernel =
          new BatchedCausalAttentionKernel() {
            @Override
            public boolean isEligible(
                int layer,
                int startPosition,
                int batchSize,
                int numHeads,
                int numKvHeads,
                int keyLength,
                int valueLength,
                int maxSequenceLength,
                int slidingWindow) {
              return batchSize > 1;
            }

            @Override
            public void attend(
                float[] output,
                float[] query,
                float[] key,
                float[] value,
                int layer,
                int startPosition,
                int batchSize,
                int numHeads,
                int numKvHeads,
                int keyLength,
                int valueLength,
                int maxSequenceLength,
                int slidingWindow) {
              invocations.incrementAndGet();
              referenceBatchedAttention(
                  output,
                  query,
                  key,
                  value,
                  keyCaches[layer],
                  valueCaches[layer],
                  startPosition,
                  batchSize,
                  numHeads,
                  numKvHeads,
                  keyLength,
                  valueLength,
                  slidingWindow);
            }

            @Override
            public void reset() {
              resets.incrementAndGet();
            }
          };
      LlamaForwardPass accelerated =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()),
              plan,
              GgufBatchedMatrixKernel.none(),
              attentionKernel);

      float[] actual = accelerated.prefill(tokens, 0);

      assertThat(invocations).hasValue(LAYERS);
      assertThat(actual).containsExactly(expected);
      assertThat(accelerated.forward(nextToken, tokens.length)).containsExactly(expectedNext);
      assertThat(invocations).hasValue(LAYERS);
      accelerated.reset();
      assertThat(resets).hasValue(1);
    }

    @Test
    void injectedKernelExecutesEligibleSingleTokenProjectionGroups() {
      GgufFile file = buildQ4NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      PureJavaExecutionPlan plan = executionPlan(config, weights, 1, false);
      LlamaForwardPass baseline =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()),
              plan);
      float[] expected = baseline.forward(5, 0).clone();
      AtomicInteger singleInvocations = new AtomicInteger();
      AtomicInteger dualInvocations = new AtomicInteger();
      AtomicInteger tripleInvocations = new AtomicInteger();
      GgufBatchedMatrixKernel kernel =
          new GgufBatchedMatrixKernel() {
            @Override
            public boolean supports(GgufTensorType type) {
              return type == GgufTensorType.Q4_0;
            }

            @Override
            public boolean isEligible(GgufTensorType type, int batchSize, int rows, int cols) {
              return batchSize == 1 && supports(type);
            }

            @Override
            public boolean isDualEligible(
                GgufTensorType firstType,
                int firstRows,
                GgufTensorType secondType,
                int secondRows,
                int batchSize,
                int cols) {
              return batchSize == 1 && supports(firstType) && supports(secondType);
            }

            @Override
            public boolean isTripleEligible(
                GgufTensorType firstType,
                int firstRows,
                GgufTensorType secondType,
                int secondRows,
                GgufTensorType thirdType,
                int thirdRows,
                int batchSize,
                int cols) {
              return batchSize == 1
                  && supports(firstType)
                  && supports(secondType)
                  && supports(thirdType);
            }

            @Override
            public void multiply(
                float[] output,
                float[] input,
                MemorySegment weights,
                GgufTensorType type,
                int batchSize,
                int rows,
                int cols) {
              singleInvocations.incrementAndGet();
              multiplyQ4(output, input, weights, batchSize, rows, cols);
            }

            @Override
            public void multiplyDual(
                float[] firstOutput,
                MemorySegment firstWeights,
                GgufTensorType firstType,
                int firstRows,
                float[] secondOutput,
                MemorySegment secondWeights,
                GgufTensorType secondType,
                int secondRows,
                float[] input,
                int batchSize,
                int cols) {
              dualInvocations.incrementAndGet();
              multiplyQ4(firstOutput, input, firstWeights, batchSize, firstRows, cols);
              multiplyQ4(secondOutput, input, secondWeights, batchSize, secondRows, cols);
            }

            @Override
            public void multiplyTriple(
                float[] firstOutput,
                MemorySegment firstWeights,
                GgufTensorType firstType,
                int firstRows,
                float[] secondOutput,
                MemorySegment secondWeights,
                GgufTensorType secondType,
                int secondRows,
                float[] thirdOutput,
                MemorySegment thirdWeights,
                GgufTensorType thirdType,
                int thirdRows,
                float[] input,
                int batchSize,
                int cols) {
              tripleInvocations.incrementAndGet();
              multiplyQ4(firstOutput, input, firstWeights, batchSize, firstRows, cols);
              multiplyQ4(secondOutput, input, secondWeights, batchSize, secondRows, cols);
              multiplyQ4(thirdOutput, input, thirdWeights, batchSize, thirdRows, cols);
            }
          };
      LlamaForwardPass accelerated =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()),
              plan,
              kernel);

      float[] actual = accelerated.forward(5, 0);

      assertThat(actual).containsExactly(expected, within(SIMD_REDUCTION_TOLERANCE));
      assertThat(singleInvocations).hasValue(LAYERS * 2);
      assertThat(dualInvocations).hasValue(LAYERS);
      assertThat(tripleInvocations).hasValue(LAYERS);
    }

    @Test
    void stagedQ4FfnPreservesPrefillCacheAndAutoregressiveStateExactly() {
      try (Arena arena = Arena.ofShared()) {
        GgufFile file = copyToSharedArena(buildQ4NanoModel(new Random(42)), arena);
        LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
        LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
        int[] tokens = new int[40];
        for (int index = 0; index < tokens.length; index++) {
          tokens[index] = (index * 7 + 5) % VOCAB_SIZE;
        }

        KvCache baselineCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass baseline =
            new LlamaForwardPass(
                config, weights, baselineCache, executionPlan(config, weights, 32, true, true));
        float[] expected = baseline.prefill(tokens, 0).clone();
        float[] expectedKeys = baselineCache.keyBuffer().clone();
        float[] expectedValues = baselineCache.valueBuffer().clone();
        int nextToken = argmax(expected);
        float[] expectedNext = baseline.forward(nextToken, tokens.length);

        PureJavaPlanConfiguration configuration =
            new PureJavaPlanConfiguration(
                true,
                true,
                GgufQ4Kernel.WIDENED,
                GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
                32,
                true,
                true,
                false,
                false,
                true,
                false,
                false,
                GgufQ8BlockMajorKernel.SCATTERED,
                false);
        KvCache stagedCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass staged =
            new LlamaForwardPass(
                config,
                weights,
                stagedCache,
                ExecutionPlanner.plan(
                    RuntimeFingerprint.capture(),
                    ModelTopology.from("llama", config, weights),
                    configuration));

        float[] actual = staged.prefill(tokens, 0);

        assertThat(staged.usesStagedQuantizedFfn()).isTrue();
        assertThat(actual).containsExactly(expected);
        assertThat(stagedCache.keyBuffer()).containsExactly(expectedKeys);
        assertThat(stagedCache.valueBuffer()).containsExactly(expectedValues);
        assertThat(staged.forward(nextToken, tokens.length)).containsExactly(expectedNext);
      }
    }

    @Test
    void stagedQ4LayerPreservesPrefillCacheAndAutoregressiveStateExactly() {
      try (Arena arena = Arena.ofShared()) {
        GgufFile file = copyToSharedArena(buildQ4NanoModel(new Random(42)), arena);
        LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
        LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
        int[] tokens = new int[40];
        for (int index = 0; index < tokens.length; index++) {
          tokens[index] = (index * 7 + 5) % VOCAB_SIZE;
        }

        PureJavaPlanConfiguration ffnConfiguration =
            new PureJavaPlanConfiguration(
                true,
                true,
                GgufQ4Kernel.WIDENED,
                GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
                32,
                true,
                true,
                false,
                false,
                true,
                false,
                false,
                GgufQ8BlockMajorKernel.SCATTERED,
                false);
        KvCache baselineCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass baseline =
            new LlamaForwardPass(
                config,
                weights,
                baselineCache,
                ExecutionPlanner.plan(
                    RuntimeFingerprint.capture(),
                    ModelTopology.from("llama", config, weights),
                    ffnConfiguration));
        float[] expected = baseline.prefill(tokens, 0).clone();
        float[] expectedKeys = baselineCache.keyBuffer().clone();
        float[] expectedValues = baselineCache.valueBuffer().clone();
        int nextToken = argmax(expected);
        float[] expectedNext = baseline.forward(nextToken, tokens.length);

        PureJavaPlanConfiguration layerConfiguration =
            new PureJavaPlanConfiguration(
                true,
                true,
                GgufQ4Kernel.WIDENED,
                GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
                32,
                true,
                true,
                false,
                false,
                true,
                true,
                false,
                GgufQ8BlockMajorKernel.SCATTERED,
                false);
        KvCache stagedCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass staged =
            new LlamaForwardPass(
                config,
                weights,
                stagedCache,
                ExecutionPlanner.plan(
                    RuntimeFingerprint.capture(),
                    ModelTopology.from("llama", config, weights),
                    layerConfiguration));

        float[] actual = staged.prefill(tokens, 0);

        assertThat(baseline.usesStagedQuantizedFfn()).isTrue();
        assertThat(baseline.usesStagedQuantizedLayer()).isFalse();
        assertThat(staged.usesStagedQuantizedLayer()).isTrue();
        assertThat(staged.stagedQuantizedLayerStageCount()).isEqualTo(7);
        assertThat(actual).containsExactly(expected);
        assertThat(stagedCache.keyBuffer()).containsExactly(expectedKeys);
        assertThat(stagedCache.valueBuffer()).containsExactly(expectedValues);
        assertThat(staged.forward(nextToken, tokens.length)).containsExactly(expectedNext);
      }
    }

    @Test
    void finalLayerPrefillPruningPreservesLogitsKvCacheAndAutoregressiveState() {
      GgufFile file = buildQ4NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = new int[40];
      for (int index = 0; index < tokens.length; index++) {
        tokens[index] = (index * 7 + 5) % VOCAB_SIZE;
      }

      KvCache baselineCache =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass baseline =
          new LlamaForwardPass(
              config, weights, baselineCache, executionPlan(config, weights, 32, false));
      float[] expected = baseline.prefill(tokens, 0).clone();
      float[] expectedKeys = baselineCache.keyBuffer().clone();
      float[] expectedValues = baselineCache.valueBuffer().clone();
      int nextToken = argmax(expected);
      float[] expectedNext = baseline.forward(nextToken, tokens.length);

      KvCache prunedCache =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass pruned =
          new LlamaForwardPass(
              config, weights, prunedCache, executionPlan(config, weights, 32, true));
      float[] actual = pruned.prefill(tokens, 0);

      assertThat(pruned.usesFinalLayerPrefillPruning()).isTrue();
      assertThat(actual).containsExactly(expected);
      assertThat(prunedCache.keyBuffer()).containsExactly(expectedKeys);
      assertThat(prunedCache.valueBuffer()).containsExactly(expectedValues);
      assertThat(pruned.forward(nextToken, tokens.length)).containsExactly(expectedNext);
    }

    @Test
    void sequentialFinalLayerPrefillPruningPreservesStateExactly() {
      GgufFile file = buildQ4NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13, 17, 19, 23, 29};

      KvCache baselineCache =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass baseline =
          new LlamaForwardPass(
              config, weights, baselineCache, executionPlan(config, weights, 1, false));
      float[] expected = baseline.prefill(tokens, 0).clone();
      float[] expectedKeys = baselineCache.keyBuffer().clone();
      float[] expectedValues = baselineCache.valueBuffer().clone();
      int nextToken = argmax(expected);
      float[] expectedNext = baseline.forward(nextToken, tokens.length);

      KvCache prunedCache =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass pruned =
          new LlamaForwardPass(
              config, weights, prunedCache, executionPlan(config, weights, 1, true));
      float[] actual = pruned.prefill(tokens, 0);

      assertThat(pruned.usesBatchedPrefill()).isFalse();
      assertThat(actual).containsExactly(expected);
      assertThat(prunedCache.keyBuffer()).containsExactly(expectedKeys);
      assertThat(prunedCache.valueBuffer()).containsExactly(expectedValues);
      assertThat(pruned.forward(nextToken, tokens.length)).containsExactly(expectedNext);
    }

    @Test
    void finalLayerKvOnlyPrefillPreservesLogitsKvCacheAndAutoregressiveState() {
      assertFinalLayerKvOnlyPrefillIsEquivalent(buildQ4NanoModel(new Random(42)));
      assertFinalLayerKvOnlyPrefillIsEquivalent(buildQ8NanoModel(new Random(42)));
    }

    @Test
    void batchedAttentionValueAccumulationPreservesCompleteAutoregressiveState() {
      assertAttentionBatchingIsEquivalent(false, false, false, true);
    }

    @Test
    void batchedAttentionScoresPreserveCompleteAutoregressiveState() {
      assertAttentionBatchingIsEquivalent(false, true, true, true);
    }

    private void assertAttentionBatchingIsEquivalent(
        boolean baselineScores,
        boolean baselineValues,
        boolean candidateScores,
        boolean candidateValues) {
      GgufFile file = buildQ4NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = new int[40];
      for (int index = 0; index < tokens.length; index++) {
        tokens[index] = (index * 7 + 5) % VOCAB_SIZE;
      }

      for (int batchSize : new int[] {1, 32}) {
        KvCache baselineCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass baseline =
            new LlamaForwardPass(
                config,
                weights,
                baselineCache,
                executionPlan(
                    config, weights, batchSize, true, true, baselineScores, baselineValues));
        float[] expected = baseline.prefill(tokens, 0).clone();
        float[] expectedKeys = baselineCache.keyBuffer().clone();
        float[] expectedValues = baselineCache.valueBuffer().clone();
        int nextToken = argmax(expected);
        float[] expectedNext = baseline.forward(nextToken, tokens.length);

        KvCache batchedCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass batched =
            new LlamaForwardPass(
                config,
                weights,
                batchedCache,
                executionPlan(
                    config, weights, batchSize, true, true, candidateScores, candidateValues));
        float[] actual = batched.prefill(tokens, 0);

        assertThat(batched.usesBatchedAttentionScores()).isEqualTo(candidateScores);
        assertThat(batched.usesBatchedAttentionValues()).isEqualTo(candidateValues);
        assertThat(actual).containsExactly(expected, within(SIMD_REDUCTION_TOLERANCE));
        assertThat(batchedCache.keyBuffer()).containsExactly(expectedKeys);
        assertThat(batchedCache.valueBuffer()).containsExactly(expectedValues);
        assertThat(batched.forward(nextToken, tokens.length))
            .containsExactly(expectedNext, within(SIMD_REDUCTION_TOLERANCE));
      }
    }

    private void assertFinalLayerKvOnlyPrefillIsEquivalent(GgufFile file) {
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = new int[40];
      for (int index = 0; index < tokens.length; index++) {
        tokens[index] = (index * 7 + 5) % VOCAB_SIZE;
      }

      for (int batchSize : new int[] {1, 32}) {
        KvCache baselineCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass baseline =
            new LlamaForwardPass(
                config,
                weights,
                baselineCache,
                executionPlan(config, weights, batchSize, true, false));
        float[] expected = baseline.prefill(tokens, 0).clone();
        float[] expectedKeys = baselineCache.keyBuffer().clone();
        float[] expectedValues = baselineCache.valueBuffer().clone();
        int nextToken = argmax(expected);
        float[] expectedNext = baseline.forward(nextToken, tokens.length);

        KvCache kvOnlyCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass kvOnly =
            new LlamaForwardPass(
                config,
                weights,
                kvOnlyCache,
                executionPlan(config, weights, batchSize, true, true));
        float[] actual = kvOnly.prefill(tokens, 0);

        assertThat(kvOnly.usesFinalLayerKvOnlyPrefill()).isTrue();
        assertThat(actual).containsExactly(expected, within(SIMD_REDUCTION_TOLERANCE));
        assertThat(kvOnlyCache.keyBuffer()).containsExactly(expectedKeys);
        assertThat(kvOnlyCache.valueBuffer()).containsExactly(expectedValues);
        assertThat(kvOnly.forward(nextToken, tokens.length))
            .containsExactly(expectedNext, within(SIMD_REDUCTION_TOLERANCE));
      }
    }

    @Test
    void q8PrefillUsesBatchedKernelAndPreservesAutoregressiveState() {
      GgufFile file = buildQ8NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13, 17, 19, 23, 29};

      LlamaForwardPass sequential =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] expected = runSequence(sequential, tokens);

      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] actual = batched.prefill(tokens, 0);

      assertThat(batched.usesBatchedPrefill()).isTrue();
      assertThat(batched.usesGroupedBatchedPrefill()).isTrue();
      assertThat(actual).containsExactly(expected);
      assertThat(batched.forward(3, tokens.length))
          .containsExactly(sequential.forward(3, tokens.length));
    }

    @Test
    void stagedQ8LayerPreservesPrefillCacheAndAutoregressiveStateExactly() {
      try (Arena arena = Arena.ofShared()) {
        GgufFile file = copyToSharedArena(buildQ8NanoModel(new Random(42)), arena);
        LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
        LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
        int[] tokens = new int[40];
        for (int index = 0; index < tokens.length; index++) {
          tokens[index] = (index * 7 + 5) % VOCAB_SIZE;
        }

        PureJavaPlanConfiguration baselineConfiguration =
            new PureJavaPlanConfiguration(
                true,
                true,
                GgufQ4Kernel.WIDENED,
                GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
                32,
                true,
                true,
                false,
                false,
                false,
                false,
                false,
                GgufQ8BlockMajorKernel.SCATTERED,
                false);
        KvCache baselineCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass baseline =
            new LlamaForwardPass(
                config,
                weights,
                baselineCache,
                ExecutionPlanner.plan(
                    RuntimeFingerprint.capture(),
                    ModelTopology.from("llama", config, weights),
                    baselineConfiguration));
        float[] expected = baseline.prefill(tokens, 0).clone();
        float[] expectedKeys = baselineCache.keyBuffer().clone();
        float[] expectedValues = baselineCache.valueBuffer().clone();
        int nextToken = argmax(expected);
        float[] expectedNext = baseline.forward(nextToken, tokens.length);

        PureJavaPlanConfiguration stagedConfiguration =
            new PureJavaPlanConfiguration(
                true,
                true,
                GgufQ4Kernel.WIDENED,
                GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
                32,
                true,
                true,
                false,
                false,
                true,
                true,
                true,
                GgufQ8BlockMajorKernel.SCATTERED,
                false);
        KvCache stagedCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass staged =
            new LlamaForwardPass(
                config,
                weights,
                stagedCache,
                ExecutionPlanner.plan(
                    RuntimeFingerprint.capture(),
                    ModelTopology.from("llama", config, weights),
                    stagedConfiguration));

        float[] actual = staged.prefill(tokens, 0).clone();
        float[] actualKeys = stagedCache.keyBuffer().clone();
        float[] actualValues = stagedCache.valueBuffer().clone();
        float[] actualNext = staged.forward(nextToken, tokens.length).clone();

        PureJavaPlanConfiguration parallelConfiguration =
            new PureJavaPlanConfiguration(
                true,
                true,
                GgufQ4Kernel.WIDENED,
                GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
                32,
                true,
                true,
                false,
                false,
                true,
                true,
                true,
                GgufQ8BlockMajorKernel.SCATTERED,
                true);
        KvCache parallelCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass parallel =
            new LlamaForwardPass(
                config,
                weights,
                parallelCache,
                ExecutionPlanner.plan(
                    RuntimeFingerprint.capture(),
                    ModelTopology.from("llama", config, weights),
                    parallelConfiguration));
        float[] parallelActual = parallel.prefill(tokens, 0).clone();

        PureJavaPlanConfiguration rowAccumulatedConfiguration =
            new PureJavaPlanConfiguration(
                true,
                true,
                GgufQ4Kernel.WIDENED,
                GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
                32,
                true,
                true,
                false,
                false,
                true,
                true,
                true,
                GgufQ8BlockMajorKernel.ROW_ACCUMULATED,
                true);
        PureJavaExecutionPlan rowAccumulatedPlan =
            ExecutionPlanner.plan(
                graalRuntime(),
                ModelTopology.from("llama", config, weights),
                rowAccumulatedConfiguration);
        KvCache rowAccumulatedCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass rowAccumulated =
            new LlamaForwardPass(config, weights, rowAccumulatedCache, rowAccumulatedPlan);
        float[] rowAccumulatedActual = rowAccumulated.prefill(tokens, 0).clone();

        PureJavaPlanConfiguration floatLaneConfiguration =
            new PureJavaPlanConfiguration(
                true,
                true,
                GgufQ4Kernel.WIDENED,
                GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
                32,
                true,
                true,
                false,
                false,
                true,
                true,
                true,
                GgufQ8BlockMajorKernel.FLOAT_LANE_ACCUMULATED,
                true);
        PureJavaExecutionPlan floatLanePlan =
            ExecutionPlanner.plan(
                graalRuntime256(),
                ModelTopology.from("llama", config, weights),
                floatLaneConfiguration);
        KvCache firstFloatLaneCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass firstFloatLane =
            new LlamaForwardPass(config, weights, firstFloatLaneCache, floatLanePlan);
        float[] firstFloatLaneActual = firstFloatLane.prefill(tokens, 0).clone();
        float[] firstFloatLaneKeys = firstFloatLaneCache.keyBuffer().clone();
        float[] firstFloatLaneValues = firstFloatLaneCache.valueBuffer().clone();
        float[] firstFloatLaneNext = firstFloatLane.forward(nextToken, tokens.length).clone();

        KvCache secondFloatLaneCache =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        LlamaForwardPass secondFloatLane =
            new LlamaForwardPass(config, weights, secondFloatLaneCache, floatLanePlan);
        float[] secondFloatLaneActual = secondFloatLane.prefill(tokens, 0).clone();
        float[] secondFloatLaneKeys = secondFloatLaneCache.keyBuffer().clone();
        float[] secondFloatLaneValues = secondFloatLaneCache.valueBuffer().clone();
        float[] secondFloatLaneNext = secondFloatLane.forward(nextToken, tokens.length).clone();

        assertThat(staged.usesStagedQuantizedLayer()).isTrue();
        assertThat(staged.usesBlockMajorQ8Activations()).isTrue();
        assertThat(staged.stagedQuantizedLayerStageCount()).isEqualTo(7);
        assertThat(actual).containsExactly(expected);
        assertThat(actualKeys).containsExactly(expectedKeys);
        assertThat(actualValues).containsExactly(expectedValues);
        assertThat(actualNext).containsExactly(expectedNext);
        assertThat(parallel.usesParallelQ8FfnPreparation()).isTrue();
        assertThat(parallelActual).containsExactly(actual);
        assertThat(parallelCache.keyBuffer()).containsExactly(actualKeys);
        assertThat(parallelCache.valueBuffer()).containsExactly(actualValues);
        assertThat(parallel.forward(nextToken, tokens.length)).containsExactly(actualNext);
        assertThat(rowAccumulatedPlan.q8BlockMajorKernel())
            .isEqualTo(GgufQ8BlockMajorKernel.ROW_ACCUMULATED);
        assertThat(rowAccumulatedActual).containsExactly(actual);
        assertThat(rowAccumulatedCache.keyBuffer()).containsExactly(actualKeys);
        assertThat(rowAccumulatedCache.valueBuffer()).containsExactly(actualValues);
        assertThat(rowAccumulated.forward(nextToken, tokens.length)).containsExactly(actualNext);
        assertThat(floatLanePlan.q8BlockMajorKernel())
            .isEqualTo(GgufQ8BlockMajorKernel.FLOAT_LANE_ACCUMULATED);
        assertThat(secondFloatLaneActual).containsExactly(firstFloatLaneActual);
        assertThat(secondFloatLaneKeys).containsExactly(firstFloatLaneKeys);
        assertThat(secondFloatLaneValues).containsExactly(firstFloatLaneValues);
        assertThat(secondFloatLaneNext).containsExactly(firstFloatLaneNext);
        assertNumericallyClose(actual, firstFloatLaneActual, 2.0e-3f, 5.0e-5f);
        assertNumericallyClose(actualKeys, firstFloatLaneKeys, 2.0e-3f, 5.0e-5f);
        assertNumericallyClose(actualValues, firstFloatLaneValues, 2.0e-3f, 5.0e-5f);
        assertNumericallyClose(actualNext, firstFloatLaneNext, 2.0e-3f, 5.0e-5f);
        assertThat(argmax(firstFloatLaneActual)).isEqualTo(argmax(actual));
        assertThat(argmax(firstFloatLaneNext)).isEqualTo(argmax(actualNext));
      }
    }

    @Test
    void q5PrefillUsesBatchedKernelAndPreservesAutoregressiveState() {
      GgufFile file = buildQ5NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13, 17, 19, 23, 29};

      LlamaForwardPass sequential =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] expected = runSequence(sequential, tokens);

      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] actual = batched.prefill(tokens, 0);

      assertThat(batched.usesBatchedPrefill()).isTrue();
      assertThat(batched.usesGroupedBatchedPrefill()).isFalse();
      assertThat(actual).containsExactly(expected);
      assertThat(batched.forward(3, tokens.length))
          .containsExactly(sequential.forward(3, tokens.length));
    }

    @Test
    void q4KPrefillUsesBatchedKernelAndPreservesAutoregressiveState() {
      GgufFile file = buildQ4KNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13, 17, 19, 23, 29};

      LlamaForwardPass sequential =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] expected = runSequence(sequential, tokens);

      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] actual = batched.prefill(tokens, 0);

      assertThat(batched.usesBatchedPrefill()).isTrue();
      assertThat(actual).containsExactly(expected);
      assertThat(batched.forward(3, tokens.length))
          .containsExactly(sequential.forward(3, tokens.length));
    }

    @Test
    void mixedQ4KQ6KPrefillUsesBatchedKernelsAndPreservesAutoregressiveState() {
      GgufFile file = buildMixedQ4KQ6KNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13, 17, 19, 23, 29};

      assertThat(weights.layer(0).wqType()).isEqualTo(GgufTensorType.Q4_K);
      assertThat(weights.layer(0).wvType()).isEqualTo(GgufTensorType.Q6_K);
      assertThat(weights.layer(0).woType()).isEqualTo(GgufTensorType.Q6_K);
      assertThat(weights.layer(0).ffnDownType()).isEqualTo(GgufTensorType.Q6_K);

      LlamaForwardPass sequential =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] expected = runSequence(sequential, tokens);

      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()),
              ExecutionPlanner.plan(
                  x86Runtime256(),
                  ModelTopology.from("llama", config, weights),
                  new PureJavaPlanConfiguration(
                      true,
                      true,
                      GgufQ4Kernel.WIDENED,
                      GgufQ6BatchedKernel.TWO_QUERY_BLOCK,
                      32,
                      true,
                      true,
                      false,
                      false,
                      false,
                      false,
                      false,
                      GgufQ8BlockMajorKernel.SCATTERED,
                      false)));
      float[] actual = batched.prefill(tokens, 0);

      assertThat(batched.usesBatchedPrefill()).isTrue();
      assertThat(batched.usesGroupedBatchedPrefill()).isTrue();
      assertThat(batched.usesFinalLayerPrefillPruning()).isFalse();
      assertThat(batched.q6BatchedKernel()).isEqualTo(GgufQ6BatchedKernel.TWO_QUERY_BLOCK);
      assertThat(actual).containsExactly(expected, within(SIMD_REDUCTION_TOLERANCE));
      assertThat(batched.forward(3, tokens.length))
          .containsExactly(sequential.forward(3, tokens.length), within(SIMD_REDUCTION_TOLERANCE));
    }

    @Test
    void mixedQ5KQ6KPrefillUsesBatchedKernelsAndPreservesAutoregressiveState() {
      GgufFile file = buildMixedQ5KQ6KNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = {5, 7, 11, 13, 17, 19, 23, 29};

      assertThat(weights.layer(0).wqType()).isEqualTo(GgufTensorType.Q5_K);
      assertThat(weights.layer(0).wvType()).isEqualTo(GgufTensorType.Q6_K);
      assertThat(weights.layer(0).woType()).isEqualTo(GgufTensorType.Q6_K);
      assertThat(weights.layer(0).ffnDownType()).isEqualTo(GgufTensorType.Q6_K);
      LlamaForwardPass sequential =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] expected = runSequence(sequential, tokens);

      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      float[] actual = batched.prefill(tokens, 0);

      assertThat(batched.usesBatchedPrefill()).isTrue();
      // Sequential and batched prefill are differently shaped float reductions, so bit-exact
      // equality is not something the platform offers here: Vector API lane reductions have
      // implementation-defined ordering, and the C2-intrinsified path can differ from the
      // non-intrinsified one by an ULP. Verified by running this class under
      // -XX:TieredStopAtLevel=1, where the two agree bit-for-bit. The tolerance below is still
      // orders of magnitude tighter than any real kernel defect would produce.
      assertNumericallyClose(expected, actual, 1.0e-5f, 1.0e-6f);
      assertNumericallyClose(
          sequential.forward(3, tokens.length),
          batched.forward(3, tokens.length),
          1.0e-5f,
          1.0e-6f);
    }

    @Test
    void independentSessionBatchPreservesLogitsKvStateAndAutoregressiveContinuation() {
      GgufFile file = buildQ4KNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] firstPrompt = {5, 7, 11};
      int[] secondPrompt = {13, 17, 19, 23};

      KvCache firstExpectedCache =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass firstExpected = new LlamaForwardPass(config, weights, firstExpectedCache);
      firstExpected.prefill(firstPrompt, 0);
      KvCache secondExpectedCache =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass secondExpected = new LlamaForwardPass(config, weights, secondExpectedCache);
      secondExpected.prefill(secondPrompt, 0);

      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      LlamaForwardPass.Session first = batched.openSession();
      LlamaForwardPass.Session second = batched.openSession();
      batched.prefill(first, firstPrompt, 0);
      batched.prefill(second, secondPrompt, 0);

      float[] firstExpectedLogits = firstExpected.forward(29, firstPrompt.length);
      float[] secondExpectedLogits = secondExpected.forward(31, secondPrompt.length);
      LogitBatch actual =
          batched.forwardBatchTransient(
              new LlamaForwardPass.Session[] {first, second}, new int[] {29, 31});

      assertThat(actual.tokenCount()).isEqualTo(2);
      assertThat(actual.copyRow(0))
          .containsExactly(firstExpectedLogits, within(SIMD_REDUCTION_TOLERANCE));
      assertThat(actual.copyRow(1))
          .containsExactly(secondExpectedLogits, within(SIMD_REDUCTION_TOLERANCE));
      assertThat(first.cache().keyBuffer()).containsExactly(firstExpectedCache.keyBuffer());
      assertThat(first.cache().valueBuffer()).containsExactly(firstExpectedCache.valueBuffer());
      assertThat(second.cache().keyBuffer()).containsExactly(secondExpectedCache.keyBuffer());
      assertThat(second.cache().valueBuffer()).containsExactly(secondExpectedCache.valueBuffer());
      assertThat(first.checkpoint()).isEqualTo(firstPrompt.length + 1);
      assertThat(second.checkpoint()).isEqualTo(secondPrompt.length + 1);

      int firstNext = argmax(firstExpectedLogits);
      int secondNext = argmax(secondExpectedLogits);
      float[] firstExpectedNext = firstExpected.forward(firstNext, firstPrompt.length + 1);
      float[] secondExpectedNext = secondExpected.forward(secondNext, secondPrompt.length + 1);
      LogitBatch next =
          batched.forwardBatchTransient(
              new LlamaForwardPass.Session[] {first, second}, new int[] {firstNext, secondNext});

      assertThat(next.copyRow(0))
          .containsExactly(firstExpectedNext, within(SIMD_REDUCTION_TOLERANCE));
      assertThat(next.copyRow(1))
          .containsExactly(secondExpectedNext, within(SIMD_REDUCTION_TOLERANCE));
      assertThat(first.cache().keyBuffer()).containsExactly(firstExpectedCache.keyBuffer());
      assertThat(first.cache().valueBuffer()).containsExactly(firstExpectedCache.valueBuffer());
      assertThat(second.cache().keyBuffer()).containsExactly(secondExpectedCache.keyBuffer());
      assertThat(second.cache().valueBuffer()).containsExactly(secondExpectedCache.valueBuffer());
    }

    @Test
    void fourSessionBatchExercisesRegisterTiledKQuantPathWithinFloatTolerance() {
      GgufFile file = buildQ4KNanoModel(new Random(43));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[][] prompts = {{3, 5}, {7, 11, 13}, {17, 19, 23, 29}, {31}};
      int[] tokens = {2, 4, 6, 8};
      KvCache[] expectedCaches = new KvCache[prompts.length];
      LlamaForwardPass[] expectedPasses = new LlamaForwardPass[prompts.length];

      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      LlamaForwardPass.Session[] sessions = new LlamaForwardPass.Session[prompts.length];
      float[][] expectedLogits = new float[prompts.length][];
      for (int index = 0; index < prompts.length; index++) {
        expectedCaches[index] =
            new KvCache(
                config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
        expectedPasses[index] = new LlamaForwardPass(config, weights, expectedCaches[index]);
        expectedPasses[index].prefill(prompts[index], 0);
        expectedLogits[index] = expectedPasses[index].forward(tokens[index], prompts[index].length);

        sessions[index] = batched.openSession();
        batched.prefill(sessions[index], prompts[index], 0);
      }

      LogitBatch actual = batched.forwardBatchTransient(sessions, tokens);

      assertThat(batched.usesGroupedBatchedPrefill()).isTrue();
      for (int index = 0; index < sessions.length; index++) {
        assertThat(actual.copyRow(index))
            .containsExactly(expectedLogits[index], within(SIMD_REDUCTION_TOLERANCE));
        assertThat(sessions[index].cache().keyBuffer())
            .containsExactly(expectedCaches[index].keyBuffer());
        assertThat(sessions[index].cache().valueBuffer())
            .containsExactly(expectedCaches[index].valueBuffer());
      }
    }

    @Test
    void independentSessionBatchReusesQuantizedOutputWeightsExactly() {
      assertIndependentSessionBatchOutputExact(
          buildMixedQ4KQ6KQuantizedOutputNanoModel(new Random(44)), GgufTensorType.Q6_K);
    }

    @Test
    void independentSessionBatchSizesScratchForQ4QuantizedOutput() {
      assertIndependentSessionBatchOutputExact(
          buildQ4QuantizedOutputNanoModel(new Random(45)), GgufTensorType.Q4_0);
    }

    private void assertIndependentSessionBatchOutputExact(
        GgufFile file, GgufTensorType expectedOutputType) {
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[][] prompts = {{3, 5}, {7, 11, 13}, {17, 19, 23, 29}, {31}};
      int[] tokens = {2, 4, 6, 8};
      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()),
              executionPlan(config, weights, 32, false));
      LlamaForwardPass.Session[] sessions = new LlamaForwardPass.Session[prompts.length];
      float[][] expected = new float[prompts.length][];
      for (int index = 0; index < prompts.length; index++) {
        LlamaForwardPass sequential =
            new LlamaForwardPass(
                config,
                weights,
                new KvCache(
                    config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()),
                executionPlan(config, weights, 32, false));
        sequential.prefill(prompts[index], 0);
        expected[index] = sequential.forward(tokens[index], prompts[index].length);
        sessions[index] = batched.openSession();
        batched.prefill(sessions[index], prompts[index], 0);
      }

      LogitBatch actual = batched.forwardBatchTransient(sessions, tokens);

      assertThat(weights.outputType()).isEqualTo(expectedOutputType);
      assertThat(batched.usesBatchedSessionOutputProjection()).isTrue();
      for (int index = 0; index < sessions.length; index++) {
        assertThat(actual.copyRow(index)).containsExactly(expected[index]);
      }
    }

    @Test
    void sessionBatchRejectsInvalidOwnershipShapeAndAliasing() {
      GgufFile file = buildQ4KNanoModel(new Random(44));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaForwardPass first =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      LlamaForwardPass second =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      LlamaForwardPass.Session owned = first.openSession();
      LlamaForwardPass.Session foreign = second.openSession();

      assertThatThrownBy(
              () -> first.forwardBatchTransient(new LlamaForwardPass.Session[0], new int[0]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessage("sessions must not be empty");
      assertThatThrownBy(
              () ->
                  first.forwardBatchTransient(
                      new LlamaForwardPass.Session[] {owned}, new int[] {1, 2}))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("tokens.length must equal sessions.length");
      assertThatThrownBy(
              () ->
                  first.forwardBatchTransient(
                      new LlamaForwardPass.Session[] {owned, owned}, new int[] {1, 2}))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessage("sessions must be distinct");
      assertThatThrownBy(
              () ->
                  first.forwardBatchTransient(
                      new LlamaForwardPass.Session[] {owned, foreign}, new int[] {1, 2}))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessage("session belongs to a different forward pass");
    }

    @Test
    void rewindAndResetAffectOnlyTheirSession() {
      GgufFile file = buildNanoModel(new Random(45));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaForwardPass batched =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      LlamaForwardPass.Session first = batched.openSession();
      LlamaForwardPass.Session second = batched.openSession();
      batched.prefill(first, new int[] {3, 5, 7}, 0);
      batched.prefill(second, new int[] {11, 13, 17}, 0);
      float[] secondKeys = second.cache().keyBuffer().clone();
      float[] secondValues = second.cache().valueBuffer().clone();

      batched.rewind(first, 2);
      assertThat(first.checkpoint()).isEqualTo(2);
      assertThat(second.checkpoint()).isEqualTo(3);
      assertThat(second.cache().keyBuffer()).containsExactly(secondKeys);
      assertThat(second.cache().valueBuffer()).containsExactly(secondValues);

      batched.reset(first);
      assertThat(first.checkpoint()).isZero();
      assertThat(second.checkpoint()).isEqualTo(3);
      assertThat(second.cache().keyBuffer()).containsExactly(secondKeys);
      assertThat(second.cache().valueBuffer()).containsExactly(secondValues);
    }

    @Test
    void q4VerificationReturnsEverySequentialLogitRow() {
      GgufFile file = buildQ4NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] prefix = {5, 7};
      int[] proposed = {11, 13, 17, 19};

      LlamaForwardPass sequential =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      sequential.prefill(prefix, 0);
      float[][] expected = new float[proposed.length][];
      for (int index = 0; index < proposed.length; index++) {
        expected[index] = sequential.forward(proposed[index], prefix.length + index);
      }

      LlamaForwardPass verifying =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      verifying.prefill(prefix, 0);
      LogitBatch actual = verifying.verify(proposed, prefix.length);

      assertThat(verifying.usesBatchedPrefill()).isTrue();
      assertThat(actual.tokenCount()).isEqualTo(proposed.length);
      for (int index = 0; index < proposed.length; index++) {
        assertThat(actual.copyRow(index)).containsExactly(expected[index]);
      }
    }

    @Test
    void rewindPreservesAcceptedPrefixAndReplacesRejectedTail() {
      GgufFile file = buildQ4NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] prefix = {5, 7};

      LlamaForwardPass expected =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      expected.prefill(prefix, 0);
      expected.forward(11, 2);
      float[] expectedReplacement = expected.forward(23, 3);

      LlamaForwardPass actual =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      actual.prefill(prefix, 0);
      int checkpoint = actual.checkpoint();
      actual.verify(new int[] {11, 13, 17}, checkpoint);
      actual.rewind(checkpoint + 1);
      float[] actualReplacement = actual.forward(23, checkpoint + 1);

      assertThat(checkpoint).isEqualTo(prefix.length);
      assertThat(actualReplacement)
          .containsExactly(expectedReplacement, within(SIMD_REDUCTION_TOLERANCE));
    }

    @Test
    void stableVerificationSnapshotSurvivesLaterTransientVerification() {
      GgufFile file = buildQ4NanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaForwardPass forwardPass =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
      forwardPass.prefill(new int[] {5, 7}, 0);

      LogitBatch stable = forwardPass.verify(new int[] {11, 13}, 2);
      float[] expectedStableRow = stable.copyRow(0);
      forwardPass.rewind(2);
      forwardPass.verifyTransient(new int[] {17, 19}, 2);

      assertThat(stable.copyRow(0)).containsExactly(expectedStableRow);
    }

    @Test
    void transientForwardReusesLogitStorageWithoutChangingForwardSnapshots() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaForwardPass forwardPass =
          new LlamaForwardPass(
              config,
              weights,
              new KvCache(
                  config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));

      float[] snapshot = forwardPass.forward(5, 0);
      float[] snapshotValues = snapshot.clone();
      float[] firstTransient = forwardPass.forwardTransient(7, 1);
      float[] firstTransientValues = firstTransient.clone();
      float[] secondTransient = forwardPass.forwardTransient(11, 2);

      assertThat(secondTransient).isSameAs(firstTransient);
      assertThat(secondTransient).isNotSameAs(snapshot);
      assertThat(firstTransient).isNotEqualTo(firstTransientValues);
      assertThat(snapshot).containsExactly(snapshotValues);
    }

    @Test
    void differentPositionsProduceDifferentLogits() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      KvCache cache =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);

      float[] logits0 = forwardPass.forward(5, 0);
      float[] logits1 = forwardPass.forward(5, 1);

      assertThat(logits0).isNotEqualTo(logits1);
    }

    @Test
    void rejectsNonSequentialPositions() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      KvCache cache =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);

      assertThatThrownBy(() -> forwardPass.forward(5, 1))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("expected 0");
    }

    @Test
    void cacheGrowthAndResetPreserveSequenceResults() {
      GgufFile file = buildNanoModel(new Random(42));
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      KvCache cache =
          new KvCache(
              config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);
      int[] tokens = new int[20];
      for (int i = 0; i < tokens.length; i++) {
        tokens[i] = i % VOCAB_SIZE;
      }

      float[] first = runSequence(forwardPass, tokens);
      assertThat(cache.allocatedSequenceCapacity()).isEqualTo(32);

      forwardPass.reset();
      float[] second = runSequence(forwardPass, tokens);

      assertThat(second).containsExactly(first);
    }
  }

  private static float[] runSequence(LlamaForwardPass forwardPass, int[] tokens) {
    float[] logits = null;
    for (int position = 0; position < tokens.length; position++) {
      logits = forwardPass.forward(tokens[position], position);
    }
    return logits;
  }

  private static void referenceBatchedAttention(
      float[] output,
      float[] query,
      float[] key,
      float[] value,
      float[] keyCache,
      float[] valueCache,
      int startPosition,
      int batchSize,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int slidingWindow) {
    int queryDim = numHeads * keyLength;
    int keyDim = numKvHeads * keyLength;
    int valueDim = numKvHeads * valueLength;
    int outputDim = numHeads * valueLength;
    for (int batch = 0; batch < batchSize; batch++) {
      System.arraycopy(key, batch * keyDim, keyCache, (startPosition + batch) * keyDim, keyDim);
      System.arraycopy(
          value, batch * valueDim, valueCache, (startPosition + batch) * valueDim, valueDim);
    }
    int groupSize = numHeads / numKvHeads;
    float scale = (float) (1.0 / Math.sqrt(keyLength));
    float[] scores = new float[startPosition + batchSize];
    for (int batch = 0; batch < batchSize; batch++) {
      int position = startPosition + batch;
      int firstPosition = slidingWindow > 0 ? Math.max(0, position - slidingWindow + 1) : 0;
      int outputOffset = batch * outputDim;
      java.util.Arrays.fill(output, outputOffset, outputOffset + outputDim, 0.0f);
      for (int head = 0; head < numHeads; head++) {
        int kvHead = head / groupSize;
        int queryOffset = batch * queryDim + head * keyLength;
        for (int cached = firstPosition; cached <= position; cached++) {
          int keyOffset = cached * keyDim + kvHead * keyLength;
          scores[cached] =
              com.integrallis.vectors.core.VectorUtil.dotProduct(
                      query, queryOffset, keyCache, keyOffset, keyLength)
                  * scale;
        }
        TensorOps.softmax(scores, firstPosition, position - firstPosition + 1);
        int headOutputOffset = outputOffset + head * valueLength;
        for (int cached = firstPosition; cached <= position; cached++) {
          int valueOffset = cached * valueDim + kvHead * valueLength;
          com.integrallis.vectors.core.VectorUtil.addScaledInPlace(
              output, headOutputOffset, valueCache, valueOffset, valueLength, scores[cached]);
        }
      }
    }
  }

  private static void multiplyQ4(
      float[] output, float[] input, MemorySegment weights, int batchSize, int rows, int cols) {
    TensorOps.ggufBatchedMatmul(
        output,
        input,
        weights,
        GgufTensorType.Q4_0,
        batchSize,
        rows,
        cols,
        new byte[batchSize * cols],
        new float[batchSize * cols / 32],
        new int[batchSize * cols / 4],
        new short[batchSize * cols / 16],
        new float[batchSize * rows * 8],
        GgufQ4Kernel.WIDENED);
  }

  private static PureJavaExecutionPlan executionPlan(
      LlamaConfig config,
      LlamaWeights weights,
      int prefillBatchSize,
      boolean finalLayerPrefillPruning) {
    return executionPlan(config, weights, prefillBatchSize, finalLayerPrefillPruning, false);
  }

  private static PureJavaExecutionPlan executionPlan(
      LlamaConfig config,
      LlamaWeights weights,
      int prefillBatchSize,
      boolean finalLayerPrefillPruning,
      boolean finalLayerKvOnlyPrefill) {
    return executionPlan(
        config,
        weights,
        prefillBatchSize,
        finalLayerPrefillPruning,
        finalLayerKvOnlyPrefill,
        false,
        false);
  }

  private static PureJavaExecutionPlan executionPlan(
      LlamaConfig config,
      LlamaWeights weights,
      int prefillBatchSize,
      boolean finalLayerPrefillPruning,
      boolean finalLayerKvOnlyPrefill,
      boolean batchedAttentionValues) {
    return executionPlan(
        config,
        weights,
        prefillBatchSize,
        finalLayerPrefillPruning,
        finalLayerKvOnlyPrefill,
        false,
        batchedAttentionValues);
  }

  private static PureJavaExecutionPlan executionPlan(
      LlamaConfig config,
      LlamaWeights weights,
      int prefillBatchSize,
      boolean finalLayerPrefillPruning,
      boolean finalLayerKvOnlyPrefill,
      boolean batchedAttentionScores,
      boolean batchedAttentionValues) {
    return ExecutionPlanner.plan(
        RuntimeFingerprint.capture(),
        ModelTopology.from("llama", config, weights),
        new PureJavaPlanConfiguration(
            true,
            true,
            GgufQ4Kernel.WIDENED,
            GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
            prefillBatchSize,
            finalLayerPrefillPruning,
            finalLayerKvOnlyPrefill,
            batchedAttentionScores,
            batchedAttentionValues,
            false,
            false,
            false,
            GgufQ8BlockMajorKernel.SCATTERED,
            false));
  }

  private static RuntimeFingerprint graalRuntime() {
    RuntimeFingerprint runtime = RuntimeFingerprint.capture();
    return new RuntimeFingerprint(
        runtime.javaVersion(),
        runtime.vmName(),
        runtime.vmVendor(),
        runtime.vmVersion(),
        "graal-jvmci",
        runtime.osName(),
        runtime.architecture(),
        runtime.cpuModel(),
        runtime.vectorProvider(),
        runtime.vectorApi(),
        runtime.preferredVectorBits(),
        runtime.vectorBits(),
        runtime.fastVectorFma(),
        runtime.fastScalarFma(),
        runtime.sve(),
        runtime.q4ShortPairwiseSupported(),
        runtime.q4UnsignedPairwiseSupported(),
        runtime.ggufParallel(),
        runtime.ggufExecutor(),
        runtime.ggufThreads(),
        runtime.ggufChunksPerThread(),
        runtime.processors());
  }

  private static RuntimeFingerprint graalRuntime256() {
    RuntimeFingerprint runtime = graalRuntime();
    return new RuntimeFingerprint(
        runtime.javaVersion(),
        runtime.vmName(),
        runtime.vmVendor(),
        runtime.vmVersion(),
        runtime.compiler(),
        runtime.osName(),
        runtime.architecture(),
        runtime.cpuModel(),
        runtime.vectorProvider(),
        true,
        Math.max(256, runtime.preferredVectorBits()),
        256,
        runtime.fastVectorFma(),
        runtime.fastScalarFma(),
        runtime.sve(),
        true,
        true,
        runtime.ggufParallel(),
        runtime.ggufExecutor(),
        runtime.ggufThreads(),
        runtime.ggufChunksPerThread(),
        runtime.processors());
  }

  private static RuntimeFingerprint x86Runtime256() {
    RuntimeFingerprint runtime = RuntimeFingerprint.capture();
    return new RuntimeFingerprint(
        runtime.javaVersion(),
        runtime.vmName(),
        runtime.vmVendor(),
        runtime.vmVersion(),
        runtime.compiler(),
        runtime.osName(),
        "amd64",
        "test-x86-256",
        runtime.vectorProvider(),
        true,
        Math.max(256, runtime.preferredVectorBits()),
        256,
        runtime.fastVectorFma(),
        runtime.fastScalarFma(),
        false,
        true,
        true,
        runtime.ggufParallel(),
        runtime.ggufExecutor(),
        runtime.ggufThreads(),
        runtime.ggufChunksPerThread(),
        runtime.processors());
  }

  private static void assertNumericallyClose(
      float[] expected, float[] actual, float absoluteTolerance, float relativeTolerance) {
    assertThat(actual).hasSameSizeAs(expected);
    for (int index = 0; index < expected.length; index++) {
      float allowedError =
          Math.max(absoluteTolerance, Math.abs(expected[index]) * relativeTolerance);
      assertThat(Math.abs(actual[index] - expected[index]))
          .as("index=%s, expected=%s, actual=%s", index, expected[index], actual[index])
          .isLessThanOrEqualTo(allowedError);
    }
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
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

  private static byte[] randomQ4(Random rng, int valueCount) {
    if (valueCount % 32 != 0) {
      throw new IllegalArgumentException("Q4_0 value count must be a multiple of 32");
    }
    byte[] data = new byte[(valueCount / 32) * 18];
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < valueCount / 32; block++) {
      int offset = block * 18;
      buffer.putShort(offset, Float.floatToFloat16(0.01f));
      for (int packed = 0; packed < 16; packed++) {
        data[offset + 2 + packed] = (byte) rng.nextInt(256);
      }
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

  private static byte[] randomQ5(Random rng, int valueCount) {
    if (valueCount % 32 != 0) {
      throw new IllegalArgumentException("Q5_0 value count must be a multiple of 32");
    }
    byte[] data = new byte[(valueCount / 32) * 22];
    rng.nextBytes(data);
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < valueCount / 32; block++) {
      buffer.putShort(block * 22, Float.floatToFloat16(0.01f));
    }
    return data;
  }

  private static byte[] randomQ4K(Random rng, int valueCount) {
    if (valueCount % 256 != 0) {
      throw new IllegalArgumentException("Q4_K value count must be a multiple of 256");
    }
    byte[] data = new byte[(valueCount / 256) * 144];
    rng.nextBytes(data);
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < valueCount / 256; block++) {
      int offset = block * 144;
      buffer.putShort(offset, Float.floatToFloat16(0.01f));
      buffer.putShort(offset + Short.BYTES, Float.floatToFloat16(0.005f));
    }
    return data;
  }

  private static byte[] randomQ5K(Random rng, int valueCount) {
    if (valueCount % 256 != 0) {
      throw new IllegalArgumentException("Q5_K value count must be a multiple of 256");
    }
    byte[] data = new byte[(valueCount / 256) * 176];
    rng.nextBytes(data);
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < valueCount / 256; block++) {
      int offset = block * 176;
      buffer.putShort(offset, Float.floatToFloat16(0.01f));
      buffer.putShort(offset + Short.BYTES, Float.floatToFloat16(0.005f));
    }
    return data;
  }

  private static byte[] randomQ6K(Random rng, int valueCount) {
    if (valueCount % 256 != 0) {
      throw new IllegalArgumentException("Q6_K value count must be a multiple of 256");
    }
    byte[] data = new byte[(valueCount / 256) * 210];
    rng.nextBytes(data);
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < valueCount / 256; block++) {
      buffer.putShort(block * 210 + 208, Float.floatToFloat16(0.01f));
    }
    return data;
  }
}
