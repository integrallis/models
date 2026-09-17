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

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliff;
import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliffRecording;
import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliffs;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import com.integrallis.models.backend.purejava.plan.ExecutionPlanner;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Attention fallbacks chosen when a Llama-family forward pass is constructed. */
@Tag("unit")
class LlamaForwardPassPerformanceCliffTest {

  private static final int DIM = 16;
  private static final int HIDDEN_DIM = 32;
  private static final int VOCAB_SIZE = 32;
  private static final int CONTEXT = 32;

  /** An injected kernel with no native grouped attention, standing in for an older library. */
  private static final GgufBatchedMatrixKernel KERNEL_WITHOUT_GROUPED_ATTENTION =
      new GgufBatchedMatrixKernel() {
        @Override
        public String implementation() {
          return "test-kernel-without-grouped-attention";
        }

        @Override
        public boolean supports(GgufTensorType type) {
          return false;
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
          throw new AssertionError("no tensor type is supported");
        }
      };

  @Test
  void groupedQueryLlamaReportsFusedGroupedAttentionNotWiredOnceAcrossPasses() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      for (int pass = 0; pass < 3; pass++) {
        LlamaForwardPass forwardPass = forwardPass("llama", 2, 1, GgufBatchedMatrixKernel.none());
        forwardPass.forward(1, 0);
        forwardPass.forward(2, 1);
      }

      assertThat(recording.count(PerformanceCliff.FUSED_GROUPED_ATTENTION_NOT_WIRED)).isEqualTo(1);
      assertThat(
              PerformanceCliffs.reported().get(PerformanceCliff.FUSED_GROUPED_ATTENTION_NOT_WIRED))
          .contains("architecture=llama")
          .contains("group-size=2");
      assertThat(recording.count(PerformanceCliff.NATIVE_GROUPED_ATTENTION_UNAVAILABLE)).isZero();
    }
  }

  @Test
  void multiHeadLlamaHasNoGroupsAndReportsNoAttentionCliff() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      forwardPass("llama", 2, 2, GgufBatchedMatrixKernel.none()).forward(1, 0);

      assertThat(recording.count(PerformanceCliff.FUSED_GROUPED_ATTENTION_NOT_WIRED)).isZero();
    }
  }

  @Test
  void graniteTakesTheFusedPathAndThePureJavaBackendReportsNoNativeCliff() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      forwardPass("granite", 2, 1, GgufBatchedMatrixKernel.none()).forward(1, 0);

      assertThat(recording.count(PerformanceCliff.FUSED_GROUPED_ATTENTION_NOT_WIRED)).isZero();
      assertThat(recording.count(PerformanceCliff.NATIVE_GROUPED_ATTENTION_UNAVAILABLE)).isZero();
    }
  }

  @Test
  void graniteWithAnInjectedKernelLackingGroupedAttentionReportsOnce() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      for (int pass = 0; pass < 3; pass++) {
        forwardPass("granite", 2, 1, KERNEL_WITHOUT_GROUPED_ATTENTION).forward(1, 0);
      }

      assertThat(recording.count(PerformanceCliff.NATIVE_GROUPED_ATTENTION_UNAVAILABLE))
          .isEqualTo(1);
      assertThat(
              PerformanceCliffs.reported()
                  .get(PerformanceCliff.NATIVE_GROUPED_ATTENTION_UNAVAILABLE))
          .contains("kernel=test-kernel-without-grouped-attention");
    }
  }

  /** A kernel that claims native grouped attention and counts calls; output is left at zero. */
  private static final class CountingGroupedAttentionKernel implements GgufBatchedMatrixKernel {
    private final java.util.concurrent.atomic.AtomicInteger calls =
        new java.util.concurrent.atomic.AtomicInteger();

    @Override
    public String implementation() {
      return "test-kernel-with-grouped-attention";
    }

    @Override
    public boolean supports(GgufTensorType type) {
      return false;
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
      throw new AssertionError("no tensor type is supported");
    }

    @Override
    public boolean supportsGroupedAttention() {
      return true;
    }

    @Override
    public void groupedAttention(
        float[] query,
        int queryOffset,
        float[] keysA,
        int keysAOffset,
        float[] valuesA,
        int valuesAOffset,
        int positionsA,
        float[] keysB,
        int keysBOffset,
        float[] valuesB,
        int valuesBOffset,
        int positionsB,
        float[] output,
        int outputOffset,
        float[] scores,
        int keyDim,
        int valueDim,
        int keyLength,
        int valueLength,
        int numHeads,
        int numKvHeads,
        float scale) {
      calls.incrementAndGet();
    }
  }

  @Test
  void graniteNativeAttentionOverTwoCacheSpansReportsNoSpanCliff() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      CountingGroupedAttentionKernel kernel = new CountingGroupedAttentionKernel();
      LlamaForwardPass forwardPass = forwardPass("granite", 2, 1, kernel);
      LlamaForwardPass.Session source = forwardPass.openSession();
      forwardPass.forward(source, 1, 0);
      forwardPass.forward(source, 2, 1);
      LlamaForwardPass.Session branch = forwardPass.freezePrefix(source).fork();
      int before = kernel.calls.get();
      forwardPass.forward(branch, 3, 2);

      assertThat(kernel.calls.get()).isEqualTo(before + 1);
      assertThat(recording.count(PerformanceCliff.NATIVE_GROUPED_ATTENTION_SPAN_LIMIT)).isZero();
    }
  }

  @Test
  void graniteNativeAttentionOverMoreThanTwoCacheSpansFallsBackAndReportsOnce() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      CountingGroupedAttentionKernel kernel = new CountingGroupedAttentionKernel();
      LlamaForwardPass forwardPass = forwardPass("granite", 2, 1, kernel);
      LlamaForwardPass.Session source = forwardPass.openSession();
      forwardPass.forward(source, 1, 0);
      LlamaForwardPass.Session middle = forwardPass.freezePrefix(source).fork();
      forwardPass.forward(middle, 2, 1);
      LlamaForwardPass.Session leaf = forwardPass.freezePrefix(middle).fork();
      int before = kernel.calls.get();
      forwardPass.forward(leaf, 3, 2);
      forwardPass.forward(leaf, 4, 3);

      assertThat(kernel.calls.get())
          .as("three spans never reach the native kernel")
          .isEqualTo(before);
      assertThat(recording.count(PerformanceCliff.NATIVE_GROUPED_ATTENTION_SPAN_LIMIT))
          .isEqualTo(1);
      assertThat(
              PerformanceCliffs.reported()
                  .get(PerformanceCliff.NATIVE_GROUPED_ATTENTION_SPAN_LIMIT))
          .contains("architecture=granite")
          .contains("spans=3")
          .contains("kernel=test-kernel-with-grouped-attention");
    }
  }

  @Test
  void projectionTypeWithoutBatchedKernelReportsBatchedPrefillCliffOnce() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      for (int pass = 0; pass < 3; pass++) {
        forwardPass("llama", 2, 2, GgufBatchedMatrixKernel.none());
      }

      assertThat(recording.count(PerformanceCliff.BATCHED_PREFILL_UNSUPPORTED_TENSOR_TYPE))
          .isEqualTo(1);
      assertThat(
              PerformanceCliffs.reported()
                  .get(PerformanceCliff.BATCHED_PREFILL_UNSUPPORTED_TENSOR_TYPE))
          .contains("architecture=llama")
          .contains("F32");
    }
  }

  @Test
  void batchedProjectionTypesReportNoBatchedPrefillCliff() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      LlamaForwardPass forwardPass =
          forwardPass("llama", 2, 2, GgufTensorType.F16, GgufBatchedMatrixKernel.none());
      forwardPass.prefill(new int[] {1, 2, 3, 4}, 0);

      assertThat(recording.count(PerformanceCliff.BATCHED_PREFILL_UNSUPPORTED_TENSOR_TYPE))
          .isZero();
    }
  }

  private static LlamaForwardPass forwardPass(
      String architecture, int heads, int kvHeads, GgufBatchedMatrixKernel kernel) {
    return forwardPass(architecture, heads, kvHeads, GgufTensorType.F32, kernel);
  }

  private static LlamaForwardPass forwardPass(
      String architecture,
      int heads,
      int kvHeads,
      GgufTensorType projectionType,
      GgufBatchedMatrixKernel kernel) {
    GgufFile file = buildModel(architecture, heads, kvHeads, projectionType, new Random(7));
    LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
    LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
    KvCache cache =
        new KvCache(config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim());
    return new LlamaForwardPass(
        config,
        weights,
        cache,
        ExecutionPlanner.plan(
            RuntimeFingerprint.capture(),
            ModelTopology.from(config.architecture().metadataId(), config, weights),
            PureJavaPlanConfiguration.defaults(),
            kernel),
        kernel);
  }

  private static GgufFile buildModel(
      String architecture, int heads, int kvHeads, GgufTensorType projectionType, Random random) {
    int kvDim = kvHeads * (DIM / heads);
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", architecture)
            .addUint32(architecture + ".embedding_length", DIM)
            .addUint32(architecture + ".block_count", 1)
            .addUint32(architecture + ".attention.head_count", heads)
            .addUint32(architecture + ".attention.head_count_kv", kvHeads)
            .addUint32(architecture + ".vocab_size", VOCAB_SIZE)
            .addUint32(architecture + ".context_length", CONTEXT)
            .addUint32(architecture + ".feed_forward_length", HIDDEN_DIM);
    if (architecture.equals("granite")) {
      builder
          .addFloat32("granite.embedding_scale", 12.0f)
          .addFloat32("granite.attention.scale", 0.25f)
          .addFloat32("granite.residual_scale", 0.22f)
          .addFloat32("granite.logit_scale", 16.0f);
    }
    builder
        .addTensor(
            "token_embd.weight",
            GgufTensorType.F32,
            new long[] {DIM, VOCAB_SIZE},
            random(random, VOCAB_SIZE * DIM))
        .addTensor("output_norm.weight", GgufTensorType.F32, new long[] {DIM}, ones(DIM))
        .addTensor(
            "output.weight",
            GgufTensorType.F32,
            new long[] {DIM, VOCAB_SIZE},
            random(random, VOCAB_SIZE * DIM))
        .addTensor("blk.0.attn_norm.weight", GgufTensorType.F32, new long[] {DIM}, ones(DIM))
        .addTensor(
            "blk.0.attn_q.weight",
            projectionType,
            new long[] {DIM, DIM},
            projection(random, projectionType, DIM * DIM))
        .addTensor(
            "blk.0.attn_k.weight",
            projectionType,
            new long[] {DIM, kvDim},
            projection(random, projectionType, kvDim * DIM))
        .addTensor(
            "blk.0.attn_v.weight",
            projectionType,
            new long[] {DIM, kvDim},
            projection(random, projectionType, kvDim * DIM))
        .addTensor(
            "blk.0.attn_output.weight",
            projectionType,
            new long[] {DIM, DIM},
            projection(random, projectionType, DIM * DIM))
        .addTensor("blk.0.ffn_norm.weight", GgufTensorType.F32, new long[] {DIM}, ones(DIM))
        .addTensor(
            "blk.0.ffn_gate.weight",
            projectionType,
            new long[] {DIM, HIDDEN_DIM},
            projection(random, projectionType, HIDDEN_DIM * DIM))
        .addTensor(
            "blk.0.ffn_up.weight",
            projectionType,
            new long[] {DIM, HIDDEN_DIM},
            projection(random, projectionType, HIDDEN_DIM * DIM))
        .addTensor(
            "blk.0.ffn_down.weight",
            projectionType,
            new long[] {HIDDEN_DIM, DIM},
            projection(random, projectionType, DIM * HIDDEN_DIM));
    byte[] data = builder.build();
    MemorySegment segment = Arena.ofAuto().allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return GgufParser.parseSegment(segment);
  }

  private static byte[] random(Random random, int count) {
    ByteBuffer buffer = ByteBuffer.allocate(count * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < count; index++) {
      buffer.putFloat((random.nextFloat() - 0.5f) * 0.2f);
    }
    return buffer.array();
  }

  private static byte[] projection(Random random, GgufTensorType type, int count) {
    if (type == GgufTensorType.F32) {
      return random(random, count);
    }
    if (type != GgufTensorType.F16) {
      throw new IllegalArgumentException("test builder writes F32 or F16 projections: " + type);
    }
    ByteBuffer buffer = ByteBuffer.allocate(count * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < count; index++) {
      buffer.putShort(Float.floatToFloat16((random.nextFloat() - 0.5f) * 0.2f));
    }
    return buffer.array();
  }

  private static byte[] ones(int count) {
    ByteBuffer buffer = ByteBuffer.allocate(count * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < count; index++) {
      buffer.putFloat(1.0f);
    }
    return buffer.array();
  }
}
