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
import static org.assertj.core.api.Assertions.within;
import static org.assertj.core.data.Offset.offset;

import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufHeader;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufTensorInfo;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.io.ByteArrayOutputStream;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Gemma4ForwardPassTest {

  @Test
  void batchedExpertOrderLeavesTheMostRecentPromptRoutesLast() {
    int[] selectedExperts = {2, 0, 1, 2, 3, 1};
    int[] order = new int[4];

    int count = Gemma4ForwardPass.orderExpertsByLastRoute(selectedExperts, 3, 2, 4, order);

    assertThat(count).isEqualTo(4);
    assertThat(Arrays.copyOf(order, count)).containsExactly(0, 2, 1, 3);
  }

  @Test
  void completeToyGraphMatchesAnIndependentScalarReference() throws Exception {
    ToyModel model = ToyModel.create();
    Gemma4Weights weights = Gemma4Weights.fromGgufFile(model.file(), model.config());
    Gemma4TensorLayout layout = weights.expertLayout();
    Gemma4ExpertLoader.PositionalReader reader = reader(model.file().fileSegment());

    try (Gemma4ExpertCache experts =
        new Gemma4ExpertCache(
            new Gemma4ExpertLoader(reader),
            model.config().numLayers(),
            model.config().numExperts(),
            1,
            (layer, expert) -> layout.layer(layer).expert(expert),
            Gemma4ExpertCache.CachePolicy.LFU)) {
      LayeredKvCache cache = Gemma4KvCache.create(model.config(), 8, 2);
      Gemma4ForwardPass actual = new Gemma4ForwardPass(model.config(), weights, cache, experts);
      ScalarReference expected = new ScalarReference(model);

      assertClose(actual.forward(0, 0), expected.forward(0));
      assertClose(actual.forward(1, 1), expected.forward(1));

      actual.reset();
      expected.reset();
      assertClose(actual.prefill(new int[] {0, 1}, 0), expected.prefill(0, 1));
      assertThat(actual.checkpoint()).isEqualTo(2);
      assertThat(experts.stats().misses()).isGreaterThan(0);
    }
  }

  /**
   * A per-layer feed-forward width, checked against the independent scalar reference.
   *
   * <p>Gemma 4 E2B was refused at load -- "blk.15.ffn_gate.weight shape must be [1536, 6144], found
   * [1536, 12288]" -- because the width was read once from layer 0 and used for every layer. The
   * config test for the array form did not catch it: every element of its array is the same number,
   * so a broadcast scalar and a genuine per-layer list are indistinguishable there.
   *
   * <p>Loading is what failed, but agreeing with the reference is what makes the fix correct rather
   * than merely permissive: the buffers now serve the widest layer, so a stride still taken from
   * layer 0's width would read the wrong slice instead of overflowing.
   */
  @Test
  void aFeedForwardWidthThatVariesPerLayerMatchesTheScalarReference() throws Exception {
    ToyModel model = ToyModel.createVaryingSharedFeedForward();

    assertThat(model.config().sharedHiddenDim(0)).isEqualTo(3);
    assertThat(model.config().sharedHiddenDim(1))
        .describedAs("the fixture must genuinely vary, or this tests the uniform case again")
        .isEqualTo(5)
        .isNotEqualTo(model.config().sharedHiddenDim(0));
    assertThat(model.config().maxSharedHiddenDim()).isEqualTo(5);

    Gemma4Weights weights = Gemma4Weights.fromGgufFile(model.file(), model.config());
    Gemma4TensorLayout layout = weights.expertLayout();
    Gemma4ExpertLoader.PositionalReader reader = reader(model.file().fileSegment());

    try (Gemma4ExpertCache experts =
        new Gemma4ExpertCache(
            new Gemma4ExpertLoader(reader),
            model.config().numLayers(),
            model.config().numExperts(),
            1,
            (layer, expert) -> layout.layer(layer).expert(expert),
            Gemma4ExpertCache.CachePolicy.LFU)) {
      LayeredKvCache cache = Gemma4KvCache.create(model.config(), 8, 2);
      Gemma4ForwardPass actual = new Gemma4ForwardPass(model.config(), weights, cache, experts);
      ScalarReference expected = new ScalarReference(model);

      assertClose(actual.forward(0, 0), expected.forward(0));
      assertClose(actual.forward(1, 1), expected.forward(1));

      // And the batched path, which strides its rows by the same width.
      actual.reset();
      expected.reset();
      assertClose(actual.prefill(new int[] {0, 1}, 0), expected.prefill(0, 1));
    }
  }

  /** The same variation on a dense model, whose two feed-forward paths must still agree. */
  @Test
  void aDenseModelWithPerLayerFeedForwardWidthsAgreesBetweenItsTwoPaths() throws Exception {
    ToyModel model = ToyModel.createDenseVaryingFeedForward();
    Gemma4Weights weights = Gemma4Weights.fromGgufFile(model.file(), model.config());

    assertThat(model.config().sharedHiddenDim(1)).isNotEqualTo(model.config().sharedHiddenDim(0));

    Gemma4Experts experts =
        new Gemma4MappedExperts(
            model.file().fileSegment(),
            model.config().numLayers(),
            model.config().numExperts(),
            (layer, expert) -> {
              throw new AssertionError("a dense model must never resolve an expert");
            });
    LayeredKvCache cache = Gemma4KvCache.create(model.config(), 8, 2);
    Gemma4ForwardPass pass = new Gemma4ForwardPass(model.config(), weights, cache, experts);

    float[] serial = pass.forward(0, 0);
    float[] serialSecond = pass.forward(1, 1);
    pass.reset();
    float[] batched = pass.prefill(new int[] {0, 1}, 0);

    assertThat(serial).isNotNull();
    assertClose(batched, serialSecond);
  }

  @Test
  void aDenseGemma4LoadsAndItsBatchedPathAgreesWithOneTokenAtATime() throws Exception {
    // Two separate dense branches were added -- one in the single-token layer step and one in the
    // batched prefill step -- so the invariant that matters is that they agree. A mistake in either
    // shows up here, and this also proves the loader tolerates the six routed tensors a dense GGUF
    // does not carry and that nothing dereferences the null router projection or expert layout.
    ToyModel model = ToyModel.createDense();
    Gemma4Weights weights = Gemma4Weights.fromGgufFile(model.file(), model.config());

    assertThat(model.config().isDense()).isTrue();
    assertThat(weights.expertLayout())
        .describedAs("a dense model has no expert tensor layout to build")
        .isNull();

    Gemma4Experts experts =
        new Gemma4MappedExperts(
            model.file().fileSegment(),
            model.config().numLayers(),
            model.config().numExperts(),
            (layer, expert) -> {
              throw new AssertionError("a dense model must never resolve an expert");
            });
    LayeredKvCache cache = Gemma4KvCache.create(model.config(), 8, 2);
    Gemma4ForwardPass pass = new Gemma4ForwardPass(model.config(), weights, cache, experts);

    float[] step0 = pass.forward(0, 0).clone();
    float[] step1 = pass.forward(1, 1).clone();
    for (float value : step1) {
      assertThat(Float.isFinite(value)).describedAs("dense logits must be finite").isTrue();
    }

    pass.reset();
    float[] batched = pass.prefill(new int[] {0, 1}, 0);

    assertThat(batched)
        .describedAs("batched dense prefill must match the single-token dense path")
        .containsExactly(step1);
    assertThat(step0).isNotEqualTo(step1);
  }

  @Test
  void isolatesSessionsAndSupportsSpeculativeVerification() throws Exception {
    ToyModel model = ToyModel.create();
    Gemma4Weights weights = Gemma4Weights.fromGgufFile(model.file(), model.config());
    Gemma4TensorLayout layout = weights.expertLayout();

    try (Gemma4ExpertCache experts =
        new Gemma4ExpertCache(
            new Gemma4ExpertLoader(reader(model.file().fileSegment())),
            model.config().numLayers(),
            model.config().numExperts(),
            1,
            (layer, expert) -> layout.layer(layer).expert(expert),
            Gemma4ExpertCache.CachePolicy.LFU)) {
      Gemma4ForwardPass actual =
          new Gemma4ForwardPass(
              model.config(), weights, Gemma4KvCache.create(model.config(), 8, 2), experts);
      Gemma4ForwardPass.Session first = actual.openSession();
      Gemma4ForwardPass.Session second = actual.openSession();
      ScalarReference expectedFirst = new ScalarReference(model);
      ScalarReference expectedSecond = new ScalarReference(model);

      assertClose(actual.prefill(first, new int[] {0}, 0), expectedFirst.forward(0));
      assertClose(actual.prefill(second, new int[] {1}, 0), expectedSecond.forward(1));

      LogitBatch batch =
          actual.forwardBatch(new Gemma4ForwardPass.Session[] {first, second}, new int[] {1, 0});
      assertClose(batch.copyRow(0), expectedFirst.forward(1));
      assertClose(batch.copyRow(1), expectedSecond.forward(0));
      assertThat(first.checkpoint()).isEqualTo(2);
      assertThat(second.checkpoint()).isEqualTo(2);

      actual.rewind(first, 1);
      ScalarReference rewoundFirst = new ScalarReference(model);
      rewoundFirst.forward(0);
      assertClose(actual.forward(first, 0, 1), rewoundFirst.forward(0));

      actual.reset(second);
      ScalarReference resetSecond = new ScalarReference(model);
      assertClose(actual.prefill(second, new int[] {0, 1}, 0), resetSecond.prefill(0, 1));

      ScalarReference expectedDefault = new ScalarReference(model);
      LogitBatch verification = actual.verify(new int[] {0, 1}, 0);
      assertClose(verification.copyRow(0), expectedDefault.forward(0));
      assertClose(verification.copyRow(1), expectedDefault.forward(1));
      assertThat(actual.checkpoint()).isEqualTo(2);
    }
  }

  @Test
  void injectedKernelExecutesEligibleGemmaProjections() throws Exception {
    ToyModel model = ToyModel.create();
    Gemma4Config config = model.withExpertsUsed(2);
    Gemma4Weights weights = Gemma4Weights.fromGgufFile(model.file(), config);
    Gemma4TensorLayout layout = weights.expertLayout();
    AtomicInteger invocations = new AtomicInteger();
    AtomicInteger dualInvocations = new AtomicInteger();
    AtomicInteger tripleInvocations = new AtomicInteger();
    AtomicInteger groupedInvocations = new AtomicInteger();
    AtomicInteger independentInvocations = new AtomicInteger();
    AtomicInteger raggedIndependentInvocations = new AtomicInteger();
    AtomicInteger maximumBatchSize = new AtomicInteger();
    GgufBatchedMatrixKernel kernel =
        new GgufBatchedMatrixKernel() {
          @Override
          public boolean supports(GgufTensorType type) {
            return type == GgufTensorType.F32;
          }

          @Override
          public void multiply(
              float[] output,
              float[] input,
              MemorySegment matrix,
              GgufTensorType type,
              int batchSize,
              int rows,
              int cols) {
            invocations.incrementAndGet();
            maximumBatchSize.accumulateAndGet(batchSize, Math::max);
            matmulBatch(output, input, matrix, type, batchSize, rows, cols);
          }

          @Override
          public boolean isDualEligible(
              GgufTensorType firstType,
              int firstRows,
              GgufTensorType secondType,
              int secondRows,
              int batchSize,
              int cols) {
            return supports(firstType) && supports(secondType);
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
            maximumBatchSize.accumulateAndGet(batchSize, Math::max);
            matmulBatch(firstOutput, input, firstWeights, firstType, batchSize, firstRows, cols);
            matmulBatch(
                secondOutput, input, secondWeights, secondType, batchSize, secondRows, cols);
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
            return supports(firstType) && supports(secondType) && supports(thirdType);
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
            maximumBatchSize.accumulateAndGet(batchSize, Math::max);
            matmulBatch(firstOutput, input, firstWeights, firstType, batchSize, firstRows, cols);
            matmulBatch(
                secondOutput, input, secondWeights, secondType, batchSize, secondRows, cols);
            matmulBatch(thirdOutput, input, thirdWeights, thirdType, batchSize, thirdRows, cols);
          }

          @Override
          public boolean isGroupedEligible(
              GgufTensorType[] types, int[] rows, int matrixCount, int batchSize, int cols) {
            for (int index = 0; index < matrixCount; index++) {
              if (!supports(types[index])) {
                return false;
              }
            }
            return matrixCount > 1;
          }

          @Override
          public void multiplyGrouped(
              float[][] outputs,
              MemorySegment[] matrices,
              GgufTensorType[] types,
              int[] rows,
              int matrixCount,
              float[] input,
              int batchSize,
              int cols) {
            groupedInvocations.incrementAndGet();
            maximumBatchSize.accumulateAndGet(batchSize, Math::max);
            for (int index = 0; index < matrixCount; index++) {
              matmulBatch(
                  outputs[index],
                  input,
                  matrices[index],
                  types[index],
                  batchSize,
                  rows[index],
                  cols);
            }
          }

          @Override
          public boolean isIndependentEligible(
              GgufTensorType[] types, int[] rows, int matrixCount, int batchSize, int cols) {
            for (int index = 0; index < matrixCount; index++) {
              if (!supports(types[index])) {
                return false;
              }
            }
            return matrixCount > 1;
          }

          @Override
          public void multiplyIndependent(
              float[][] outputs,
              MemorySegment[] matrices,
              GgufTensorType[] types,
              int[] rows,
              int matrixCount,
              float[][] inputs,
              int batchSize,
              int cols) {
            independentInvocations.incrementAndGet();
            maximumBatchSize.accumulateAndGet(batchSize, Math::max);
            for (int index = 0; index < matrixCount; index++) {
              matmulBatch(
                  outputs[index],
                  inputs[index],
                  matrices[index],
                  types[index],
                  batchSize,
                  rows[index],
                  cols);
            }
          }

          @Override
          public boolean isRaggedIndependentEligible(
              GgufTensorType[] types, int[] rows, int[] batchSizes, int matrixCount, int cols) {
            for (int index = 0; index < matrixCount; index++) {
              if (!supports(types[index]) || batchSizes[index] < 1) {
                return false;
              }
            }
            return matrixCount > 1;
          }

          @Override
          public void multiplyRaggedIndependent(
              float[][] outputs,
              MemorySegment[] matrices,
              GgufTensorType[] types,
              int[] rows,
              int[] batchSizes,
              int matrixCount,
              float[][] inputs,
              int cols) {
            raggedIndependentInvocations.incrementAndGet();
            for (int index = 0; index < matrixCount; index++) {
              matmulBatch(
                  outputs[index],
                  inputs[index],
                  matrices[index],
                  types[index],
                  batchSizes[index],
                  rows[index],
                  cols);
            }
          }
        };

    try (Gemma4ExpertCache baselineExperts =
            new Gemma4ExpertCache(
                new Gemma4ExpertLoader(reader(model.file().fileSegment())),
                config.numLayers(),
                config.numExperts(),
                2,
                (layer, expert) -> layout.layer(layer).expert(expert),
                Gemma4ExpertCache.CachePolicy.LFU);
        Gemma4ExpertCache acceleratedExperts =
            new Gemma4ExpertCache(
                new Gemma4ExpertLoader(reader(model.file().fileSegment())),
                config.numLayers(),
                config.numExperts(),
                2,
                (layer, expert) -> layout.layer(layer).expert(expert),
                Gemma4ExpertCache.CachePolicy.LFU)) {
      Gemma4ForwardPass baseline =
          new Gemma4ForwardPass(
              config, weights, Gemma4KvCache.create(config, 8, 2), baselineExperts);
      Gemma4ForwardPass accelerated =
          new Gemma4ForwardPass(
              config, weights, Gemma4KvCache.create(config, 8, 2), acceleratedExperts, kernel);

      assertThat(baseline.prefillBatchSize()).isEqualTo(1);
      assertThat(accelerated.prefillBatchSize()).isEqualTo(128);
      assertClose(accelerated.forward(0, 0), baseline.forward(0, 0));
      assertThat(invocations).hasValueGreaterThan(0);
      assertThat(dualInvocations).hasValueGreaterThan(0);
      assertThat(tripleInvocations).hasValueGreaterThan(0);
      assertThat(groupedInvocations).hasValueGreaterThan(0);
      assertThat(independentInvocations).hasValueGreaterThan(0);

      accelerated.reset();
      baseline.reset();
      maximumBatchSize.set(0);
      int[] prompt = {0, 1, 1, 0, 1, 0, 0, 1};
      assertClose(accelerated.prefill(prompt, 0), baseline.prefill(prompt, 0));
      assertThat(raggedIndependentInvocations).hasValueGreaterThan(0);
      assertThat(maximumBatchSize).hasValue(8);
    }
  }

  @Test
  void singletonQuantizedExpertProjectionFallsBackWhenNativeKernelRequiresBatching()
      throws Exception {
    ToyModel model = ToyModel.create();
    Gemma4Weights weights = Gemma4Weights.fromGgufFile(model.file(), model.config());
    Gemma4TensorLayout layout = weights.expertLayout();
    GgufBatchedMatrixKernel batchOnlyKernel =
        new GgufBatchedMatrixKernel() {
          @Override
          public boolean supports(GgufTensorType type) {
            return type == GgufTensorType.Q4_K;
          }

          @Override
          public boolean isEligible(GgufTensorType type, int batchSize, int rows, int cols) {
            return batchSize > 1 && supports(type);
          }

          @Override
          public void multiply(
              float[] output,
              float[] input,
              MemorySegment matrix,
              GgufTensorType type,
              int batchSize,
              int rows,
              int cols) {
            throw new AssertionError("singleton projection must use the Java fallback");
          }
        };

    try (Gemma4ExpertCache experts =
        new Gemma4ExpertCache(
            new Gemma4ExpertLoader(reader(model.file().fileSegment())),
            model.config().numLayers(),
            model.config().numExperts(),
            1,
            (layer, expert) -> layout.layer(layer).expert(expert),
            Gemma4ExpertCache.CachePolicy.LFU)) {
      Gemma4ForwardPass forwardPass =
          new Gemma4ForwardPass(
              model.config(),
              weights,
              Gemma4KvCache.create(model.config(), 8, 2),
              experts,
              batchOnlyKernel);
      float[] expectedInput = new float[256];
      for (int index = 0; index < expectedInput.length; index++) {
        expectedInput[index] = (index - 128) * 0.0078125f;
      }
      float[] oversizedInput = new float[128 * 256];
      System.arraycopy(expectedInput, 0, oversizedInput, 0, expectedInput.length);
      for (int index = expectedInput.length; index < oversizedInput.length; index++) {
        oversizedInput[index] = 1.0f;
      }
      MemorySegment matrix = MemorySegment.ofArray(q4KBlock(0.125f, 0.0625f, 7));
      float[] expected = new float[1];
      float[] actual = new float[128];
      TensorOps.ggufMatmul(expected, expectedInput, matrix, GgufTensorType.Q4_K, 1, 256);

      forwardPass.projectBatched(matrix, GgufTensorType.Q4_K, 1, 256, oversizedInput, 1, actual);

      assertThat(actual[0]).isEqualTo(expected[0]);
      assertThat(Arrays.copyOfRange(actual, 1, actual.length)).containsOnly(0.0f);
    }
  }

  private static void assertClose(float[] actual, float[] expected) {
    assertThat(actual).hasSameSizeAs(expected);
    for (int index = 0; index < actual.length; index++) {
      assertThat(actual[index]).as("logit %s", index).isCloseTo(expected[index], offset(2.0e-5f));
    }
  }

  private static void matmulBatch(
      float[] output,
      float[] input,
      MemorySegment matrix,
      GgufTensorType type,
      int batchSize,
      int rows,
      int cols) {
    for (int batch = 0; batch < batchSize; batch++) {
      float[] inputRow = Arrays.copyOfRange(input, batch * cols, (batch + 1) * cols);
      float[] outputRow = new float[rows];
      TensorOps.ggufMatmul(outputRow, inputRow, matrix, type, rows, cols);
      System.arraycopy(outputRow, 0, output, batch * rows, rows);
    }
  }

  private static Gemma4ExpertLoader.PositionalReader reader(MemorySegment source) {
    return (destination, filePosition) -> {
      long available = source.byteSize() - filePosition;
      if (available <= 0) {
        return -1;
      }
      int count = Math.toIntExact(Math.min(destination.remaining(), available));
      destination.put(source.asSlice(filePosition, count).asByteBuffer());
      return count;
    };
  }

  @Test
  void anESeriesModelRunsWithPerLayerEmbeddingsAndASharedKeyValueLayer() {
    // These two features were refused outright until now, so nothing exercised them. The mapping
    // asserted here is the reference's: a sliding sharing layer reads kvOwningLayers - 2.
    ToyModel model = ToyModel.createESeries(220);
    Gemma4Config config = model.config();

    assertThat(config.usesPerLayerEmbeddings()).isTrue();
    assertThat(config.perLayerEmbeddingDim()).isEqualTo(2);
    assertThat(config.kvOwningLayers()).isEqualTo(2);
    assertThat(config.ownsKvCache(0)).isTrue();
    assertThat(config.ownsKvCache(1)).isTrue();
    assertThat(config.ownsKvCache(2)).isFalse();
    assertThat(config.kvSourceLayer(2)).isEqualTo(0);

    Gemma4ForwardPass pass = eSeriesPass(model);
    float[] first = pass.forward(0, 0).clone();
    float[] second = pass.forward(1, 1).clone();

    for (float value : first) {
      assertThat(Float.isFinite(value)).describedAs("E-series logits must be finite").isTrue();
    }
    for (float value : second) {
      assertThat(Float.isFinite(value)).isTrue();
    }
    assertThat(second).isNotEqualTo(first);
  }

  @Test
  void theSharingLayersOwnKeyAndValueWeightsNeverReachTheOutput() {
    // The sharpest statement of key-value sharing, and unlike probing the cache it cannot be
    // confused by cache layout: layer 2 attends to layer 0's cache, so its OWN key and value
    // projections are dead weight. Two models differing in nothing but those tensors must agree
    // exactly. Were layer 2 computing and storing them it would also be overwriting the very cache
    // it is meant to read.
    float[] withOneKv = eSeriesLogits(ToyModel.createESeries(220, 32, 30));
    float[] withAnother = eSeriesLogits(ToyModel.createESeries(220, 55, 30));

    assertThat(withAnother).containsExactly(withOneKv);

    // The control: varying an OWNING layer's key weights DOES change the output, so the equality
    // above is a real property of sharing and not the fixture ignoring these tensors.
    float[] owningChanged = eSeriesLogits(ToyModel.createESeries(220, 32, 44));
    assertThat(owningChanged).isNotEqualTo(withOneKv);
  }

  /**
   * The same tensor as F16, which is how Gemma 4 E4B ships it where E2B ships BF16.
   *
   * <p>Two distinct gaps, one after the other, in the same field of the same tensor: adding BF16
   * got E2B loading and left E4B refused with "unsupported matrix type F16". And F16 needed more
   * than the accepted-type set -- {@code ggufMatmul} handled F16 only in its batched form, while
   * the per-layer projection runs a token at a time, so accepting the type alone would have moved
   * the failure from the loader into the kernel.
   *
   * <p>Checked against an F32 twin holding the same F16-rounded values rather than just "it loads",
   * because the half-precision path has to compute the right numbers, not merely be admitted. (Not
   * against the scalar reference: that reference implements the mixture-of-experts layout, and the
   * E-series fixture is dense.)
   */
  @Test
  void anESeriesModelWhosePerLayerProjectionIsF16MatchesAnF32Twin() throws Exception {
    float[] half = eSeriesLogits(ToyModel.createESeriesF16Projection());
    float[] single = eSeriesLogits(ToyModel.createESeriesF16ProjectionAsFloat32());

    assertThat(half).hasSameSizeAs(single);
    for (int index = 0; index < single.length; index++) {
      assertThat(half[index])
          .describedAs("F16 logit %s must equal its F32 twin", index)
          .isEqualTo(single[index], within(1.0e-5f));
    }
  }

  @Test
  void anESeriesModelWhosePerLayerProjectionIsBf16Loads() {
    // The published E-series files store per_layer_model_proj as BF16 while everything around it is
    // a
    // K-quant. The loader's accepted-matrix set omitted BF16, so every real E-series model was
    // rejected at load with "unsupported matrix type BF16" -- a gap the all-F32 fixtures could not
    // show, and which only a real model surfaced. ggufMatmul has handled BF16 all along.
    ToyModel model = ToyModel.createESeriesBf16Projection();

    Gemma4ForwardPass pass =
        new Gemma4ForwardPass(
            model.config(),
            Gemma4Weights.fromGgufFile(model.file(), model.config()),
            Gemma4KvCache.create(model.config(), 8, 2),
            noExperts(model));
    float[] logits = pass.forward(0, 0);

    for (float value : logits) {
      assertThat(Float.isFinite(value)).isTrue();
    }
  }

  @Test
  void thePerLayerEmbeddingTableChangesTheOutput() {
    // Two models differing in nothing but the per-layer embedding table. If the table were ignored
    // -- the failure mode that matters, because it produces plausible output -- these would agree.
    float[] withOneTable = eSeriesLogits(ToyModel.createESeries(220));
    float[] withAnother = eSeriesLogits(ToyModel.createESeries(221));

    assertThat(withAnother)
        .describedAs("the per-layer embedding table must actually reach the residual stream")
        .isNotEqualTo(withOneTable);
  }

  @Test
  void theESeriesDoesNotTakeTheBatchedPrefillPath() {
    // Its batched layer step carries neither the per-layer fold nor the key-value sharing, so
    // batching it would store keys and values for the sharing layers. Serial is correct and slower.
    ToyModel model = ToyModel.createESeries(220);
    Gemma4ForwardPass pass = eSeriesPass(model);

    assertThat(pass.usesBatchedPrefill()).isFalse();
    float[] serial = eSeriesLogits(model);
    pass.reset();
    float[] prefilled = pass.prefill(new int[] {0, 1}, 0);
    assertThat(prefilled).hasSameSizeAs(serial);
  }

  private static float[] eSeriesLogits(ToyModel model) {
    Gemma4ForwardPass pass = eSeriesPass(model);
    pass.forward(0, 0);
    return pass.forward(1, 1).clone();
  }

  private static Gemma4ForwardPass eSeriesPass(ToyModel model) {
    return new Gemma4ForwardPass(
        model.config(),
        Gemma4Weights.fromGgufFile(model.file(), model.config()),
        Gemma4KvCache.create(model.config(), 8, 2),
        noExperts(model));
  }

  private static Gemma4Experts noExperts(ToyModel model) {
    return new Gemma4MappedExperts(
        model.file().fileSegment(),
        model.config().numLayers(),
        model.config().numExperts(),
        (layer, expert) -> {
          throw new AssertionError("a dense or E-series model must never resolve an expert");
        });
  }

  private record ToyModel(Gemma4Config config, GgufFile file, Map<String, float[]> tensors) {

    private Gemma4Config withExpertsUsed(int expertsUsed) {
      return new Gemma4Config(
          config.embeddingDim(),
          config.numLayers(),
          config.numHeads(),
          config.kvHeadsByLayer(),
          config.fullKeyLength(),
          config.slidingKeyLength(),
          config.fullValueLength(),
          config.slidingValueLength(),
          config.vocabSize(),
          config.contextLength(),
          config.sharedHiddenDim(0),
          config.expertHiddenDim(),
          config.numExperts(),
          expertsUsed,
          config.fullRopeTheta(),
          config.slidingRopeTheta(),
          config.fullRopeDimension(),
          config.slidingRopeDimension(),
          config.rmsNormEps(),
          config.slidingWindow(),
          config.slidingWindowByLayer(),
          config.finalLogitSoftcap());
    }

    private static ToyModel create() {
      Gemma4Config config =
          new Gemma4Config(
              4,
              2,
              1,
              List.of(1, 1),
              4,
              4,
              4,
              4,
              4,
              8,
              3,
              2,
              2,
              1,
              1_000_000.0f,
              10_000.0f,
              4,
              4,
              1.0e-6f,
              2,
              List.of(true, false),
              30.0f);
      FixtureBuilder fixture = new FixtureBuilder();
      fixture.f32(
          "token_embd.weight",
          new long[] {4, 4},
          new float[] {
            0.20f, -0.10f, 0.30f, 0.40f,
            -0.30f, 0.50f, 0.20f, -0.40f,
            0.60f, 0.10f, -0.20f, 0.30f,
            -0.20f, -0.30f, 0.40f, 0.50f
          });
      fixture.f32("output_norm.weight", new long[] {4}, norm(9));
      fixture.f32("rope_freqs.weight", new long[] {2}, new float[] {1.0f, 1.0e30f});
      addLayer(fixture, 0, true);
      addLayer(fixture, 1, false);
      return new ToyModel(config, fixture.build(), Map.copyOf(fixture.values));
    }

    /**
     * The routed toy model with a <b>per-layer</b> shared feed-forward width, so the independent
     * scalar reference can check the arithmetic and not merely that the file loads.
     */
    private static ToyModel createVaryingSharedFeedForward() {
      Gemma4Config config =
          new Gemma4Config(
              4,
              2,
              1,
              List.of(1, 1),
              4,
              4,
              4,
              4,
              4,
              8,
              List.of(3, 5),
              2,
              2,
              1,
              1_000_000.0f,
              10_000.0f,
              4,
              4,
              1.0e-6f,
              2,
              List.of(true, false),
              30.0f,
              0,
              0);
      FixtureBuilder fixture = new FixtureBuilder();
      fixture.f32(
          "token_embd.weight",
          new long[] {4, 4},
          new float[] {
            0.20f, -0.10f, 0.30f, 0.40f,
            -0.30f, 0.50f, 0.20f, -0.40f,
            0.60f, 0.10f, -0.20f, 0.30f,
            -0.20f, -0.30f, 0.40f, 0.50f
          });
      fixture.f32("output_norm.weight", new long[] {4}, norm(9));
      fixture.f32("rope_freqs.weight", new long[] {2}, new float[] {1.0f, 1.0e30f});
      addLayer(fixture, 0, true, 3);
      addLayer(fixture, 1, false, 5);
      return new ToyModel(config, fixture.build(), Map.copyOf(fixture.values));
    }

    /**
     * A dense model whose feed-forward is <b>wider on layer 1 than on layer 0</b>, as an E-series
     * MatFormer is.
     *
     * <p>Layer 1's width is deliberately the larger one, so a buffer or a shape check taken from
     * layer 0 is too small rather than merely different -- the shape of the defect that refused
     * Gemma 4 E2B at blk.15.
     */
    private static ToyModel createDenseVaryingFeedForward() {
      Gemma4Config config =
          new Gemma4Config(
              4,
              2,
              1,
              List.of(1, 1),
              4,
              4,
              4,
              4,
              4,
              8,
              List.of(3, 5),
              0,
              0,
              0,
              1_000_000.0f,
              10_000.0f,
              4,
              4,
              1.0e-6f,
              2,
              List.of(true, false),
              30.0f,
              0,
              0);
      FixtureBuilder fixture = new FixtureBuilder();
      fixture.f32(
          "token_embd.weight",
          new long[] {4, 4},
          new float[] {
            0.20f, -0.10f, 0.30f, 0.40f,
            -0.30f, 0.50f, 0.20f, -0.40f,
            0.60f, 0.10f, -0.20f, 0.30f,
            -0.20f, -0.30f, 0.40f, 0.50f
          });
      fixture.f32("output_norm.weight", new long[] {4}, norm(9));
      fixture.f32("rope_freqs.weight", new long[] {2}, new float[] {1.0f, 1.0e30f});
      addDenseLayer(fixture, 0, true, 30, 3);
      addDenseLayer(fixture, 1, false, 31, 5);
      return new ToyModel(config, fixture.build(), Map.copyOf(fixture.values));
    }

    /**
     * A dense Gemma 4: no expert metadata, and none of the routed tensors. Mirrors what the real
     * 12B/31B GGUFs carry -- zero {@code *_exps}, no router, and a single {@code post_ffw_norm}.
     */
    private static ToyModel createDense() {
      Gemma4Config config =
          new Gemma4Config(
              4,
              2,
              1,
              List.of(1, 1),
              4,
              4,
              4,
              4,
              4,
              8,
              3, // sharedHiddenDim: the dense feed-forward width
              0,
              0,
              0, // expertHiddenDim, numExperts, numExpertsUsed -- dense
              1_000_000.0f,
              10_000.0f,
              4,
              4,
              1.0e-6f,
              2,
              List.of(true, false),
              30.0f);
      FixtureBuilder fixture = new FixtureBuilder();
      fixture.f32(
          "token_embd.weight",
          new long[] {4, 4},
          new float[] {
            0.20f, -0.10f, 0.30f, 0.40f,
            -0.30f, 0.50f, 0.20f, -0.40f,
            0.60f, 0.10f, -0.20f, 0.30f,
            -0.20f, -0.30f, 0.40f, 0.50f
          });
      fixture.f32("output_norm.weight", new long[] {4}, norm(9));
      fixture.f32("rope_freqs.weight", new long[] {2}, new float[] {1.0f, 1.0e30f});
      addDenseLayer(fixture, 0, true);
      addDenseLayer(fixture, 1, false);
      return new ToyModel(config, fixture.build(), Map.copyOf(fixture.values));
    }

    /**
     * An E-series shaped model: per-layer input embeddings plus a shared key-value layer.
     *
     * <p>Three layers with {@code sharedKvLayers=1}, so layers 0 and 1 own a cache and layer 2
     * shares one. Layer 2 is sliding, so it reads {@code kvOwningLayers - 2 = 0} -- layer 0, also
     * sliding, with the same head dimension. That is the mapping the reference uses and the reason
     * the -2/-1 is there.
     *
     * @param perLayerSeed varies only the per-layer embedding table, so two models built with
     *     different seeds differ in nothing else
     */
    private static ToyModel createESeries(int perLayerSeed) {
      return createESeries(perLayerSeed, 32, 30);
    }

    private static ToyModel createESeries(
        int perLayerSeed, int sharingLayerKvSeed, int owningLayerKvSeed) {
      Gemma4Config config =
          new Gemma4Config(
              4,
              3,
              1,
              List.of(1, 1, 1),
              4,
              4,
              4,
              4,
              4,
              8,
              3,
              0,
              0,
              0,
              1_000_000.0f,
              10_000.0f,
              4,
              4,
              1.0e-6f,
              2,
              List.of(true, false, true),
              30.0f,
              1,
              2);
      FixtureBuilder fixture = new FixtureBuilder();
      fixture.f32(
          "token_embd.weight",
          new long[] {4, 4},
          new float[] {
            0.20f, -0.10f, 0.30f, 0.40f,
            -0.30f, 0.50f, 0.20f, -0.40f,
            0.60f, 0.10f, -0.20f, 0.30f,
            -0.20f, -0.30f, 0.40f, 0.50f
          });
      fixture.f32("output_norm.weight", new long[] {4}, norm(9));
      fixture.f32("rope_freqs.weight", new long[] {2}, new float[] {1.0f, 1.0e30f});
      // perLayerEmbeddingDim * numLayers = 2 * 3 = 6
      fixture.f32("per_layer_token_embd.weight", new long[] {6, 4}, matrix(4, 6, perLayerSeed));
      fixture.f32("per_layer_model_proj.weight", new long[] {4, 6}, matrix(6, 4, 170));
      // Length is perLayerEmbeddingDim, not the model width: this norm runs over each 2-wide slice.
      fixture.f32("per_layer_proj_norm.weight", new long[] {2}, norm(180, 2));
      addESeriesLayer(fixture, 0, true, owningLayerKvSeed);
      addESeriesLayer(fixture, 1, false, 31);
      addESeriesLayer(fixture, 2, true, sharingLayerKvSeed);
      return new ToyModel(config, fixture.build(), Map.copyOf(fixture.values));
    }

    /**
     * The F16 fixture's twin, with the per-layer projection written as F32 holding exactly the
     * values F16 rounds to. The reference for the half-precision path: same numbers, different
     * encoding.
     */
    private static ToyModel createESeriesF16ProjectionAsFloat32() {
      FixtureBuilder fixture = new FixtureBuilder();
      fixture.f32(
          "token_embd.weight",
          new long[] {4, 4},
          new float[] {
            0.20f, -0.10f, 0.30f, 0.40f,
            -0.30f, 0.50f, 0.20f, -0.40f,
            0.60f, 0.10f, -0.20f, 0.30f,
            -0.20f, -0.30f, 0.40f, 0.50f
          });
      fixture.f32("output_norm.weight", new long[] {4}, norm(9));
      fixture.f32("rope_freqs.weight", new long[] {2}, new float[] {1.0f, 1.0e30f});
      fixture.f32("per_layer_token_embd.weight", new long[] {6, 4}, matrix(4, 6, 220));
      fixture.f32("per_layer_model_proj.weight", new long[] {4, 6}, halfRounded(matrix(6, 4, 170)));
      fixture.f32("per_layer_proj_norm.weight", new long[] {2}, norm(180, 2));
      addESeriesLayer(fixture, 0, true, 30);
      addESeriesLayer(fixture, 1, false, 31);
      addESeriesLayer(fixture, 2, true, 32);
      return new ToyModel(
          ToyModel.createESeries(220).config(), fixture.build(), Map.copyOf(fixture.values));
    }

    /** What F16 storage rounds each value to, so an F32 twin holds the same numbers. */
    private static float[] halfRounded(float[] source) {
      float[] rounded = new float[source.length];
      for (int index = 0; index < source.length; index++) {
        rounded[index] = Float.float16ToFloat(Float.floatToFloat16(source[index]));
      }
      return rounded;
    }

    /** The E-series with its per-layer projection as F16, which is how E4B ships it. */
    private static ToyModel createESeriesF16Projection() {
      FixtureBuilder fixture = new FixtureBuilder();
      fixture.f32(
          "token_embd.weight",
          new long[] {4, 4},
          new float[] {
            0.20f, -0.10f, 0.30f, 0.40f,
            -0.30f, 0.50f, 0.20f, -0.40f,
            0.60f, 0.10f, -0.20f, 0.30f,
            -0.20f, -0.30f, 0.40f, 0.50f
          });
      fixture.f32("output_norm.weight", new long[] {4}, norm(9));
      fixture.f32("rope_freqs.weight", new long[] {2}, new float[] {1.0f, 1.0e30f});
      fixture.f32("per_layer_token_embd.weight", new long[] {6, 4}, matrix(4, 6, 220));
      fixture.f16("per_layer_model_proj.weight", new long[] {4, 6}, matrix(6, 4, 170));
      fixture.f32("per_layer_proj_norm.weight", new long[] {2}, norm(180, 2));
      addESeriesLayer(fixture, 0, true, 30);
      addESeriesLayer(fixture, 1, false, 31);
      addESeriesLayer(fixture, 2, true, 32);
      return new ToyModel(
          ToyModel.createESeries(220).config(), fixture.build(), Map.copyOf(fixture.values));
    }

    /**
     * The E-series model with its per-layer projection stored as BF16, which is how the published
     * files ship it while their other matrices are K-quants.
     */
    private static ToyModel createESeriesBf16Projection() {
      FixtureBuilder fixture = new FixtureBuilder();
      fixture.f32(
          "token_embd.weight",
          new long[] {4, 4},
          new float[] {
            0.20f, -0.10f, 0.30f, 0.40f,
            -0.30f, 0.50f, 0.20f, -0.40f,
            0.60f, 0.10f, -0.20f, 0.30f,
            -0.20f, -0.30f, 0.40f, 0.50f
          });
      fixture.f32("output_norm.weight", new long[] {4}, norm(9));
      fixture.f32("rope_freqs.weight", new long[] {2}, new float[] {1.0f, 1.0e30f});
      fixture.f32("per_layer_token_embd.weight", new long[] {6, 4}, matrix(4, 6, 220));
      fixture.bf16("per_layer_model_proj.weight", new long[] {4, 6}, matrix(6, 4, 170));
      fixture.f32("per_layer_proj_norm.weight", new long[] {2}, norm(180, 2));
      addESeriesLayer(fixture, 0, true, 30);
      addESeriesLayer(fixture, 1, false, 31);
      addESeriesLayer(fixture, 2, true, 32);
      return new ToyModel(
          ToyModel.createESeries(220).config(), fixture.build(), Map.copyOf(fixture.values));
    }

    private static void addESeriesLayer(FixtureBuilder fixture, int layer, boolean sliding) {
      addESeriesLayer(fixture, layer, sliding, 30 + layer);
    }

    private static void addESeriesLayer(
        FixtureBuilder fixture, int layer, boolean sliding, int keyValueSeed) {
      addDenseLayer(fixture, layer, sliding, keyValueSeed);
      String prefix = "blk." + layer + ".";
      fixture.f32(prefix + "inp_gate.weight", new long[] {4, 2}, matrix(2, 4, 190 + layer));
      fixture.f32(prefix + "proj.weight", new long[] {2, 4}, matrix(4, 2, 200 + layer));
      fixture.f32(prefix + "post_norm.weight", new long[] {4}, norm(210 + layer));
    }

    private static void addDenseLayer(FixtureBuilder fixture, int layer, boolean sliding) {
      addDenseLayer(fixture, layer, sliding, 30 + layer);
    }

    /**
     * @param keyValueSeed seeds the key and value projections, so one layer's can be varied alone.
     *     FixtureBuilder APPENDS, so writing a tensor twice leaves two entries under one name and
     *     the loader reads the first -- an override after the fact is silently ignored, which is
     *     why the seed is threaded in here instead.
     */
    private static void addDenseLayer(
        FixtureBuilder fixture, int layer, boolean sliding, int keyValueSeed) {
      addDenseLayer(fixture, layer, sliding, keyValueSeed, 3);
    }

    private static void addDenseLayer(
        FixtureBuilder fixture, int layer, boolean sliding, int keyValueSeed, int hidden) {
      String prefix = "blk." + layer + ".";
      fixture.f32(prefix + "attn_norm.weight", new long[] {4}, norm(11 + layer));
      fixture.f32(prefix + "attn_q.weight", new long[] {4, 4}, matrix(4, 4, 20 + layer));
      fixture.f32(prefix + "attn_k.weight", new long[] {4, 4}, matrix(4, 4, keyValueSeed));
      if (sliding) {
        fixture.f32(prefix + "attn_v.weight", new long[] {4, 4}, matrix(4, 4, keyValueSeed + 10));
      }
      fixture.f32(prefix + "attn_output.weight", new long[] {4, 4}, matrix(4, 4, 50 + layer));
      fixture.f32(prefix + "attn_q_norm.weight", new long[] {4}, norm(60 + layer));
      fixture.f32(prefix + "attn_k_norm.weight", new long[] {4}, norm(70 + layer));
      fixture.f32(prefix + "post_attention_norm.weight", new long[] {4}, norm(80 + layer));
      fixture.f32(prefix + "ffn_norm.weight", new long[] {4}, norm(90 + layer));
      fixture.f32(
          prefix + "ffn_gate.weight", new long[] {4, hidden}, matrix(hidden, 4, 100 + layer));
      fixture.f32(prefix + "ffn_up.weight", new long[] {4, hidden}, matrix(hidden, 4, 110 + layer));
      fixture.f32(
          prefix + "ffn_down.weight", new long[] {hidden, 4}, matrix(4, hidden, 120 + layer));
      // Only post_ffw_norm -- no pre_ffw_norm_2, post_ffw_norm_1, post_ffw_norm_2, router or
      // experts.
      fixture.f32(prefix + "post_ffw_norm.weight", new long[] {4}, norm(160 + layer));
      fixture.f32(
          prefix + "layer_output_scale.weight",
          new long[] {1},
          new float[] {0.90f + 0.05f * layer});
    }

    private static void addLayer(FixtureBuilder fixture, int layer, boolean sliding) {
      addLayer(fixture, layer, sliding, 3);
    }

    private static void addLayer(
        FixtureBuilder fixture, int layer, boolean sliding, int sharedHidden) {
      String prefix = "blk." + layer + ".";
      fixture.f32(prefix + "attn_norm.weight", new long[] {4}, norm(11 + layer));
      fixture.f32(prefix + "attn_q.weight", new long[] {4, 4}, matrix(4, 4, 20 + layer));
      fixture.f32(prefix + "attn_k.weight", new long[] {4, 4}, matrix(4, 4, 30 + layer));
      if (sliding) {
        fixture.f32(prefix + "attn_v.weight", new long[] {4, 4}, matrix(4, 4, 40));
      }
      fixture.f32(prefix + "attn_output.weight", new long[] {4, 4}, matrix(4, 4, 50 + layer));
      fixture.f32(prefix + "attn_q_norm.weight", new long[] {4}, norm(60 + layer));
      fixture.f32(prefix + "attn_k_norm.weight", new long[] {4}, norm(70 + layer));
      fixture.f32(prefix + "post_attention_norm.weight", new long[] {4}, norm(80 + layer));

      fixture.f32(prefix + "ffn_norm.weight", new long[] {4}, norm(90 + layer));
      fixture.f32(
          prefix + "ffn_gate.weight",
          new long[] {4, sharedHidden},
          matrix(sharedHidden, 4, 100 + layer));
      fixture.f32(
          prefix + "ffn_up.weight",
          new long[] {4, sharedHidden},
          matrix(sharedHidden, 4, 110 + layer));
      fixture.f32(
          prefix + "ffn_down.weight",
          new long[] {sharedHidden, 4},
          matrix(4, sharedHidden, 120 + layer));
      fixture.f32(prefix + "pre_ffw_norm_2.weight", new long[] {4}, norm(130 + layer));
      fixture.f32(prefix + "post_ffw_norm_1.weight", new long[] {4}, norm(140 + layer));
      fixture.f32(prefix + "post_ffw_norm_2.weight", new long[] {4}, norm(150 + layer));
      fixture.f32(prefix + "post_ffw_norm.weight", new long[] {4}, norm(160 + layer));

      fixture.f32(
          prefix + "ffn_gate_inp.scale", new long[] {4}, new float[] {1.0f, 0.8f, 1.1f, 0.9f});
      fixture.f32(prefix + "ffn_gate_inp.weight", new long[] {4, 2}, matrix(2, 4, 170 + layer));
      fixture.f32(prefix + "ffn_down_exps.scale", new long[] {2}, new float[] {0.75f, 1.25f});
      fixture.f32(
          prefix + "layer_output_scale.weight",
          new long[] {1},
          new float[] {0.90f + 0.05f * layer});
      fixture.f32(
          prefix + "ffn_gate_up_exps.weight",
          new long[] {4, 4, 2},
          concat(matrix(4, 4, 180 + layer), matrix(4, 4, 190 + layer)));
      fixture.f32(
          prefix + "ffn_down_exps.weight",
          new long[] {2, 4, 2},
          concat(matrix(4, 2, 200 + layer), matrix(4, 2, 210 + layer)));
    }
  }

  private static final class ScalarReference {
    private final ToyModel model;
    private final List<List<float[]>> keys;
    private final List<List<float[]>> values;

    private ScalarReference(ToyModel model) {
      this.model = model;
      this.keys = new ArrayList<>(model.config().numLayers());
      this.values = new ArrayList<>(model.config().numLayers());
      reset();
    }

    private float[] prefill(int... tokens) {
      float[] logits = null;
      for (int token : tokens) {
        logits = forward(token);
      }
      return logits;
    }

    private float[] forward(int token) {
      Gemma4Config config = model.config();
      int position = keys.getFirst().size();
      float[] state = row(tensor("token_embd.weight"), token, config.embeddingDim());
      scale(state, config.embeddingScale());

      for (int layer = 0; layer < config.numLayers(); layer++) {
        String prefix = "blk." + layer + ".";
        float[] attentionInput =
            rms(state, tensor(prefix + "attn_norm.weight"), config.rmsNormEps());
        float[] query =
            matmul(tensor(prefix + "attn_q.weight"), config.queryDim(layer), attentionInput);
        float[] rawKey =
            matmul(tensor(prefix + "attn_k.weight"), config.keyDim(layer), attentionInput);
        float[] value =
            config.usesSlidingWindow(layer)
                ? matmul(tensor(prefix + "attn_v.weight"), config.valueDim(layer), attentionInput)
                : rawKey.clone();
        float[] key = rms(rawKey, tensor(prefix + "attn_k_norm.weight"), config.rmsNormEps());
        value = rms(value, null, config.rmsNormEps());
        query = rms(query, tensor(prefix + "attn_q_norm.weight"), config.rmsNormEps());
        float[] factors = config.usesSlidingWindow(layer) ? null : tensor("rope_freqs.weight");
        ropeNeox(query, position, config.ropeTheta(layer), factors);
        ropeNeox(key, position, config.ropeTheta(layer), factors);
        keys.get(layer).add(key);
        values.get(layer).add(value);

        int start = config.attentionStartPosition(layer, position);
        float[] attention = attention(query, keys.get(layer), values.get(layer), start);
        float[] projected =
            matmul(tensor(prefix + "attn_output.weight"), config.embeddingDim(), attention);
        projected =
            rms(projected, tensor(prefix + "post_attention_norm.weight"), config.rmsNormEps());
        add(state, projected);

        float[] sharedInput = rms(state, tensor(prefix + "ffn_norm.weight"), config.rmsNormEps());
        float[] sharedGate =
            matmul(tensor(prefix + "ffn_gate.weight"), config.sharedHiddenDim(layer), sharedInput);
        float[] sharedUp =
            matmul(tensor(prefix + "ffn_up.weight"), config.sharedHiddenDim(layer), sharedInput);
        float[] shared = geluGlu(sharedGate, sharedUp);
        shared = matmul(tensor(prefix + "ffn_down.weight"), config.embeddingDim(), shared);
        shared = rms(shared, tensor(prefix + "post_ffw_norm_1.weight"), config.rmsNormEps());

        float[] routedInput =
            rms(state, tensor(prefix + "pre_ffw_norm_2.weight"), config.rmsNormEps());
        float[] routerInput = rms(state, null, config.rmsNormEps());
        float[] routerScale = tensor(prefix + "ffn_gate_inp.scale");
        for (int index = 0; index < routerInput.length; index++) {
          routerInput[index] *= routerScale[index] / (float) Math.sqrt(config.embeddingDim());
        }
        float[] routerLogits =
            matmul(tensor(prefix + "ffn_gate_inp.weight"), config.numExperts(), routerInput);
        int expert = routerLogits[1] > routerLogits[0] ? 1 : 0;
        float expertWeight = tensor(prefix + "ffn_down_exps.scale")[expert];
        int gateUpStride = 2 * config.expertHiddenDim() * config.embeddingDim();
        float[] gateUp =
            matmul(
                tensor(prefix + "ffn_gate_up_exps.weight"),
                expert * gateUpStride,
                2 * config.expertHiddenDim(),
                config.embeddingDim(),
                routedInput);
        float[] routed =
            geluGlu(
                slice(gateUp, 0, config.expertHiddenDim()),
                slice(gateUp, config.expertHiddenDim(), config.expertHiddenDim()));
        int downStride = config.embeddingDim() * config.expertHiddenDim();
        routed =
            matmul(
                tensor(prefix + "ffn_down_exps.weight"),
                expert * downStride,
                config.embeddingDim(),
                config.expertHiddenDim(),
                routed);
        scale(routed, expertWeight);
        routed = rms(routed, tensor(prefix + "post_ffw_norm_2.weight"), config.rmsNormEps());

        add(shared, routed);
        float[] combined =
            rms(shared, tensor(prefix + "post_ffw_norm.weight"), config.rmsNormEps());
        add(state, combined);
        scale(state, tensor(prefix + "layer_output_scale.weight")[0]);
      }

      float[] normalized = rms(state, tensor("output_norm.weight"), config.rmsNormEps());
      float[] logits = matmul(tensor("token_embd.weight"), config.vocabSize(), normalized);
      for (int index = 0; index < logits.length; index++) {
        logits[index] =
            config.finalLogitSoftcap()
                * (float) Math.tanh(logits[index] / config.finalLogitSoftcap());
      }
      return logits;
    }

    private void reset() {
      keys.clear();
      values.clear();
      for (int layer = 0; layer < model.config().numLayers(); layer++) {
        keys.add(new ArrayList<>());
        values.add(new ArrayList<>());
      }
    }

    private float[] tensor(String name) {
      return model.tensors().get(name);
    }

    private static float[] attention(
        float[] query, List<float[]> keys, List<float[]> values, int start) {
      int count = keys.size() - start;
      float[] scores = new float[count];
      float maximum = Float.NEGATIVE_INFINITY;
      for (int index = 0; index < count; index++) {
        scores[index] = dot(query, keys.get(start + index));
        maximum = Math.max(maximum, scores[index]);
      }
      float sum = 0;
      for (int index = 0; index < count; index++) {
        scores[index] = (float) Math.exp(scores[index] - maximum);
        sum += scores[index];
      }
      float[] result = new float[query.length];
      for (int index = 0; index < count; index++) {
        float weight = scores[index] / sum;
        float[] value = values.get(start + index);
        for (int column = 0; column < result.length; column++) {
          result[column] += weight * value[column];
        }
      }
      return result;
    }

    private static float[] rms(float[] input, float[] weight, float epsilon) {
      float[] result = new float[input.length];
      float inverse = (float) (1.0 / Math.sqrt(dot(input, input) / input.length + epsilon));
      for (int index = 0; index < input.length; index++) {
        result[index] = input[index] * inverse * (weight == null ? 1.0f : weight[index]);
      }
      return result;
    }

    private static void ropeNeox(float[] vector, int position, float theta, float[] factors) {
      int pairs = vector.length / 2;
      for (int pair = 0; pair < pairs; pair++) {
        float frequency = (float) (1.0 / Math.pow(theta, (double) (pair * 2) / vector.length));
        float divisor = factors == null ? 1.0f : factors[pair];
        float angle = position * frequency / divisor;
        float cosine = (float) Math.cos(angle);
        float sine = (float) Math.sin(angle);
        float first = vector[pair];
        float second = vector[pairs + pair];
        vector[pair] = first * cosine - second * sine;
        vector[pairs + pair] = first * sine + second * cosine;
      }
    }

    private static float[] geluGlu(float[] gate, float[] up) {
      float[] result = new float[gate.length];
      for (int index = 0; index < gate.length; index++) {
        float value = gate[index];
        float gelu =
            0.5f
                * value
                * (1.0f
                    + (float)
                        Math.tanh(
                            0.7978845608028654f * value * (1.0f + 0.044715f * value * value)));
        result[index] = gelu * up[index];
      }
      return result;
    }
  }

  private static float[] matmul(float[] weights, int rows, float[] input) {
    return matmul(weights, 0, rows, input.length, input);
  }

  private static float[] matmul(float[] weights, int offset, int rows, int columns, float[] input) {
    float[] result = new float[rows];
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        result[row] += weights[offset + row * columns + column] * input[column];
      }
    }
    return result;
  }

  private static float dot(float[] left, float[] right) {
    float result = 0;
    for (int index = 0; index < left.length; index++) {
      result += left[index] * right[index];
    }
    return result;
  }

  private static float[] row(float[] values, int row, int columns) {
    return slice(values, row * columns, columns);
  }

  private static float[] slice(float[] values, int start, int length) {
    float[] result = new float[length];
    System.arraycopy(values, start, result, 0, length);
    return result;
  }

  private static void add(float[] target, float[] addition) {
    for (int index = 0; index < target.length; index++) {
      target[index] += addition[index];
    }
  }

  private static void scale(float[] values, float scale) {
    for (int index = 0; index < values.length; index++) {
      values[index] *= scale;
    }
  }

  /** A norm vector of an explicit length, for tensors narrower than the model width. */
  private static float[] norm(int seed, int length) {
    float[] values = new float[length];
    for (int index = 0; index < values.length; index++) {
      values[index] = 0.8f + ((seed + index * 3) % 7) * 0.05f;
    }
    return values;
  }

  private static float[] norm(int seed) {
    float[] values = new float[4];
    for (int index = 0; index < values.length; index++) {
      values[index] = 0.8f + ((seed + index * 3) % 7) * 0.05f;
    }
    return values;
  }

  private static float[] matrix(int rows, int columns, int seed) {
    float[] values = new float[rows * columns];
    for (int index = 0; index < values.length; index++) {
      values[index] = (((seed + index * 11) % 19) - 9) * 0.035f;
    }
    return values;
  }

  private static float[] concat(float[] first, float[] second) {
    float[] result = new float[first.length + second.length];
    System.arraycopy(first, 0, result, 0, first.length);
    System.arraycopy(second, 0, result, first.length, second.length);
    return result;
  }

  private static byte[] q4KBlock(float scale, float minScale, int quant) {
    byte[] block = new byte[144];
    ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putShort(0, Float.floatToFloat16(scale));
    buffer.putShort(2, Float.floatToFloat16(minScale));
    for (int group = 0; group < 4; group++) {
      block[4 + group] = 1;
    }
    for (int group = 4; group < 8; group++) {
      block[4 + group + 4] = 1;
    }
    Arrays.fill(block, 16, block.length, (byte) (quant | (quant << 4)));
    return block;
  }

  private static final class FixtureBuilder {
    private final List<GgufTensorInfo> infos = new ArrayList<>();
    private final Map<String, float[]> values = new HashMap<>();
    private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();

    /** Writes a BF16 matrix: the top 16 bits of each float, which is exactly what BF16 is. */
    private void bf16(String name, long[] shape, float[] tensorValues) {
      GgufTensorInfo info =
          new GgufTensorInfo(name, shape.length, shape, GgufTensorType.BF16, bytes.size());
      if (tensorValues.length != info.elementCount()) {
        throw new IllegalArgumentException(name + " has the wrong value count");
      }
      ByteBuffer encoded =
          ByteBuffer.allocate(tensorValues.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      for (float value : tensorValues) {
        encoded.putShort((short) (Float.floatToIntBits(value) >>> 16));
      }
      infos.add(info);
      values.put(name, tensorValues.clone());
      bytes.writeBytes(encoded.array());
    }

    /** Writes an F16 matrix, which is how Gemma 4 E4B carries its per-layer projection. */
    private void f16(String name, long[] shape, float[] tensorValues) {
      GgufTensorInfo info =
          new GgufTensorInfo(name, shape.length, shape, GgufTensorType.F16, bytes.size());
      if (tensorValues.length != info.elementCount()) {
        throw new IllegalArgumentException(name + " has the wrong value count");
      }
      ByteBuffer encoded =
          ByteBuffer.allocate(tensorValues.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      for (float value : tensorValues) {
        encoded.putShort(Float.floatToFloat16(value));
      }
      infos.add(info);
      // The values a loader will read back are the F16-rounded ones, not the originals.
      float[] rounded = new float[tensorValues.length];
      for (int index = 0; index < tensorValues.length; index++) {
        rounded[index] = Float.float16ToFloat(Float.floatToFloat16(tensorValues[index]));
      }
      values.put(name, rounded);
      bytes.writeBytes(encoded.array());
    }

    private void f32(String name, long[] shape, float[] tensorValues) {
      GgufTensorInfo info =
          new GgufTensorInfo(name, shape.length, shape, GgufTensorType.F32, bytes.size());
      if (tensorValues.length != info.elementCount()) {
        throw new IllegalArgumentException(name + " has the wrong value count");
      }
      ByteBuffer encoded =
          ByteBuffer.allocate(tensorValues.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      for (float value : tensorValues) {
        encoded.putFloat(value);
      }
      infos.add(info);
      values.put(name, tensorValues.clone());
      bytes.writeBytes(encoded.array());
    }

    private GgufFile build() {
      return new GgufFile(
          new GgufHeader(3, infos.size(), 0),
          new GgufMetadata(Map.of()),
          infos,
          0,
          MemorySegment.ofArray(bytes.toByteArray()));
    }
  }
}
