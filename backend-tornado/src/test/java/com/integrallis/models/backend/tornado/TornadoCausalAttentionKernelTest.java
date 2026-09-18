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
package com.integrallis.models.backend.tornado;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel.AttentionScope;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Routing, mirror bookkeeping, and refusal reporting for the device attention kernel.
 *
 * <p>The plans here are {@link HostPlan}s: they hold the same buffers the TornadoVM plan holds and
 * drive the same {@link TornadoAttentionKernel} methods, executed as ordinary Java. What is not
 * covered is PTX code generation, device scheduling, and the residency of a {@code FIRST_EXECUTION}
 * buffer across executions — those need the GPU gate, and nothing here should be read as evidence
 * about them.
 */
class TornadoCausalAttentionKernelTest {

  private static final int HEADS = 8;
  private static final int KV_HEADS = 2;
  private static final int HEAD_LENGTH = 32;
  private static final int MAX_SEQUENCE = 64;
  private static final float SCALE = (float) (1.0 / Math.sqrt(HEAD_LENGTH));
  private static final AttentionScope SEQUENCE_ONE = new AttentionScope(1L, 0, SCALE, false);
  private static final AttentionScope SEQUENCE_TWO = new AttentionScope(2L, 0, SCALE, false);

  @Test
  void aDecodeStepRunsOnTheDeviceAndIsReportedAsOne() {
    try (Harness harness = new Harness()) {
      harness.mirrorAndAttend(SEQUENCE_ONE, 0, 12, 1);

      TornadoAttentionRouting routing = harness.kernel.routing();
      assertThat(routing.deviceDecodeLayerSteps()).isEqualTo(1);
      assertThat(routing.devicePrefillLayerChunks()).isZero();
      assertThat(routing.mirroredPositions()).isEqualTo(12);
      assertThat(routing.ranOnDevice()).isTrue();
      assertThat(routing.planCount()).isEqualTo(1);
      assertThat(routing.summary()).contains("device-layer-steps=1");
    }
  }

  @Test
  void aDecodeStepReproducesTheKernelsOwnResultThroughTheMirror() {
    try (Harness harness = new Harness()) {
      float[] throughMirror = harness.mirrorAndAttend(SEQUENCE_ONE, 0, 20, 1);
      float[] direct = harness.directKernelResult(20);

      // Backfilling through chunked store calls and attending must land on exactly the result the
      // kernel produces from a fully populated cache; a mirror off by a row would not show up in
      // the magnitudes, only here.
      assertThat(throughMirror).containsExactly(direct);
    }
  }

  @Test
  void aPrefillChunkIsRefusedBecauseItWouldNeedASecondMirrorPerLayer() {
    try (Harness harness = new Harness()) {
      harness.kernel.selectScope(SEQUENCE_ONE);

      assertThat(harness.eligible(0, 0, 8, 0)).isFalse();

      assertThat(harness.kernel.lastRefusal())
          .isEqualTo(TornadoCausalAttentionKernel.REFUSAL_PREFILL_CHUNK);
      assertThat(harness.kernel.routing().devicePrefillLayerChunks()).isZero();
    }
  }

  @Test
  void aDecodeStepAfterAJavaPathPrefillBackfillsTheWholeWindowOnce() {
    try (Harness harness = new Harness()) {
      harness.mirrorAndAttend(SEQUENCE_ONE, 0, 8, 1);
      long afterFirst = harness.kernel.routing().mirroredPositions();

      harness.mirrorAndAttend(SEQUENCE_ONE, 0, 9, 1);

      // The first decode paid for the prefill history; the second continues from the row the
      // first one stored and backfills nothing.
      assertThat(afterFirst).isEqualTo(8);
      assertThat(harness.kernel.routing().mirroredPositions()).isEqualTo(8);
      assertThat(harness.kernel.mirroredPosition(0)).isEqualTo(10);
    }
  }

  @Test
  void everyPlanIsASingleTokenShapeSoALayerHoldsExactlyOneMirror() {
    try (Harness harness = new Harness()) {
      harness.mirrorAndAttend(SEQUENCE_ONE, 0, 4, 1);
      harness.mirrorAndAttend(SEQUENCE_ONE, 0, 5, 1);
      harness.mirrorAndAttend(SEQUENCE_ONE, 1, 5, 1);

      assertThat(harness.plans).hasSize(2);
      assertThat(harness.plans).allMatch(plan -> plan.shape.executionBatchSize() == 1);
    }
  }

  @Test
  void switchingSequenceDiscardsTheMirrorAndCountsTheRebuild() {
    try (Harness harness = new Harness()) {
      harness.mirrorAndAttend(SEQUENCE_ONE, 0, 10, 1);
      assertThat(harness.kernel.routing().mirrorRebuilds()).isZero();

      harness.kernel.selectScope(SEQUENCE_TWO);

      assertThat(harness.kernel.mirroredPosition(0)).isZero();
      assertThat(harness.kernel.routing().mirrorRebuilds()).isEqualTo(1);
    }
  }

  @Test
  void twoBranchesOfOneSharedPrefixEachRebuildTheirOwnMirror() {
    // The host cache shares the prefix arrays between forks by identity. The device mirror does
    // not: this is the cost the design accepts, and it is counted rather than hidden.
    AttentionScope branchOne = new AttentionScope(11L, 16, SCALE, false);
    AttentionScope branchTwo = new AttentionScope(12L, 16, SCALE, false);
    try (Harness harness = new Harness()) {
      harness.mirrorAndAttend(branchOne, 0, 16, 1);
      harness.mirrorAndAttend(branchTwo, 0, 16, 1);
      harness.mirrorAndAttend(branchOne, 0, 17, 1);

      TornadoAttentionRouting routing = harness.kernel.routing();
      assertThat(routing.mirrorRebuilds()).isEqualTo(2);
      assertThat(routing.mirroredPositions()).isEqualTo(16L + 16L + 17L);
      assertThat(routing.deviceDecodeLayerSteps()).isEqualTo(3);
    }
  }

  @Test
  void slidingWindowAttentionIsRefusedWithItsReason() {
    try (Harness harness = new Harness()) {
      harness.kernel.selectScope(SEQUENCE_ONE);

      assertThat(harness.eligible(0, 4, 1, 16)).isFalse();

      assertThat(harness.kernel.lastRefusal())
          .isEqualTo(TornadoCausalAttentionKernel.REFUSAL_SLIDING_WINDOW);
      assertThat(harness.kernel.routing().refusalReasons())
          .containsEntry(TornadoCausalAttentionKernel.REFUSAL_SLIDING_WINDOW, 1L);
      assertThat(harness.kernel.routing().ranOnDevice()).isFalse();
    }
  }

  @Test
  void fusedGroupedArithmeticIsRefusedBecauseItsJavaReferenceIsADifferentKernel() {
    try (Harness harness = new Harness()) {
      harness.kernel.selectScope(new AttentionScope(3L, 0, 0.0078125f, true));

      assertThat(harness.eligible(0, 4, 1, 0)).isFalse();
      assertThat(harness.kernel.lastRefusal())
          .isEqualTo(TornadoCausalAttentionKernel.REFUSAL_FUSED_ARITHMETIC);
    }
  }

  @Test
  void anUnselectedScopeIsRefusedRatherThanAssumed() {
    try (Harness harness = new Harness()) {
      assertThat(harness.eligible(0, 4, 1, 0)).isFalse();
      assertThat(harness.kernel.lastRefusal())
          .isEqualTo(TornadoCausalAttentionKernel.REFUSAL_NO_SCOPE);
    }
  }

  @Test
  void aNonPositiveBatchIsRefused() {
    try (Harness harness = new Harness()) {
      harness.kernel.selectScope(SEQUENCE_ONE);

      assertThat(harness.eligible(0, 0, 0, 0)).isFalse();
      assertThat(harness.kernel.lastRefusal())
          .isEqualTo(TornadoCausalAttentionKernel.REFUSAL_BATCH_SHAPE);
    }
  }

  @Test
  void aChunkPastTheCompiledContextIsRefused() {
    try (Harness harness = new Harness()) {
      harness.kernel.selectScope(SEQUENCE_ONE);

      assertThat(harness.eligible(0, MAX_SEQUENCE, 1, 0)).isFalse();
      assertThat(harness.kernel.lastRefusal())
          .isEqualTo(TornadoCausalAttentionKernel.REFUSAL_CONTEXT);
    }
  }

  @Test
  void aMirrorAheadOfTheRequestedPositionIsRefusedRatherThanSilentlyReused() {
    try (Harness harness = new Harness()) {
      harness.mirrorAndAttend(SEQUENCE_ONE, 0, 10, 1);
      harness.kernel.selectScope(SEQUENCE_ONE);

      assertThat(harness.eligible(0, 5, 1, 0)).isFalse();
      assertThat(harness.kernel.lastRefusal())
          .isEqualTo(TornadoCausalAttentionKernel.REFUSAL_MIRROR_AHEAD);
    }
  }

  @Test
  void aRewindLetsSpeculativePositionsBeReattended() {
    try (Harness harness = new Harness()) {
      harness.mirrorAndAttend(SEQUENCE_ONE, 0, 10, 1);

      harness.kernel.rewind(6);

      assertThat(harness.kernel.mirroredPosition(0)).isEqualTo(6);
      harness.kernel.selectScope(SEQUENCE_ONE);
      assertThat(harness.eligible(0, 6, 1, 0)).isTrue();
    }
  }

  @Test
  void aResetClearsContinuityAndTheSelectedScope() {
    try (Harness harness = new Harness()) {
      harness.mirrorAndAttend(SEQUENCE_ONE, 0, 10, 1);

      harness.kernel.reset();

      assertThat(harness.kernel.mirroredPosition(0)).isZero();
      assertThat(harness.eligible(0, 0, 1, 0)).isFalse();
      assertThat(harness.kernel.lastRefusal())
          .isEqualTo(TornadoCausalAttentionKernel.REFUSAL_NO_SCOPE);
    }
  }

  @Test
  void aClosedKernelRefusesEveryStep() {
    Harness harness = new Harness();
    harness.mirrorAndAttend(SEQUENCE_ONE, 0, 4, 1);
    harness.close();

    harness.kernel.selectScope(SEQUENCE_ONE);
    assertThat(harness.eligible(0, 0, 1, 0)).isFalse();
    assertThat(harness.kernel.lastRefusal()).isEqualTo(TornadoCausalAttentionKernel.REFUSAL_CLOSED);
    assertThat(harness.plans).allMatch(plan -> plan.closed);
  }

  @Test
  void aMirrorThatWouldExceedTheDeviceBudgetIsRefusedWithItsReason() {
    // One layer's plan fits; the budget then has nothing left for a second layer.
    TornadoAttentionShape oneLayer =
        new TornadoAttentionShape(
            1, 64, HEADS, KV_HEADS, HEAD_LENGTH, HEAD_LENGTH, MAX_SEQUENCE, 0, SCALE);
    try (Harness harness = new Harness(oneLayer.deviceBytes(1))) {
      harness.mirrorAndAttend(SEQUENCE_ONE, 0, 4, 1);
      harness.kernel.selectScope(SEQUENCE_ONE);

      assertThat(harness.eligible(1, 0, 1, 0)).isFalse();
      assertThat(harness.kernel.lastRefusal())
          .isEqualTo(TornadoCausalAttentionKernel.REFUSAL_BUDGET);
      assertThat(harness.kernel.committedDeviceBytes()).isEqualTo(oneLayer.deviceBytes(1));
    }
  }

  @Test
  void attendingWithoutAnAcceptedEligibilityCheckIsRejected() {
    try (Harness harness = new Harness()) {
      harness.kernel.selectScope(SEQUENCE_ONE);

      assertThatThrownBy(
              () ->
                  harness.kernel.attend(
                      new float[HEADS * HEAD_LENGTH],
                      new float[HEADS * HEAD_LENGTH],
                      new float[KV_HEADS * HEAD_LENGTH],
                      new float[KV_HEADS * HEAD_LENGTH],
                      0,
                      0,
                      1,
                      HEADS,
                      KV_HEADS,
                      HEAD_LENGTH,
                      HEAD_LENGTH,
                      MAX_SEQUENCE,
                      0))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("was not accepted");
    }
  }

  @Test
  void aMirrorSpanThatDoesNotContinueTheWindowIsRejected() {
    try (Harness harness = new Harness()) {
      harness.kernel.selectScope(SEQUENCE_ONE);
      assertThat(harness.eligible(0, 5, 1, 0)).isTrue();

      assertThatThrownBy(
              () ->
                  harness.kernel.mirrorSpan(
                      0,
                      2,
                      1,
                      new float[KV_HEADS * HEAD_LENGTH],
                      0,
                      KV_HEADS * HEAD_LENGTH,
                      new float[KV_HEADS * HEAD_LENGTH],
                      0,
                      KV_HEADS * HEAD_LENGTH))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("must continue at 0");
    }
  }

  @Test
  void theNoOpKernelReportsThatItIsNotConfigured() {
    BatchedCausalAttentionKernel none = BatchedCausalAttentionKernel.none();

    assertThat(none.isEligible(0, 0, 1, 8, 2, 32, 32, 64, 0)).isFalse();
    assertThat(none.lastRefusal()).contains("no batched causal-attention kernel");
    assertThat(none.mirroredPosition(0)).isEqualTo(-1);
    assertThat(TornadoAttentionRouting.none().ranOnDevice()).isFalse();
  }

  /** Drives the kernel the way {@code LlamaForwardPass} does: select, check, mirror, attend. */
  private static final class Harness implements AutoCloseable {
    private final TornadoCausalAttentionKernel kernel;
    private final List<HostPlan> plans = new ArrayList<>();
    private final float[] keys = new float[MAX_SEQUENCE * KV_HEADS * HEAD_LENGTH];
    private final float[] values = new float[MAX_SEQUENCE * KV_HEADS * HEAD_LENGTH];
    private final float[] queries = new float[MAX_SEQUENCE * HEADS * HEAD_LENGTH];

    Harness() {
      this(Long.MAX_VALUE);
    }

    Harness(long deviceByteBudget) {
      this.kernel =
          new TornadoCausalAttentionKernel(
              deviceByteBudget,
              64,
              (name, shape) -> {
                HostPlan plan = new HostPlan(shape);
                plans.add(plan);
                return plan;
              });
      Random random = new Random(19L);
      fill(random, keys);
      fill(random, values);
      fill(random, queries);
    }

    private static void fill(Random random, float[] target) {
      for (int index = 0; index < target.length; index++) {
        target[index] = random.nextFloat(-1.0f, 1.0f);
      }
    }

    boolean eligible(int layer, int startPosition, int batchSize, int slidingWindow) {
      return kernel.isEligible(
          layer,
          startPosition,
          batchSize,
          HEADS,
          KV_HEADS,
          HEAD_LENGTH,
          HEAD_LENGTH,
          MAX_SEQUENCE,
          slidingWindow);
    }

    /** Backfills the window the Java path would already hold, then attends the chunk. */
    float[] mirrorAndAttend(AttentionScope scope, int layer, int startPosition, int batchSize) {
      kernel.selectScope(scope);
      assertThat(eligible(layer, startPosition, batchSize, 0)).isTrue();
      int keyDim = KV_HEADS * HEAD_LENGTH;
      int mirrored = kernel.mirroredPosition(layer);
      if (mirrored < startPosition) {
        kernel.mirrorSpan(
            layer,
            mirrored,
            startPosition - mirrored,
            keys,
            mirrored * keyDim,
            keyDim,
            values,
            mirrored * keyDim,
            keyDim);
      }
      float[] output = new float[batchSize * HEADS * HEAD_LENGTH];
      kernel.attend(
          output,
          Arrays.copyOfRange(
              queries,
              startPosition * HEADS * HEAD_LENGTH,
              (startPosition + batchSize) * HEADS * HEAD_LENGTH),
          Arrays.copyOfRange(keys, startPosition * keyDim, (startPosition + batchSize) * keyDim),
          Arrays.copyOfRange(values, startPosition * keyDim, (startPosition + batchSize) * keyDim),
          layer,
          startPosition,
          batchSize,
          HEADS,
          KV_HEADS,
          HEAD_LENGTH,
          HEAD_LENGTH,
          MAX_SEQUENCE,
          0);
      return output;
    }

    /** The same decode step computed straight from a fully populated cache. */
    float[] directKernelResult(int position) {
      FloatArray output = new FloatArray(HEADS * HEAD_LENGTH);
      TornadoAttentionKernel.attend(
          IntArray.fromArray(new int[] {0, 0, position, 1}),
          FloatArray.fromArray(
              Arrays.copyOfRange(
                  queries, position * HEADS * HEAD_LENGTH, (position + 1) * HEADS * HEAD_LENGTH)),
          FloatArray.fromArray(keys),
          FloatArray.fromArray(values),
          new FloatArray(HEADS * MAX_SEQUENCE),
          output,
          HEADS,
          HEADS / KV_HEADS,
          HEAD_LENGTH,
          HEAD_LENGTH,
          KV_HEADS * HEAD_LENGTH,
          KV_HEADS * HEAD_LENGTH,
          MAX_SEQUENCE,
          0,
          SCALE);
      return output.toHeapArray();
    }

    @Override
    public void close() {
      kernel.close();
    }
  }

  /**
   * The buffers a {@link TornadoAttentionPlan} holds, driven by the same kernel methods as ordinary
   * Java. Device scheduling and PTX generation are what this deliberately does not stand in for.
   */
  private static final class HostPlan implements AttentionPlan {
    private final TornadoAttentionShape shape;
    private final FloatArray keyMirror;
    private final FloatArray valueMirror;
    private final FloatArray scores;
    private final FloatArray stagedKey;
    private final FloatArray stagedValue;
    private final FloatArray stagedQuery;
    private final FloatArray output;
    private final IntArray state;
    private boolean closed;

    HostPlan(TornadoAttentionShape shape) {
      this.shape = shape;
      this.keyMirror = new FloatArray(shape.maxSequenceLength() * shape.keyDim());
      this.valueMirror = new FloatArray(shape.maxSequenceLength() * shape.valueDim());
      this.scores =
          new FloatArray(shape.executionBatchSize() * shape.numHeads() * shape.maxSequenceLength());
      this.stagedKey = new FloatArray(shape.stagingRows() * shape.keyDim());
      this.stagedValue = new FloatArray(shape.stagingRows() * shape.valueDim());
      this.stagedQuery = new FloatArray(shape.executionBatchSize() * shape.queryDim());
      this.output = new FloatArray(shape.executionBatchSize() * shape.outputDim());
      this.state = new IntArray(TornadoAttentionKernel.STATE_LENGTH);
      TornadoAttentionKernel.validate(
          stagedQuery,
          output,
          scores,
          shape.executionBatchSize(),
          shape.numHeads(),
          shape.numKvHeads(),
          shape.keyLength(),
          shape.valueLength(),
          shape.maxSequenceLength());
    }

    @Override
    public int mirrorChunkPositions() {
      return shape.stagingRows();
    }

    @Override
    public void mirror(
        int firstPosition,
        int positionCount,
        float[] keys,
        int keyOffset,
        int keyRowStride,
        float[] values,
        int valueOffset,
        int valueRowStride) {
      stagedKey.clear();
      stagedValue.clear();
      for (int row = 0; row < positionCount; row++) {
        for (int index = 0; index < shape.keyDim(); index++) {
          stagedKey.set(row * shape.keyDim() + index, keys[keyOffset + row * keyRowStride + index]);
        }
        for (int index = 0; index < shape.valueDim(); index++) {
          stagedValue.set(
              row * shape.valueDim() + index, values[valueOffset + row * valueRowStride + index]);
        }
      }
      run(firstPosition, positionCount, 0, 0);
    }

    @Override
    public void attend(
        float[] outputRows,
        float[] query,
        float[] key,
        float[] value,
        int startPosition,
        int batchSize) {
      stagedKey.clear();
      stagedValue.clear();
      stagedQuery.clear();
      copy(key, stagedKey, batchSize * shape.keyDim());
      copy(value, stagedValue, batchSize * shape.valueDim());
      copy(query, stagedQuery, batchSize * shape.queryDim());
      run(startPosition, batchSize, startPosition, batchSize);
      for (int index = 0; index < batchSize * shape.outputDim(); index++) {
        outputRows[index] = output.get(index);
      }
    }

    private static void copy(float[] source, FloatArray target, int entries) {
      for (int index = 0; index < entries; index++) {
        target.set(index, source[index]);
      }
    }

    private void run(int storePosition, int storeRows, int attendPosition, int attendRows) {
      state.set(TornadoAttentionKernel.STORE_POSITION_INDEX, storePosition);
      state.set(TornadoAttentionKernel.STORE_ROWS_INDEX, storeRows);
      state.set(TornadoAttentionKernel.ATTEND_POSITION_INDEX, attendPosition);
      state.set(TornadoAttentionKernel.ATTEND_ROWS_INDEX, attendRows);
      TornadoAttentionKernel.store(
          state, stagedKey, stagedValue, keyMirror, valueMirror, shape.keyDim(), shape.valueDim());
      TornadoAttentionKernel.attend(
          state,
          stagedQuery,
          keyMirror,
          valueMirror,
          scores,
          output,
          shape.numHeads(),
          shape.groupSize(),
          shape.keyLength(),
          shape.valueLength(),
          shape.keyDim(),
          shape.valueDim(),
          shape.maxSequenceLength(),
          shape.slidingWindow(),
          shape.scale());
    }

    @Override
    public void close() {
      closed = true;
    }
  }
}
