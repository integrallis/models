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

import com.integrallis.models.backend.purejava.cache.KvCache;
import java.util.Random;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Off-device parity for the Java-authored attention kernel.
 *
 * <p>Every test here executes the kernel methods as ordinary Java, which is what a
 * {@code @Parallel} loop degrades to when it is called directly rather than handed to a TornadoVM
 * task graph. That is the same approach {@link Q4ProjectionKernelTest} takes for the projection
 * kernel: it exercises the arithmetic, the indexing, the grouped-query mapping, and the causal
 * bound on every CI host, and it leaves only PTX code generation and device scheduling to the GPU
 * gate.
 *
 * <p>Two contracts are checked, deliberately at different strengths.
 *
 * <ul>
 *   <li><b>Bit-exact</b> against a scalar reference that performs the same operations in the same
 *       order. This is achievable and is therefore required: it is what proves the kernel's
 *       indexing and masking rather than its rounding.
 *   <li><b>Bounded</b> against the production Java attention path, which scores through the Panama
 *       {@code VectorUtil} dot product with a vector-tree reduction. A scalar kernel cannot
 *       reproduce that summation order bit for bit on any device, so the contract is a stated
 *       numeric tolerance ({@link #RELATIVE_L2_CONTRACT}) rather than equality, and the tolerance
 *       is tested.
 * </ul>
 */
class TornadoAttentionKernelTest {

  /**
   * The stated numeric contract against the production Java attention path.
   *
   * <p>Same value the Q4 projection kernel is held to, and the same value the standing device
   * attention experiment gates on. It is a relative L2 over one attention output block.
   */
  private static final double RELATIVE_L2_CONTRACT = 2.0e-5;

  @Test
  void decodeMatchesAScalarReferenceBitForBit() {
    Shape shape = new Shape(4, 2, 32, 64, 0);
    Fixture fixture = Fixture.random(shape, 23, 37L);

    float[] actual = fixture.runKernel(23);
    float[] expected = fixture.scalarReference(23);

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void decodeMatchesTheProductionJavaAttentionWithinTheStatedContract() {
    Shape shape = new Shape(8, 2, 64, 128, 0);
    Fixture fixture = Fixture.random(shape, 61, 101L);

    float[] actual = fixture.runKernel(61);
    float[] expected = fixture.productionJavaReference(61);

    assertThat(relativeL2(actual, expected)).isLessThan(RELATIVE_L2_CONTRACT);
  }

  @Test
  void decodeAtTheFirstPositionAttendsOnlyToItself() {
    Shape shape = new Shape(4, 4, 4, 32, 0);
    Fixture fixture = Fixture.random(shape, 0, 11L);

    assertThat(fixture.runKernel(0)).containsExactly(fixture.scalarReference(0));
  }

  @Test
  void groupedQueryHeadsShareTheirKeyValueHeadRatherThanTheirOwnIndex() {
    // numHeads == numKvHeads * groupSize with groupSize > 1 is the case a wrong kvHead mapping
    // still produces plausible-looking output for, so it is checked on its own.
    Shape shape = new Shape(12, 3, 16, 32, 0);
    Fixture fixture = Fixture.random(shape, 9, 5L);

    assertThat(fixture.runKernel(9)).containsExactly(fixture.scalarReference(9));
  }

  @Test
  void multiHeadAttentionWithoutGroupingIsSupported() {
    Shape shape = new Shape(4, 4, 8, 16, 0);
    Fixture fixture = Fixture.random(shape, 6, 17L);

    assertThat(fixture.runKernel(6)).containsExactly(fixture.scalarReference(6));
  }

  @Test
  void slidingWindowBoundsTheAttendedPositionsExactlyAsTheJavaPathDoes() {
    int window = 5;
    Shape shape = new Shape(4, 2, 32, 32, window);
    Fixture fixture = Fixture.random(shape, 19, 71L);

    float[] actual = fixture.runKernel(19);

    assertThat(actual).containsExactly(fixture.scalarReference(19));
    // The window has to actually bind at this position, or the test proves nothing about it.
    assertThat(fixture.scalarReference(19)).isNotEqualTo(fixture.unwindowedScalarReference(19));
  }

  @Test
  void prefillComputesEveryRowOfAChunkAgainstItsOwnCausalPrefix() {
    Shape shape = new Shape(4, 2, 16, 32, 0);
    Fixture fixture = Fixture.random(shape, 0, 3L);

    float[] actual = fixture.runPrefill(0, 6);

    for (int row = 0; row < 6; row++) {
      float[] expectedRow = fixture.scalarReference(row);
      for (int index = 0; index < shape.outputDim(); index++) {
        assertThat(actual[row * shape.outputDim() + index])
            .as("row %d index %d", row, index)
            .isEqualTo(expectedRow[index]);
      }
    }
  }

  @Test
  void paddedBatchRowsAreZeroedRatherThanLeftStale() {
    Shape shape = new Shape(4, 2, 16, 32, 0);
    Fixture fixture = Fixture.random(shape, 0, 13L);

    float[] actual = fixture.runPrefillPadded(0, 3, 8);

    for (int row = 3; row < 8; row++) {
      for (int index = 0; index < shape.outputDim(); index++) {
        assertThat(actual[row * shape.outputDim() + index]).isZero();
      }
    }
  }

  @Test
  void attendingZeroRowsClearsTheOutputBlockRatherThanLeavingIt() {
    Shape shape = new Shape(4, 2, 16, 32, 0);
    Fixture fixture = Fixture.random(shape, 5, 29L);

    // A backfill execution sets attendRows to zero; the output block it shares must not be read
    // afterwards as though it held a result.
    assertThat(fixture.runPrefillPadded(0, 0, 4)).containsOnly(0.0f);
  }

  @Test
  void storeWritesTheChunkIntoPositionMajorMirrorSlots() {
    Shape shape = new Shape(2, 1, 8, 4, 0);
    int batch = 3;
    int startPosition = 2;
    FloatArray key = FloatArray.fromArray(ramp(batch * shape.keyDim(), 1.0f));
    FloatArray value = FloatArray.fromArray(ramp(batch * shape.valueDim(), 100.0f));
    FloatArray keyCache = new FloatArray(shape.maxSequenceLength() * shape.keyDim());
    FloatArray valueCache = new FloatArray(shape.maxSequenceLength() * shape.valueDim());
    IntArray state = IntArray.fromArray(new int[] {startPosition, batch, 0, 0});

    TornadoAttentionKernel.store(
        state, key, value, keyCache, valueCache, shape.keyDim(), shape.valueDim());

    for (int row = 0; row < batch; row++) {
      for (int index = 0; index < shape.keyDim(); index++) {
        assertThat(keyCache.get((startPosition + row) * shape.keyDim() + index))
            .isEqualTo(key.get(row * shape.keyDim() + index));
      }
      for (int index = 0; index < shape.valueDim(); index++) {
        assertThat(valueCache.get((startPosition + row) * shape.valueDim() + index))
            .isEqualTo(value.get(row * shape.valueDim() + index));
      }
    }
    // Position 0 and 1 belong to the already-mirrored prefix and must not be touched.
    assertThat(keyCache.get(0)).isZero();
    assertThat(keyCache.get(shape.keyDim())).isZero();
  }

  @Test
  void storeIgnoresRowsBeyondTheActualBatch() {
    Shape shape = new Shape(2, 1, 8, 4, 0);
    FloatArray key = FloatArray.fromArray(ramp(4 * shape.keyDim(), 1.0f));
    FloatArray value = FloatArray.fromArray(ramp(4 * shape.valueDim(), 100.0f));
    FloatArray keyCache = new FloatArray(shape.maxSequenceLength() * shape.keyDim());
    FloatArray valueCache = new FloatArray(shape.maxSequenceLength() * shape.valueDim());

    TornadoAttentionKernel.store(
        IntArray.fromArray(new int[] {0, 2, 0, 0}),
        key,
        value,
        keyCache,
        valueCache,
        shape.keyDim(),
        shape.valueDim());

    assertThat(keyCache.get(2 * shape.keyDim())).isZero();
    assertThat(valueCache.get(2 * shape.valueDim())).isZero();
  }

  @Test
  void validateRejectsStorageThatDoesNotMatchTheShape() {
    assertThatThrownBy(
            () ->
                TornadoAttentionKernel.validate(
                    new FloatArray(8), new FloatArray(8), new FloatArray(4), 1, 4, 2, 2, 2, 8))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("scores");
  }

  @Test
  void validateRejectsAHeadCountThatIsNotAMultipleOfTheKeyValueHeadCount() {
    assertThatThrownBy(
            () ->
                TornadoAttentionKernel.validate(
                    new FloatArray(6), new FloatArray(6), new FloatArray(24), 1, 3, 2, 2, 2, 8))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("numKvHeads");
  }

  private static double relativeL2(float[] actual, float[] expected) {
    assertThat(actual).hasSameSizeAs(expected);
    double difference = 0.0;
    double magnitude = 0.0;
    for (int index = 0; index < expected.length; index++) {
      double delta = (double) actual[index] - expected[index];
      difference += delta * delta;
      magnitude += (double) expected[index] * expected[index];
    }
    return magnitude == 0.0 ? Math.sqrt(difference) : Math.sqrt(difference / magnitude);
  }

  private static float[] ramp(int length, float base) {
    float[] values = new float[length];
    for (int index = 0; index < length; index++) {
      values[index] = base + index;
    }
    return values;
  }

  /** One attention geometry as the forward pass reports it. */
  private record Shape(
      int numHeads, int numKvHeads, int maxSequenceLength, int headLength, int slidingWindow) {
    int groupSize() {
      return numHeads / numKvHeads;
    }

    int queryDim() {
      return numHeads * headLength;
    }

    int keyDim() {
      return numKvHeads * headLength;
    }

    int valueDim() {
      return keyDim();
    }

    int outputDim() {
      return numHeads * headLength;
    }

    float scale() {
      return (float) (1.0 / Math.sqrt(headLength));
    }
  }

  /**
   * A populated mirror plus one query row per position, shared by the kernel and the references.
   */
  private record Fixture(Shape shape, float[] keyCache, float[] valueCache, float[] queries) {

    static Fixture random(Shape shape, int lastPosition, long seed) {
      Random random = new Random(seed);
      float[] keyCache = new float[shape.maxSequenceLength() * shape.keyDim()];
      float[] valueCache = new float[shape.maxSequenceLength() * shape.valueDim()];
      float[] queries = new float[shape.maxSequenceLength() * shape.queryDim()];
      for (int position = 0; position <= lastPosition; position++) {
        fill(random, keyCache, position * shape.keyDim(), shape.keyDim());
        fill(random, valueCache, position * shape.valueDim(), shape.valueDim());
        fill(random, queries, position * shape.queryDim(), shape.queryDim());
      }
      return new Fixture(shape, keyCache, valueCache, queries);
    }

    private static void fill(Random random, float[] target, int offset, int length) {
      for (int index = 0; index < length; index++) {
        target[offset + index] = random.nextFloat(-1.5f, 1.5f);
      }
    }

    /** The kernel, executed off-device over a single-token decode plan shape. */
    float[] runKernel(int position) {
      return runPrefillPadded(position, 1, 1);
    }

    /** The kernel over a prefill chunk of {@code batchSize} rows starting at {@code start}. */
    float[] runPrefill(int start, int batchSize) {
      return runPrefillPadded(start, batchSize, batchSize);
    }

    float[] runPrefillPadded(int start, int batchSize, int executionBatchSize) {
      float[] query = new float[executionBatchSize * shape.queryDim()];
      System.arraycopy(queries, start * shape.queryDim(), query, 0, batchSize * shape.queryDim());
      FloatArray output = new FloatArray(executionBatchSize * shape.outputDim());
      FloatArray scores =
          new FloatArray(executionBatchSize * shape.numHeads() * shape.maxSequenceLength());
      TornadoAttentionKernel.attend(
          IntArray.fromArray(new int[] {0, 0, start, batchSize}),
          FloatArray.fromArray(query),
          FloatArray.fromArray(keyCache),
          FloatArray.fromArray(valueCache),
          scores,
          output,
          shape.numHeads(),
          shape.groupSize(),
          shape.headLength(),
          shape.headLength(),
          shape.keyDim(),
          shape.valueDim(),
          shape.maxSequenceLength(),
          shape.slidingWindow(),
          shape.scale());
      return output.toHeapArray();
    }

    /**
     * The same arithmetic in the same order, written plainly. Bit-exact agreement with the kernel
     * is required, so this reference is deliberately scalar and sequential.
     */
    float[] scalarReference(int position) {
      return scalarReference(position, shape.slidingWindow());
    }

    float[] unwindowedScalarReference(int position) {
      return scalarReference(position, 0);
    }

    private float[] scalarReference(int position, int slidingWindow) {
      float[] output = new float[shape.outputDim()];
      int firstPosition = slidingWindow > 0 ? Math.max(0, position - slidingWindow + 1) : 0;
      int count = position - firstPosition + 1;
      for (int head = 0; head < shape.numHeads(); head++) {
        int kvHead = head / shape.groupSize();
        int queryOffset = position * shape.queryDim() + head * shape.headLength();
        float[] weights = new float[count];
        float maximum = Float.NEGATIVE_INFINITY;
        for (int row = 0; row < count; row++) {
          int keyOffset = (firstPosition + row) * shape.keyDim() + kvHead * shape.headLength();
          float score = 0.0f;
          for (int column = 0; column < shape.headLength(); column++) {
            score += queries[queryOffset + column] * keyCache[keyOffset + column];
          }
          weights[row] = score * shape.scale();
          maximum = Math.max(maximum, weights[row]);
        }
        float sum = 0.0f;
        for (int row = 0; row < count; row++) {
          weights[row] = (float) Math.exp(weights[row] - maximum);
          sum += weights[row];
        }
        float inverseSum = 1.0f / sum;
        for (int row = 0; row < count; row++) {
          weights[row] *= inverseSum;
        }
        for (int row = 0; row < count; row++) {
          int valueOffset = (firstPosition + row) * shape.valueDim() + kvHead * shape.headLength();
          for (int column = 0; column < shape.headLength(); column++) {
            output[head * shape.headLength() + column] +=
                weights[row] * valueCache[valueOffset + column];
          }
        }
      }
      return output;
    }

    /**
     * The production Java attention path itself: a real {@link KvCache} scored and accumulated
     * through the same {@code VectorUtil} entry points {@code LlamaForwardPass} uses head by head.
     */
    float[] productionJavaReference(int position) {
      KvCache cache = new KvCache(1, shape.maxSequenceLength(), shape.keyDim(), shape.valueDim());
      for (int stored = 0; stored <= position; stored++) {
        cache.store(
            0, stored, keyCache, stored * shape.keyDim(), valueCache, stored * shape.valueDim());
      }
      float[] output = new float[shape.outputDim()];
      float[] scores = new float[shape.maxSequenceLength()];
      int firstPosition =
          shape.slidingWindow() > 0 ? Math.max(0, position - shape.slidingWindow() + 1) : 0;
      KvCache.AttentionView view = cache.attentionView(0, firstPosition, position + 1);
      for (int head = 0; head < shape.numHeads(); head++) {
        int kvHead = head / shape.groupSize();
        cache.writeAttentionScores(
            view,
            firstPosition,
            position + 1,
            kvHead * shape.headLength(),
            queries,
            position * shape.queryDim() + head * shape.headLength(),
            shape.headLength(),
            shape.scale(),
            scores,
            0,
            false);
        com.integrallis.models.backend.purejava.ops.TensorOps.softmax(
            scores, firstPosition, position - firstPosition + 1);
        cache.addAttentionValues(
            view,
            firstPosition,
            position + 1,
            kvHead * shape.headLength(),
            output,
            head * shape.headLength(),
            shape.headLength(),
            scores,
            0,
            false);
      }
      return output;
    }
  }

  @Test
  void theStatedContractIsTightEnoughToFailOnAWrongResult() {
    // A tolerance nothing can fail is not a contract. A single head's values swapped for another's
    // must exceed it.
    Shape shape = new Shape(8, 2, 64, 128, 0);
    Fixture fixture = Fixture.random(shape, 61, 101L);
    float[] expected = fixture.productionJavaReference(61);
    float[] perturbed = expected.clone();
    System.arraycopy(expected, 0, perturbed, shape.headLength(), shape.headLength());

    assertThat(relativeL2(perturbed, expected)).isGreaterThan(RELATIVE_L2_CONTRACT);
  }
}
