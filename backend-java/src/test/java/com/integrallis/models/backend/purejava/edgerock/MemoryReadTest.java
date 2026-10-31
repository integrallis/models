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
package com.integrallis.models.backend.purejava.edgerock;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class MemoryReadTest {

  private static final int DIM = 4;
  private static final float TOLERANCE = 1.0e-6f;

  @Test
  void theZeroSinkHalvesTheWeightOfASingleTiedRecord() {
    // The decisive test for the sink. A zero query makes the record's logit exactly 0, so the
    // softmax runs over {record: exp(0), sink: exp(0)} and the record takes half the weight.
    // Without the sink the record would take all of it and this would read 1.0 instead of 0.5.
    MemoryBank bank = bankWith(value(1f));
    float[][] queries = {new float[DIM]};
    float[][] out = {new float[DIM]};

    MemoryRead.read(bank, queries, new float[] {1f}, 1, out);

    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(0.5f);
  }

  @Test
  void twoTiedRecordsShareTheWeightWithTheSink() {
    // Three equal unnormalised weights -- two records and the sink -- so each record gets a third.
    MemoryBank bank = new MemoryBank(1, 4, 1, DIM);
    bank.stage(new float[][] {new float[DIM]}, new float[][] {value(3f)}, new float[] {0.9f});
    bank.stage(new float[][] {new float[DIM]}, new float[][] {value(6f)}, new float[] {0.8f});
    float[][] out = {new float[DIM]};

    MemoryRead.read(bank, new float[][] {new float[DIM]}, new float[] {1f}, 1, out);

    // (3 + 6) / 3 = 3
    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(3f);
  }

  @Test
  void anEmptyBankReadsZeroBecauseTheSinkTakesEverything() {
    // Every sequence's first chunk is this case, so it is asserted rather than assumed.
    MemoryBank bank = new MemoryBank(1, 4, 2, DIM);
    float[][] out = {value(99f)};

    MemoryRead.read(bank, new float[][] {new float[DIM]}, new float[] {1f}, 1, out);

    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(0f);
  }

  @Test
  void theGateScalesTheReadAndCanInvertIt() {
    MemoryBank bank = bankWith(value(1f));
    float[][] out = {new float[DIM]};

    MemoryRead.read(bank, new float[][] {new float[DIM]}, new float[] {-2f}, 1, out);

    // Half the weight, gate -2: the layer subtracts what it recalls.
    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(-1f);
  }

  @Test
  void anUntrainedGateIsExactlyZero() {
    // read_scale is zero-initialised in the reference, and tanh(0) = 0, so a freshly initialised
    // memory layer contributes nothing at all rather than noise.
    float[][] queries = {value(1f)};
    float[][] gateWeights = {value(1f)};
    float[] gates = new float[1];

    MemoryRead.gates(queries, gateWeights, new float[] {0.5f}, new float[] {0f}, gates);

    assertThat(gates[0]).isZero();
  }

  @Test
  void theGateIsBoundedByTwoInMagnitude() {
    float[][] queries = {value(1000f)};
    float[][] gateWeights = {value(1000f)};
    float[] gates = new float[1];

    MemoryRead.gates(queries, gateWeights, new float[] {0f}, new float[] {50f}, gates);

    // tanh(50) ~ 1, sigmoid(huge) ~ 1, so the gate approaches 2 without exceeding it.
    assertThat(gates[0]).isLessThanOrEqualTo(2f).isGreaterThan(1.99f);
  }

  @Test
  void headsInAGroupShareKeysButKeepTheirOwnGate() {
    MemoryBank bank = bankWith(value(1f));
    float[][] queries = {new float[DIM], new float[DIM]};
    float[][] out = {new float[DIM], new float[DIM]};

    MemoryRead.read(bank, queries, new float[] {1f, 0.5f}, 2, out);

    assertThat(out[0]).usingComparatorWithPrecision(TOLERANCE).containsOnly(0.5f);
    assertThat(out[1]).usingComparatorWithPrecision(TOLERANCE).containsOnly(0.25f);
  }

  /** A one-group bank holding exactly one committed record with the given value. */
  private static MemoryBank bankWith(float[] recordValue) {
    MemoryBank bank = new MemoryBank(1, 4, 1, DIM);
    bank.stage(new float[][] {new float[DIM]}, new float[][] {recordValue}, new float[] {0.9f});
    return bank;
  }

  private static float[] value(float fill) {
    float[] out = new float[DIM];
    java.util.Arrays.fill(out, fill);
    return out;
  }
}
