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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class EmbeddingEquivalenceTest {

  private static float[] unit(int dimension, int seed) {
    float[] vector = new float[dimension];
    for (int index = 0; index < dimension; index++) {
      vector[index] = (float) Math.sin(seed * 0.7 + index * 0.13);
    }
    double norm = 0;
    for (float value : vector) {
      norm += (double) value * value;
    }
    norm = Math.sqrt(norm);
    for (int index = 0; index < dimension; index++) {
      vector[index] /= (float) norm;
    }
    return vector;
  }

  private static float[] nudged(float[] source, float delta) {
    float[] copy = source.clone();
    copy[0] += delta;
    return copy;
  }

  private static float[] scaled(float[] source, float factor) {
    float[] copy = source.clone();
    for (int index = 0; index < copy.length; index++) {
      copy[index] *= factor;
    }
    return copy;
  }

  private static EmbeddingEquivalence.Result compare(List<float[]> left, List<float[]> right) {
    return EmbeddingEquivalence.compare(left, right, true);
  }

  @Nested
  static class Comparison {

    @Test
    void reportsPerfectAgreementForIdenticalVectors() {
      float[] vector = unit(64, 1);

      EmbeddingEquivalence.Result result = compare(List.of(vector), List.of(vector.clone()));

      assertThat(result.minimumCosine())
          .isCloseTo(1.0, org.assertj.core.data.Offset.offset(1.0e-9));
      assertThat(result.maxComponentDelta()).isZero();
      assertThat(result.passed()).isTrue();
    }

    @Test
    void takesTheWorstProbeAsTheGate() {
      // One bad probe must fail the run; averaging would let a broken case hide behind good ones.
      float[] good = unit(64, 1);
      float[] other = unit(64, 2);

      EmbeddingEquivalence.Result result =
          compare(List.of(good, other), List.of(good.clone(), unit(64, 9)));

      assertThat(result.minimumCosine()).isLessThan(result.meanCosine());
      assertThat(result.passed()).isFalse();
    }

    @Test
    void toleratesFloatingPointDivergenceBetweenImplementations() {
      // Independent implementations accumulate differently; a tiny nudge must still pass.
      float[] vector = unit(1024, 3);

      EmbeddingEquivalence.Result result =
          compare(List.of(vector), List.of(nudged(vector, 0.0005f)));

      assertThat(result.minimumCosine()).isGreaterThan(EmbeddingEquivalence.MINIMUM_COSINE);
      assertThat(result.passed()).isTrue();
    }

    @Test
    void failsWhenAgreementDropsBelowTheGate() {
      float[] vector = unit(64, 4);

      EmbeddingEquivalence.Result result = compare(List.of(vector), List.of(nudged(vector, 0.5f)));

      assertThat(result.minimumCosine()).isLessThan(EmbeddingEquivalence.MINIMUM_COSINE);
      assertThat(result.passed()).isFalse();
    }

    @Test
    void recordsTheLargestComponentDifference() {
      float[] vector = unit(64, 5);

      EmbeddingEquivalence.Result result = compare(List.of(vector), List.of(nudged(vector, 0.25f)));

      assertThat(result.maxComponentDelta())
          .isCloseTo(0.25, org.assertj.core.data.Offset.offset(1.0e-6));
    }
  }

  @Nested
  static class Normalization {

    @Test
    void cosineAloneCannotSeeAMissingNormalization() {
      // Documents why the length check exists: cosine is scale-invariant, so a runtime that skips
      // L2 normalization agrees with a normalized reference at exactly 1.0. Measured against
      // llama.cpp with --embd-normalize -1, which scored 1.000000 on every probe.
      float[] vector = unit(64, 6);

      EmbeddingEquivalence.Result result = compare(List.of(vector), List.of(scaled(vector, 12.0f)));

      assertThat(result.minimumCosine())
          .isCloseTo(1.0, org.assertj.core.data.Offset.offset(1.0e-9));
      assertThat(result.passed()).isFalse();
    }

    @Test
    void failsWhenProducedVectorsAreNotUnitLength() {
      float[] vector = unit(64, 7);

      EmbeddingEquivalence.Result result = compare(List.of(vector), List.of(scaled(vector, 12.0f)));

      assertThat(result.normalizationHeld()).isFalse();
      assertThat(result.maxNormDeviation())
          .isCloseTo(11.0, org.assertj.core.data.Offset.offset(1.0e-5));
    }

    @Test
    void toleratesFloatAccumulationInTheLengthCheck() {
      float[] vector = scaled(unit(1024, 8), 1.0f + 1.0e-5f);

      EmbeddingEquivalence.Result result = compare(List.of(unit(1024, 8)), List.of(vector));

      assertThat(result.maxNormDeviation()).isLessThan(EmbeddingEquivalence.MAX_NORM_DEVIATION);
      assertThat(result.passed()).isTrue();
    }

    @Test
    void skipsTheLengthCheckWhenTheReferenceIsNotNormalized() {
      // A reference generated without normalization says nothing about length, so requiring it
      // would fail runs that are correct.
      float[] vector = unit(64, 9);

      EmbeddingEquivalence.Result result =
          EmbeddingEquivalence.compare(List.of(vector), List.of(scaled(vector, 12.0f)), false);

      assertThat(result.normalizationHeld()).isTrue();
      assertThat(result.passed()).isTrue();
    }
  }

  @Nested
  static class Validation {

    @Test
    void rejectsMismatchedProbeCounts() {
      // A missing probe means the run did not compare what it claims to have compared.
      assertThatThrownBy(() -> compare(List.of(unit(8, 1)), List.of(unit(8, 1), unit(8, 2))))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("probe count");
    }

    @Test
    void rejectsMismatchedDimensions() {
      assertThatThrownBy(() -> compare(List.of(unit(8, 1)), List.of(unit(16, 1))))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("dimension");
    }

    @Test
    void rejectsAnEmptyComparison() {
      assertThatThrownBy(() -> compare(List.of(), List.of()))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Nested
  static class PinnedReference {

    @Test
    void loadsTheCommittedOracleReference() {
      EmbeddingEquivalence.Reference reference = EmbeddingEquivalence.loadReference();

      assertThat(reference.oracleBackend()).isEqualTo("llama.cpp");
      assertThat(reference.oracleVersion()).isNotBlank();
      assertThat(reference.pooling()).isEqualTo("last-token");
      assertThat(reference.normalized()).isTrue();
      assertThat(reference.embeddingDimension()).isEqualTo(1024);
      assertThat(reference.vectors()).hasSize(8);
      assertThat(reference.vectors().get(0)).hasSize(1024);
    }

    @Test
    void referenceMatchesTheProbeSetItWasGeneratedFrom() {
      // Guards the one way this gate can pass falsely: probes edited without regenerating the
      // reference would compare new text against stale vectors.
      EmbeddingEquivalence.Reference reference = EmbeddingEquivalence.loadReference();

      assertThat(reference.probeSetSha256())
          .isEqualTo(EmbeddingEquivalence.sha256(EmbeddingEquivalence.loadProbesRaw()));
    }

    @Test
    void listsEveryCommittedReference() {
      // The gate covers more than one model, so the reference set is an index rather than a
      // single pinned file.
      assertThat(EmbeddingEquivalence.referenceNames()).isNotEmpty();
      for (String name : EmbeddingEquivalence.referenceNames()) {
        assertThat(EmbeddingEquivalence.loadReference(name).vectors()).isNotEmpty();
      }
    }

    @Test
    void selectsTheReferenceGeneratedFromTheSameArtifact() {
      EmbeddingEquivalence.Reference reference = EmbeddingEquivalence.loadReference();

      assertThat(EmbeddingEquivalence.referenceFor(reference.artifactSha256()))
          .isPresent()
          .get()
          .extracting(EmbeddingEquivalence.Reference::artifactSha256)
          .isEqualTo(reference.artifactSha256());
    }

    @Test
    void hasNoReferenceForAnUnknownArtifact() {
      assertThat(EmbeddingEquivalence.referenceFor("f".repeat(64))).isEmpty();
    }

    @Test
    void everyReferenceCoversTheSameProbeSet() {
      String probeSet = EmbeddingEquivalence.sha256(EmbeddingEquivalence.loadProbesRaw());
      for (String name : EmbeddingEquivalence.referenceNames()) {
        assertThat(EmbeddingEquivalence.loadReference(name).probeSetSha256()).isEqualTo(probeSet);
      }
    }

    @Test
    void probeCountMatchesTheReferenceVectorCount() {
      assertThat(EmbeddingEquivalence.loadProbes())
          .hasSameSizeAs(EmbeddingEquivalence.loadReference().vectors());
    }
  }
}
