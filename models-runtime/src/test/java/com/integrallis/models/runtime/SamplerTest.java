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
package com.integrallis.models.runtime;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.SamplingOptions;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SamplerTest {

  @Nested
  class Constraints {

    @Test
    void greedySelectsBestAllowedToken() {
      float[] logits = {1.0f, 100.0f, 5.0f};
      Sampler sampler = new Sampler(SamplingOptions.builder().temperature(0.0f).build());

      assertThat(sampler.sample(logits, List.of(), token -> token != 1)).isEqualTo(2);
    }

    @Test
    void probabilisticSamplingOnlyReturnsAllowedTokens() {
      SamplingOptions options =
          SamplingOptions.builder().temperature(1.0f).topK(4).topP(1.0f).seed(42L).build();
      Sampler sampler = new Sampler(options);
      Set<Integer> sampled = new HashSet<>();

      for (int index = 0; index < 200; index++) {
        sampled.add(
            sampler.sample(new float[] {10.0f, 9.0f, 8.0f, 7.0f}, List.of(), token -> token >= 2));
      }

      assertThat(sampled).isSubsetOf(Set.of(2, 3));
    }

    @Test
    void rejectsAConstraintWithNoAllowedTokens() {
      Sampler sampler = new Sampler(SamplingOptions.builder().temperature(0.0f).build());

      assertThatThrownBy(() -> sampler.sample(new float[] {1.0f, 2.0f}, List.of(), token -> false))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("rejected every token");
    }
  }

  @Nested
  class Greedy {

    @Test
    void selectsArgmax() {
      float[] logits = {1.0f, 5.0f, 3.0f, 2.0f};
      Sampler sampler = new Sampler(SamplingOptions.builder().temperature(0.0f).build());
      assertThat(sampler.sample(logits, List.of())).isEqualTo(1);
    }

    @Test
    void selectsFirstOfTiedMaximum() {
      float[] logits = {3.0f, 3.0f, 1.0f};
      Sampler sampler = new Sampler(SamplingOptions.builder().temperature(0.0f).build());
      assertThat(sampler.sample(logits, List.of())).isEqualTo(0);
    }
  }

  @Nested
  class Temperature {

    @Test
    void highTempFlattensDistribution() {
      // With very high temperature, even low-probability tokens become likely
      float[] logits = {10.0f, 0.0f, 0.0f, 0.0f};
      SamplingOptions opts =
          SamplingOptions.builder().temperature(100.0f).topK(4).topP(1.0f).seed(42L).build();
      Sampler sampler = new Sampler(opts);

      Set<Integer> sampled = new HashSet<>();
      for (int i = 0; i < 100; i++) {
        sampled.add(sampler.sample(logits, List.of()));
      }
      // With high temperature, we should sample multiple different tokens
      assertThat(sampled.size()).isGreaterThan(1);
    }
  }

  @Nested
  class TopK {

    @Test
    void topK2KeepsOnlyTwo() {
      // logits: token 0 is much higher, but topK=2 keeps only top 2
      float[] logits = {10.0f, 5.0f, 1.0f, 0.1f};
      SamplingOptions opts =
          SamplingOptions.builder().temperature(1.0f).topK(2).topP(1.0f).seed(42L).build();
      Sampler sampler = new Sampler(opts);

      Set<Integer> sampled = new HashSet<>();
      for (int i = 0; i < 200; i++) {
        sampled.add(sampler.sample(logits, List.of()));
      }
      // Should only sample from top 2 tokens (indices 0 and 1)
      assertThat(sampled).isSubsetOf(Set.of(0, 1));
    }

    @Test
    void clampsTopKLargerThanTheVocabulary() {
      SamplingOptions options =
          SamplingOptions.builder().temperature(100.0f).topK(100).topP(1.0f).seed(42L).build();
      Sampler sampler = new Sampler(options);

      Set<Integer> sampled = new HashSet<>();
      for (int index = 0; index < 100; index++) {
        sampled.add(sampler.sample(new float[] {1.0f, 1.0f}, List.of()));
      }

      assertThat(sampled).containsExactlyInAnyOrder(0, 1);
    }
  }

  @Nested
  class TopP {

    @Test
    void topPKeepsMinimalSet() {
      // Token 0 has overwhelming probability after softmax
      float[] logits = {20.0f, 1.0f, 1.0f, 1.0f};
      SamplingOptions opts =
          SamplingOptions.builder().temperature(1.0f).topK(40).topP(0.5f).seed(42L).build();
      Sampler sampler = new Sampler(opts);

      Set<Integer> sampled = new HashSet<>();
      for (int i = 0; i < 100; i++) {
        sampled.add(sampler.sample(logits, List.of()));
      }
      // Token 0 should dominate with topP=0.5 since it has >50% probability
      assertThat(sampled).contains(0);
      assertThat(sampled.size()).isLessThanOrEqualTo(2);
    }

    @Test
    void includesTheTokenThatReachesTheTopPBoundary() {
      SamplingOptions options =
          SamplingOptions.builder().temperature(1.0f).topK(2).topP(0.5f).seed(42L).build();
      Sampler sampler = new Sampler(options);

      Set<Integer> sampled = new HashSet<>();
      for (int index = 0; index < 100; index++) {
        sampled.add(sampler.sample(new float[] {0.0f, 0.0f}, List.of()));
      }

      assertThat(sampled).containsExactly(0);
    }

    @Test
    void topPAppliesToTheRenormalizedTopKDistribution() {
      SamplingOptions options =
          SamplingOptions.builder().temperature(1.0f).topK(2).topP(0.5f).seed(42L).build();
      Sampler sampler = new Sampler(options);

      Set<Integer> sampled = new HashSet<>();
      for (int index = 0; index < 100; index++) {
        sampled.add(sampler.sample(new float[] {0.0f, 0.0f, 0.0f, 0.0f}, List.of()));
      }

      assertThat(sampled).containsExactly(0);
    }
  }

  @Nested
  class RepetitionPenalty {

    @Test
    void reducesRepeatedTokenProbability() {
      float[] logits = {5.0f, 5.0f, 5.0f, 5.0f};
      SamplingOptions opts =
          SamplingOptions.builder().temperature(0.0f).repetitionPenalty(2.0f).build();
      Sampler sampler = new Sampler(opts);

      // Penalize token 0 — after penalty it should be lower than others
      int result = sampler.sample(logits, List.of(0));
      assertThat(result).isNotEqualTo(0);
    }

    @Test
    void multipliesNegativeRepeatedLogitsByThePenalty() {
      SamplingOptions options =
          SamplingOptions.builder().temperature(0.0f).repetitionPenalty(2.0f).build();
      Sampler sampler = new Sampler(options);

      assertThat(sampler.sample(new float[] {-1.0f, -0.75f}, List.of(1))).isZero();
    }

    @Test
    void penalizesADuplicatedPositiveTokenOnlyOnce() {
      SamplingOptions options =
          SamplingOptions.builder().temperature(0.0f).repetitionPenalty(2.0f).build();
      Sampler sampler = new Sampler(options);

      assertThat(sampler.sample(new float[] {8.0f, 3.0f}, List.of(0, 0))).isZero();
    }

    @Test
    void penalizesADuplicatedNegativeTokenOnlyOnce() {
      SamplingOptions options =
          SamplingOptions.builder().temperature(0.0f).repetitionPenalty(2.0f).build();
      Sampler sampler = new Sampler(options);

      assertThat(sampler.sample(new float[] {-1.0f, -3.0f}, List.of(0, 0))).isZero();
    }

    @Test
    void ignoresOutOfRangeTokensWithoutChangingDuplicateHandling() {
      SamplingOptions options =
          SamplingOptions.builder().temperature(0.0f).repetitionPenalty(2.0f).build();
      Sampler sampler = new Sampler(options);

      assertThat(sampler.sample(new float[] {8.0f, 3.0f}, List.of(-1, 0, 2, 0))).isZero();
    }

    @Test
    void logitBatchMatchesArraySamplingWithPenaltyAndTemperature() {
      SamplingOptions options =
          SamplingOptions.builder()
              .temperature(0.7f)
              .topK(4)
              .topP(1.0f)
              .repetitionPenalty(1.5f)
              .seed(8675309L)
              .build();
      float[] row = {1.25f, -0.5f, 0.75f, 2.0f};
      LogitBatch batch =
          new LogitBatch(
              2, row.length, new float[] {9.0f, 0.0f, 0.0f, 0.0f, 1.25f, -0.5f, 0.75f, 2.0f});

      int fromArray = new Sampler(options).sample(row, List.of(0, 1));
      int fromBatch = new Sampler(options).sample(batch, 1, List.of(0, 1));

      assertThat(fromBatch).isEqualTo(fromArray);
    }
  }

  @Nested
  class Selection {

    @Test
    void topKBoundaryPrefersLowerTokenIdsAmongTies() {
      // All four tokens are equally likely; topK=2 must keep the two lowest ids.
      float[] logits = {1.0f, 1.0f, 1.0f, 1.0f};
      SamplingOptions options =
          SamplingOptions.builder().temperature(1.0f).topK(2).topP(1.0f).seed(42L).build();
      Sampler sampler = new Sampler(options);

      Set<Integer> sampled = new HashSet<>();
      for (int index = 0; index < 200; index++) {
        sampled.add(sampler.sample(logits, List.of()));
      }

      assertThat(sampled).isSubsetOf(Set.of(0, 1));
    }

    @Test
    void matchesFullSortReferenceAcrossRandomInputs() {
      Random generator = new Random(20260804L);
      for (int trial = 0; trial < 500; trial++) {
        int vocabulary = 2 + generator.nextInt(64);
        float[] logits = new float[vocabulary];
        for (int index = 0; index < vocabulary; index++) {
          // Coarse values so ties occur often and exercise tie-breaking.
          logits[index] = generator.nextInt(5) - 2.0f;
        }
        long seed = generator.nextLong();
        SamplingOptions options =
            SamplingOptions.builder()
                .temperature(0.1f + generator.nextFloat() * 2.0f)
                .topK(1 + generator.nextInt(vocabulary + 4))
                .topP(0.05f + generator.nextFloat() * 0.94f)
                .seed(seed)
                .build();

        int expected = fullSortReference(options, logits.clone(), new Random(seed));
        int actual = new Sampler(options).sample(logits.clone(), List.of());

        assertThat(actual).as("trial %d, vocabulary %d", trial, vocabulary).isEqualTo(expected);
      }
    }

    @Test
    void matchesFullSortReferenceWithMinPAcrossRandomInputs() {
      Random generator = new Random(20260916L);
      for (int trial = 0; trial < 500; trial++) {
        int vocabulary = 2 + generator.nextInt(64);
        float[] logits = new float[vocabulary];
        for (int index = 0; index < vocabulary; index++) {
          logits[index] = generator.nextInt(9) - 4.0f;
        }
        long seed = generator.nextLong();
        SamplingOptions options =
            SamplingOptions.builder()
                .temperature(0.1f + generator.nextFloat() * 2.0f)
                .topK(1 + generator.nextInt(vocabulary + 4))
                .topP(0.05f + generator.nextFloat() * 0.95f)
                .minP(generator.nextFloat())
                .seed(seed)
                .build();

        int expected = fullSortReference(options, logits.clone(), new Random(seed));
        int actual = new Sampler(options).sample(logits.clone(), List.of());

        assertThat(actual)
            .as("trial %d, vocabulary %d, minP %s", trial, vocabulary, options.minP())
            .isEqualTo(expected);
      }
    }

    /** Stable full-sort implementation used as an oracle for the bounded-heap fast path. */
    private static int fullSortReference(SamplingOptions options, float[] adjusted, Random rng) {
      for (int index = 0; index < adjusted.length; index++) {
        adjusted[index] /= options.temperature();
      }

      record TokenProb(int id, float prob) {}
      List<TokenProb> sorted = new ArrayList<>(adjusted.length);
      for (int index = 0; index < adjusted.length; index++) {
        sorted.add(new TokenProb(index, adjusted[index]));
      }
      sorted.sort(Comparator.comparingDouble(TokenProb::prob).reversed());

      int topK = Math.min(options.topK(), sorted.size());
      sorted = new ArrayList<>(sorted.subList(0, topK));

      float max = sorted.getFirst().prob();
      for (int index = 0; index < sorted.size(); index++) {
        TokenProb candidate = sorted.get(index);
        float probability = (float) Math.exp(candidate.prob() - max);
        sorted.set(index, new TokenProb(candidate.id(), probability));
      }
      if (options.minP() > 0) {
        // Standard min-p: keep p >= minP * p_max. Relative to p_max the survivors are the ones
        // whose unnormalized weight exp(logit/T - max) is at least minP.
        sorted =
            new ArrayList<>(sorted.stream().filter(tp -> tp.prob() >= options.minP()).toList());
      }
      float topKMass = 0;
      for (TokenProb tokenProb : sorted) {
        topKMass += tokenProb.prob();
      }

      float cumulative = 0;
      int cutoff = sorted.size();
      for (int index = 0; index < sorted.size(); index++) {
        cumulative += sorted.get(index).prob() / topKMass;
        if (cumulative >= options.topP()) {
          cutoff = index + 1;
          break;
        }
      }
      sorted = new ArrayList<>(sorted.subList(0, cutoff));

      float totalProb = 0;
      for (TokenProb tokenProb : sorted) {
        totalProb += tokenProb.prob();
      }

      float target = rng.nextFloat() * totalProb;
      float accumulated = 0;
      for (TokenProb tokenProb : sorted) {
        accumulated += tokenProb.prob();
        if (accumulated >= target) {
          return tokenProb.id();
        }
      }
      return sorted.getLast().id();
    }
  }

  /**
   * Min-p keeps tokens with {@code p >= minP * p_max}, measured on the temperature-scaled
   * distribution and applied before top-p. Every case uses hand-built logits whose probabilities
   * are exact enough that the surviving set is unambiguous.
   */
  @Nested
  class MinP {

    /** Logits whose softmax at temperature 1 is exactly {@code probabilities}. */
    private static float[] logitsFor(double... probabilities) {
      float[] logits = new float[probabilities.length];
      for (int index = 0; index < probabilities.length; index++) {
        logits[index] = (float) Math.log(probabilities[index]);
      }
      return logits;
    }

    private static Set<Integer> sampledSet(SamplingOptions options, float[] logits, int draws) {
      Sampler sampler = new Sampler(options);
      Set<Integer> sampled = new HashSet<>();
      for (int draw = 0; draw < draws; draw++) {
        sampled.add(sampler.sample(logits, List.of()));
      }
      return sampled;
    }

    private static SamplingOptions.Builder unrestricted() {
      return SamplingOptions.builder().temperature(1.0f).topK(1_000).topP(1.0f).seed(7L);
    }

    @Test
    void keepsExactlyTheTokensAtOrAboveTheScaledMaximum() {
      // p = [0.5, 0.3, 0.15, 0.05]; minP 0.2 => threshold 0.10 keeps {0, 1, 2}.
      float[] logits = logitsFor(0.5, 0.3, 0.15, 0.05);

      assertThat(sampledSet(unrestricted().minP(0.2f).build(), logits, 4_000))
          .containsExactlyInAnyOrder(0, 1, 2);
      // minP 0.7 => threshold 0.35 keeps only the top token.
      assertThat(sampledSet(unrestricted().minP(0.7f).build(), logits, 4_000)).containsExactly(0);
    }

    @Test
    void zeroMinPLeavesEveryTokenReachable() {
      float[] logits = logitsFor(0.5, 0.3, 0.15, 0.05);

      assertThat(sampledSet(unrestricted().build(), logits, 4_000))
          .containsExactlyInAnyOrder(0, 1, 2, 3);
    }

    @Test
    void thresholdIsMeasuredAfterTemperature() {
      // At T=1 token 1 has ratio 0.3/0.5 = 0.6 < 0.7 and is dropped. At T=4 the ratio becomes
      // 0.6^(1/4) ~= 0.88 >= 0.7, so a threshold measured after temperature keeps it.
      float[] logits = logitsFor(0.5, 0.3, 0.2);

      assertThat(sampledSet(unrestricted().minP(0.7f).build(), logits, 4_000)).containsExactly(0);
      assertThat(sampledSet(unrestricted().temperature(4.0f).minP(0.7f).build(), logits, 4_000))
          .contains(0, 1);
    }

    @Test
    void composesWithTopKByIntersection() {
      float[] logits = logitsFor(0.4, 0.3, 0.2, 0.1);

      // top-k is the tighter filter.
      assertThat(sampledSet(unrestricted().topK(2).minP(0.01f).build(), logits, 4_000))
          .containsExactlyInAnyOrder(0, 1);
      // min-p is the tighter filter (threshold 0.32 keeps only token 0).
      assertThat(sampledSet(unrestricted().topK(3).minP(0.8f).build(), logits, 4_000))
          .containsExactly(0);
    }

    @Test
    void appliesBeforeTopPSoTopPSeesTheRenormalizedSurvivors() {
      // p = [0.4, 0.3, 0.2, 0.1]; minP 0.6 keeps {0, 1} (ratios 1.0, 0.75), renormalized to
      // [0.571, 0.429]. top-p 0.5 over that distribution keeps only token 0. Had top-p run first
      // on the original distribution it would have kept {0, 1}.
      float[] logits = logitsFor(0.4, 0.3, 0.2, 0.1);

      assertThat(sampledSet(unrestricted().topP(0.5f).minP(0.6f).build(), logits, 4_000))
          .containsExactly(0);
      assertThat(sampledSet(unrestricted().topP(0.5f).build(), logits, 4_000))
          .containsExactlyInAnyOrder(0, 1);
    }

    @Test
    void seededSamplingWithMinPIsReproducible() {
      float[] logits = logitsFor(0.3, 0.25, 0.2, 0.15, 0.1);
      SamplingOptions options = unrestricted().minP(0.4f).seed(99L).build();
      Sampler first = new Sampler(options);
      Sampler second = new Sampler(options);

      int[] firstSequence = new int[64];
      int[] secondSequence = new int[64];
      for (int index = 0; index < firstSequence.length; index++) {
        firstSequence[index] = first.sample(logits, List.of());
        secondSequence[index] = second.sample(logits, List.of());
      }

      assertThat(firstSequence).isEqualTo(secondSequence);
      // Threshold 0.4 * 0.3 = 0.12 excludes only token 4 (p = 0.1).
      assertThat(Set.of(0, 1, 2, 3)).containsAll(toSet(firstSequence));
    }

    @Test
    void appliesOnTheLogitBatchPath() {
      float[] row = logitsFor(0.5, 0.3, 0.15, 0.05);
      LogitBatch batch = new LogitBatch(1, row.length, row.clone());
      Sampler sampler = new Sampler(unrestricted().minP(0.7f).build());

      for (int draw = 0; draw < 1_000; draw++) {
        assertThat(sampler.sample(batch, 0, List.of())).isZero();
      }
    }

    private static Set<Integer> toSet(int[] values) {
      Set<Integer> set = new HashSet<>();
      for (int value : values) {
        set.add(value);
      }
      return set;
    }
  }

  @Nested
  class Reproducibility {

    @Test
    void sameSeedSameSequence() {
      float[] logits = {2.0f, 2.0f, 2.0f, 2.0f};
      SamplingOptions opts = SamplingOptions.builder().temperature(1.0f).seed(123L).build();

      Sampler s1 = new Sampler(opts);
      Sampler s2 = new Sampler(opts);

      int[] seq1 = new int[10];
      int[] seq2 = new int[10];
      for (int i = 0; i < 10; i++) {
        seq1[i] = s1.sample(logits, List.of());
        seq2[i] = s2.sample(logits, List.of());
      }

      assertThat(seq1).isEqualTo(seq2);
    }

    @Test
    void differentSeedsDifferentSequences() {
      float[] logits = {2.0f, 2.0f, 2.0f, 2.0f};
      Sampler s1 = new Sampler(SamplingOptions.builder().temperature(1.0f).seed(1L).build());
      Sampler s2 = new Sampler(SamplingOptions.builder().temperature(1.0f).seed(999L).build());

      int[] seq1 = new int[20];
      int[] seq2 = new int[20];
      for (int i = 0; i < 20; i++) {
        seq1[i] = s1.sample(logits, List.of());
        seq2[i] = s2.sample(logits, List.of());
      }

      assertThat(seq1).isNotEqualTo(seq2);
    }
  }
}
