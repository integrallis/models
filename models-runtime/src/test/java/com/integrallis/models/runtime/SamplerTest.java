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
  static class Greedy {

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
  static class Temperature {

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
  static class TopK {

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
  static class TopP {

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
  }

  @Nested
  static class RepetitionPenalty {

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
  static class Selection {

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

    /** Mirrors the original stable-full-sort implementation, as an oracle for the fast path. */
    private static int fullSortReference(SamplingOptions options, float[] adjusted, Random rng) {
      for (int index = 0; index < adjusted.length; index++) {
        adjusted[index] /= options.temperature();
      }
      float max = Float.NEGATIVE_INFINITY;
      for (float value : adjusted) {
        if (value > max) max = value;
      }
      float sum = 0;
      for (int index = 0; index < adjusted.length; index++) {
        adjusted[index] = (float) Math.exp(adjusted[index] - max);
        sum += adjusted[index];
      }
      for (int index = 0; index < adjusted.length; index++) {
        adjusted[index] /= sum;
      }

      record TokenProb(int id, float prob) {}
      List<TokenProb> sorted = new ArrayList<>(adjusted.length);
      for (int index = 0; index < adjusted.length; index++) {
        sorted.add(new TokenProb(index, adjusted[index]));
      }
      sorted.sort(Comparator.comparingDouble(TokenProb::prob).reversed());

      int topK = Math.min(options.topK(), sorted.size());
      sorted = new ArrayList<>(sorted.subList(0, topK));

      float cumulative = 0;
      int cutoff = sorted.size();
      for (int index = 0; index < sorted.size(); index++) {
        cumulative += sorted.get(index).prob();
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

  @Nested
  static class Reproducibility {

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
