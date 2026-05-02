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

import com.integrallis.models.api.SamplingOptions;
import java.util.HashSet;
import java.util.List;
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
