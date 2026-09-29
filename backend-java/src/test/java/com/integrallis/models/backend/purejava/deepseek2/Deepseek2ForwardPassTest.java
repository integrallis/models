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
package com.integrallis.models.backend.purejava.deepseek2;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The DeepSeek-V2 graph against an independent scalar implementation.
 *
 * <p>{@link Deepseek2ScalarReference} recomputes the YaRN frequency ramp and the expert ranking
 * itself rather than calling {@code RotaryTable} or {@code ExpertRouting}, so the two agreeing is
 * evidence about latent attention and unnormalised routing and not about shared helpers.
 */
@Tag("unit")
class Deepseek2ForwardPassTest {

  private static final float TOLERANCE = 1.0e-4f;

  @Test
  void theCompleteGraphMatchesAnIndependentScalarReference() {
    Deepseek2ToyModel model = Deepseek2ToyModel.create();
    Deepseek2Config config = Deepseek2ToyModel.config();
    Deepseek2ForwardPass graph = graph(model, config);
    Deepseek2ScalarReference reference = new Deepseek2ScalarReference(model, config);

    for (int position = 0; position < 4; position++) {
      int token = position % Deepseek2ToyModel.VOCAB;
      float[] actual = graph.forward(token, position).clone();
      float[] expected = reference.forward(token);

      assertThat(actual).hasSameSizeAs(expected);
      for (int index = 0; index < expected.length; index++) {
        assertThat(actual[index])
            .describedAs("logit %s at position %s", index, position)
            .isEqualTo(expected[index], within(TOLERANCE));
      }
    }
  }

  @Test
  void everyLogitIsFinite() {
    Deepseek2ForwardPass graph = graph(Deepseek2ToyModel.create(), Deepseek2ToyModel.config());

    for (int position = 0; position < 6; position++) {
      float[] logits = graph.forward(position % Deepseek2ToyModel.VOCAB, position);
      assertThat(logits).hasSize(Deepseek2ToyModel.VOCAB);
      for (float logit : logits) {
        assertThat(Float.isFinite(logit)).isTrue();
      }
    }
  }

  /** The fixture must run both feed-forward paths, or one of them is untested. */
  @Test
  void theFixtureExercisesBothFeedForwardPaths() {
    Deepseek2Config config = Deepseek2ToyModel.config();

    assertThat(config.usesMixtureOfExperts(0)).isFalse();
    assertThat(config.usesMixtureOfExperts(1)).isTrue();
    assertThat(config.usesMixtureOfExperts(2)).isTrue();
    // And more than one head, so the shared rotary key part has to be copied rather than placed
    // once.
    assertThat(config.numHeads()).isGreaterThan(1);
  }

  /**
   * The key-value latent must reach the output.
   *
   * <p>Latent attention is two projections with a norm between them, so a graph that skipped the
   * decompressing one and used the latent directly would have the wrong widths and fail loudly --
   * but one that ignored the <b>norm</b> would not. Perturbing the norm is the direct check.
   */
  @Test
  void theLatentNormReachesTheOutput() {
    Deepseek2Config config = Deepseek2ToyModel.config();
    float[] baseline = firstLogits(Deepseek2ToyModel.create(), config);

    float[] altered = Deepseek2ToyModel.norm(40 + 1, Deepseek2ToyModel.KV_LORA_RANK);
    altered[0] += 0.5f;
    float[] changed =
        firstLogits(
            Deepseek2ToyModel.create(Map.of("blk.1.attn_kv_a_norm.weight", altered)), config);

    assertThat(changed).isNotEqualTo(baseline);
  }

  /** The fused shared expert must contribute: it is added unweighted, so zeroing it must show. */
  @Test
  void theFusedSharedExpertReachesTheOutput() {
    Deepseek2Config config = Deepseek2ToyModel.config();
    float[] baseline = firstLogits(Deepseek2ToyModel.create(), config);

    float[] zeroed = new float[config.embeddingDim() * config.sharedExpertHiddenDim()];
    float[] changed =
        firstLogits(
            Deepseek2ToyModel.create(Map.of("blk.1.ffn_down_shexp.weight", zeroed)), config);

    assertThat(changed).isNotEqualTo(baseline);
  }

  @Test
  void aNonSequentialPositionIsRefused() {
    Deepseek2ForwardPass graph = graph(Deepseek2ToyModel.create(), Deepseek2ToyModel.config());
    graph.forward(1, 0);

    assertThatThrownBy(() -> graph.forward(1, 5))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("sequential");
  }

  /**
   * The graph's own rewind bound, which the adapter's check would otherwise hide.
   *
   * <p>{@code rewindTo} is public, so a caller can reach it without going through the adapter. Its
   * bound has to hold on its own -- a forward jump would leave the cache holding nothing for the
   * positions in between and attention would read them as zeros.
   */
  @Test
  void theGraphRefusesAForwardRewind() {
    Deepseek2ForwardPass graph = graph(Deepseek2ToyModel.create(), Deepseek2ToyModel.config());
    graph.forward(1, 0);
    graph.forward(2, 1);

    assertThatThrownBy(() -> graph.rewindTo(5))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("checkpoint");
    assertThatThrownBy(() -> graph.rewindTo(-1)).isInstanceOf(IllegalArgumentException.class);
    // And an exact rewind is honoured.
    graph.rewindTo(1);
    assertThat(graph.nextPosition()).isEqualTo(1);
  }

  @Test
  void resetReturnsTheGraphToItsStartingState() {
    Deepseek2ForwardPass graph = graph(Deepseek2ToyModel.create(), Deepseek2ToyModel.config());
    float[] first = graph.forward(1, 0).clone();
    graph.forward(2, 1);
    graph.reset();

    assertThat(graph.nextPosition()).isZero();
    assertThat(graph.forward(1, 0)).containsExactly(first);
  }

  private static float[] firstLogits(Deepseek2ToyModel model, Deepseek2Config config) {
    return graph(model, config).forward(1, 0).clone();
  }

  private static Deepseek2ForwardPass graph(Deepseek2ToyModel model, Deepseek2Config config) {
    return new Deepseek2ForwardPass(
        config, Deepseek2Weights.fromGgufFile(model.file(), config), 16);
  }
}
