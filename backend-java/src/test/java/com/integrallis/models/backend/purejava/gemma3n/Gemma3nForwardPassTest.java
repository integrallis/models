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
package com.integrallis.models.backend.purejava.gemma3n;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The Gemma 3n graph against an independent scalar implementation.
 *
 * <p>{@link Gemma3nScalarReference} was written from the specification, not from the graph: plain
 * loops, fresh arrays, keys and values in lists rather than a ring buffer. The two share no code,
 * so agreement is evidence about the algorithm. Everything else here -- that a stream actually
 * contributes, that sparsity actually fires -- would pass on a decoder that quietly computed
 * something else, which is why the reference comparison comes first.
 */
@Tag("unit")
class Gemma3nForwardPassTest {

  private static final float TOLERANCE = 1.0e-4f;

  @Test
  void theCompleteGraphMatchesAnIndependentScalarReference() {
    Gemma3nToyModel model = Gemma3nToyModel.create();
    Gemma3nConfig config = Gemma3nToyModel.config();
    Gemma3nForwardPass graph = graph(model, config);
    Gemma3nScalarReference reference = new Gemma3nScalarReference(model, config);

    // Four tokens: past the sliding window of 2, so a sliding layer drops positions while the
    // full-attention layer keeps them, and the sharing layer reads a window that has already
    // rolled.
    for (int position = 0; position < 4; position++) {
      int token = position % Gemma3nToyModel.VOCAB;
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
  void everyLogitIsFiniteAcrossTheWholeWindow() {
    Gemma3nToyModel model = Gemma3nToyModel.create();
    Gemma3nConfig config = Gemma3nToyModel.config();
    Gemma3nForwardPass graph = graph(model, config);

    for (int position = 0; position < 6; position++) {
      float[] logits = graph.forward(position % Gemma3nToyModel.VOCAB, position);
      assertThat(logits).hasSize(Gemma3nToyModel.VOCAB);
      for (float logit : logits) {
        assertThat(Float.isFinite(logit))
            .describedAs("logit at position %s must be finite", position)
            .isTrue();
      }
    }
  }

  /**
   * The fixture must actually exercise both feed-forward branches, or the sparsity guard is
   * untested.
   *
   * <p>Layer 0 carries a finite multiplier and the rest carry {@code -inf}, exactly as the
   * published file does. If the {@code -inf} were treated as a cutoff the logits would be infinite,
   * which {@link #everyLogitIsFiniteAcrossTheWholeWindow()} would catch -- so these two together
   * pin it.
   */
  @Test
  void theFixtureExercisesBothFeedForwardBranches() {
    Gemma3nConfig config = Gemma3nToyModel.config();

    assertThat(config.usesActivationSparsity(0)).isTrue();
    for (int layer = 1; layer < config.numLayers(); layer++) {
      assertThat(config.usesActivationSparsity(layer)).isFalse();
    }
    assertThat(config.sparsityScaleByLayer().get(1)).isEqualTo(Float.NEGATIVE_INFINITY);
  }

  /** And both attention branches: an owning layer and a sharing one. */
  @Test
  void theFixtureExercisesBothAttentionBranches() {
    Gemma3nConfig config = Gemma3nToyModel.config();

    assertThat(config.ownsKvCache(0)).isTrue();
    assertThat(config.ownsKvCache(2)).isTrue();
    assertThat(config.ownsKvCache(3)).isFalse();
    assertThat(config.kvSourceLayer(3)).isEqualTo(1);
    // Both spans present, so a sliding layer and a full-attention layer both attend.
    assertThat(config.usesSlidingWindow(1)).isTrue();
    assertThat(config.usesSlidingWindow(2)).isFalse();
  }

  /**
   * An inactive stream must reach the output.
   *
   * <p>The streams are merged only at the very end, so a graph that carried them faithfully and
   * then kept just the active one would still produce plausible logits, and the reference
   * comparison would not see it -- the reference would have to make the same mistake, which it does
   * not. Perturbing one inactive stream's unembed projection is the direct check: if the inactive
   * streams are read, the logits move.
   */
  @Test
  void theInactiveStreamsReachTheOutput() {
    Gemma3nConfig config = Gemma3nToyModel.config();
    float[] baseline = firstLogits(Gemma3nToyModel.create(), config);

    float[] perturbed =
        Gemma3nToyModel.matrix(
            (Gemma3nToyModel.ALTUP - 1) * Gemma3nToyModel.DIM, Gemma3nToyModel.DIM, 11);
    perturbed[0] += 0.5f;
    float[] changed =
        firstLogits(
            Gemma3nToyModel.create(java.util.Map.of("altup_unembd_proj.weight", perturbed)),
            config);

    assertThat(changed)
        .describedAs("the inactive streams' unembed projection must affect the logits")
        .isNotEqualTo(baseline);
  }

  /** And the projection that CREATES them: without it every stream would start out identical. */
  @Test
  void theStreamInitialisationProjectionReachesTheOutput() {
    Gemma3nConfig config = Gemma3nToyModel.config();
    float[] baseline = firstLogits(Gemma3nToyModel.create(), config);

    float[] perturbed =
        Gemma3nToyModel.matrix(
            (Gemma3nToyModel.ALTUP - 1) * Gemma3nToyModel.DIM, Gemma3nToyModel.DIM, 11);
    perturbed[0] += 0.5f;
    float[] changed =
        firstLogits(
            Gemma3nToyModel.create(java.util.Map.of("altup_proj.weight", perturbed)), config);

    assertThat(changed).isNotEqualTo(baseline);
  }

  /**
   * LAuReL is a residual around attention, so removing its low-rank pair must change the answer.
   */
  @Test
  void theLaurelResidualReachesTheOutput() {
    Gemma3nConfig config = Gemma3nToyModel.config();
    float[] baseline = firstLogits(Gemma3nToyModel.create(), config);

    float[] zeroed = new float[Gemma3nToyModel.LAUREL_RANK * Gemma3nToyModel.DIM];
    float[] changed =
        firstLogits(
            Gemma3nToyModel.create(java.util.Map.of("blk.0.laurel_l.weight", zeroed)), config);

    assertThat(changed).isNotEqualTo(baseline);
  }

  /** The second embedding table must reach the output through the per-layer injection. */
  @Test
  void thePerLayerEmbeddingTableReachesTheOutput() {
    Gemma3nConfig config = Gemma3nToyModel.config();
    float[] baseline = firstLogits(Gemma3nToyModel.create(), config);

    float[] zeroed =
        new float[Gemma3nToyModel.VOCAB * Gemma3nToyModel.PER_LAYER_DIM * Gemma3nToyModel.LAYERS];
    float[] changed =
        firstLogits(
            Gemma3nToyModel.create(java.util.Map.of("per_layer_token_embd.weight", zeroed)),
            config);

    assertThat(changed).isNotEqualTo(baseline);
  }

  /**
   * The published file stores {@code laurel_l}, {@code laurel_r} and {@code altup_router} as F16
   * among otherwise-Q8_0 tensors, so those types have to be decoded as F16 and not as whatever the
   * neighbours are.
   *
   * <p>The reference here is an F32 twin holding exactly the values F16 rounds to, so the two
   * differ only in how the bytes are encoded. Decoding half-precision bytes as full-precision
   * floats is not a small error, so a loose tolerance still separates "reads the type" from
   * "ignores it".
   */
  @Test
  void theHalfPrecisionTensorsAreDecodedAsF16() {
    Gemma3nConfig config = Gemma3nToyModel.config();
    float[] half = firstLogits(Gemma3nToyModel.createWithHalfPrecisionExtras(), config);

    java.util.Map<String, float[]> rounded = new java.util.HashMap<>();
    for (int layer = 0; layer < Gemma3nToyModel.LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      rounded.put(
          prefix + "laurel_l.weight",
          Gemma3nToyModel.halfRounded(
              Gemma3nToyModel.matrix(
                  Gemma3nToyModel.LAUREL_RANK, Gemma3nToyModel.DIM, 230 + layer)));
      rounded.put(
          prefix + "laurel_r.weight",
          Gemma3nToyModel.halfRounded(
              Gemma3nToyModel.matrix(
                  Gemma3nToyModel.DIM, Gemma3nToyModel.LAUREL_RANK, 240 + layer)));
      rounded.put(
          prefix + "altup_router.weight",
          Gemma3nToyModel.halfRounded(
              Gemma3nToyModel.matrix(Gemma3nToyModel.ALTUP, Gemma3nToyModel.DIM, 180 + layer)));
    }
    float[] twin = firstLogits(Gemma3nToyModel.create(rounded), config);

    assertThat(half).hasSameSizeAs(twin);
    for (float logit : half) {
      assertThat(Float.isFinite(logit)).describedAs("a misread type gives NaN or huge").isTrue();
    }
    for (int index = 0; index < twin.length; index++) {
      assertThat(half[index])
          .describedAs("F16 logit %s against its F32 twin", index)
          .isEqualTo(twin[index], within(1.0e-5f));
    }
  }

  /**
   * And the twin must genuinely differ from the unrounded fixture, or it is not a twin of anything.
   */
  @Test
  void theHalfPrecisionRoundingIsObservable() {
    Gemma3nConfig config = Gemma3nToyModel.config();

    assertThat(firstLogits(Gemma3nToyModel.createWithHalfPrecisionExtras(), config))
        .describedAs("F16 storage loses precision, so it cannot equal the unrounded fixture")
        .isNotEqualTo(firstLogits(Gemma3nToyModel.create(), config));
  }

  @Test
  void aNonSequentialPositionIsRefused() {
    Gemma3nToyModel model = Gemma3nToyModel.create();
    Gemma3nConfig config = Gemma3nToyModel.config();
    Gemma3nForwardPass graph = graph(model, config);
    graph.forward(1, 0);

    assertThatThrownBy(() -> graph.forward(1, 5))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("sequential");
  }

  @Test
  void resetReturnsTheGraphToItsStartingState() {
    Gemma3nToyModel model = Gemma3nToyModel.create();
    Gemma3nConfig config = Gemma3nToyModel.config();
    Gemma3nForwardPass graph = graph(model, config);

    float[] first = graph.forward(1, 0).clone();
    graph.forward(2, 1);
    graph.reset();

    assertThat(graph.nextPosition()).isZero();
    assertThat(graph.forward(1, 0)).containsExactly(first);
  }

  private static float[] firstLogits(Gemma3nToyModel model, Gemma3nConfig config) {
    return graph(model, config).forward(1, 0).clone();
  }

  private static Gemma3nForwardPass graph(Gemma3nToyModel model, Gemma3nConfig config) {
    LayeredKvCache cache = Gemma3nKvCache.create(config, 16, 1);
    return new Gemma3nForwardPass(config, Gemma3nWeights.fromGgufFile(model.file(), config), cache);
  }
}
