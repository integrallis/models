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
package com.integrallis.models.backend.purejava.llama;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Pins the attention window arithmetic for encoders.
 *
 * <p>The equivalence gate cannot cover this. Its longest probe is a couple of dozen tokens, and a
 * window of 512 only begins to mask anything past position 256 — so every probe in it would pass
 * unchanged if the window were removed entirely. Verified by hand once against llama.cpp on a
 * 940-token input (cosine 0.9994 with the window, 0.8603 without); this test is what keeps it from
 * regressing after that.
 *
 * <p>The values are llama.cpp's {@code is_masked_swa} under {@code LLAMA_SWA_TYPE_SYMMETRIC}: a key
 * at {@code p0} is visible to a query at {@code p1} exactly when {@code |p1 - p0| <= n_swa / 2}.
 */
@Tag("unit")
class EncoderAttentionWindowTest {

  private static final int SLIDING_WINDOW = 512;
  private static final int HALF_WINDOW = SLIDING_WINDOW / 2;
  private static final int PATTERN = 6;
  private static final int LAYERS = 24;

  private static LlamaConfig config(DecoderArchitecture architecture) {
    return new LlamaConfig(
        architecture,
        768,
        LAYERS,
        3,
        1,
        256,
        256,
        262144,
        2048,
        1152,
        1_000_000.0f,
        1.0f,
        10_000.0f,
        1e-6f,
        SLIDING_WINDOW,
        PATTERN,
        0.0f);
  }

  /** Layer 0 slides; layer 5 is the full-attention layer in a period of six. */
  private static final int SLIDING_LAYER = 0;

  private static final int FULL_LAYER = PATTERN - 1;

  @Nested
  class SymmetricWindow {

    private final LlamaConfig config = config(DecoderArchitecture.GEMMA_EMBEDDING);

    @Test
    void reachesHalfTheWindowInEachDirection() {
      int position = 700;
      // Far enough past the forward reach that the window bounds it rather than the sequence end.
      int lastPosition = 1500;

      assertThat(config.attentionStartPosition(SLIDING_LAYER, position))
          .isEqualTo(position - HALF_WINDOW);
      assertThat(config.attentionEndPosition(SLIDING_LAYER, position, lastPosition))
          .isEqualTo(position + HALF_WINDOW);
    }

    @Test
    void isNotTheCausalWindowWidth() {
      // The causal reading would be position - n_swa + 1, twice as far back. Both are plausible
      // and only one matches the oracle, so the difference is asserted rather than implied.
      int position = 700;

      assertThat(config.attentionStartPosition(SLIDING_LAYER, position))
          .isNotEqualTo(position - SLIDING_WINDOW + 1);
    }

    @Test
    void clampsToTheSequenceRatherThanRunningOffEitherEnd() {
      assertThat(config.attentionStartPosition(SLIDING_LAYER, 10)).isZero();
      assertThat(config.attentionEndPosition(SLIDING_LAYER, 900, 940)).isEqualTo(940);
    }

    @Test
    void seesEverythingOnTheFullAttentionLayer() {
      assertThat(config.attentionStartPosition(FULL_LAYER, 700)).isZero();
      assertThat(config.attentionEndPosition(FULL_LAYER, 700, 940)).isEqualTo(940);
    }

    @Test
    void slidesOnEveryLayerButTheLastOfEachPeriod() {
      for (int layer = 0; layer < LAYERS; layer++) {
        assertThat(config.usesSlidingWindow(layer))
            .describedAs("layer %d", layer)
            .isEqualTo(layer % PATTERN != PATTERN - 1);
      }
    }

    @Test
    void windowIsInertBelowItsHalfWidth() {
      // Why the gate's short probes cannot detect a broken window: below this length every
      // position sees the whole sequence whatever the arithmetic says.
      assertThat(config.attentionStartPosition(SLIDING_LAYER, 25)).isZero();
      assertThat(config.attentionEndPosition(SLIDING_LAYER, 25, 30)).isEqualTo(30);
    }
  }

  @Nested
  class CausalWindow {

    private final LlamaConfig config = config(DecoderArchitecture.GEMMA3);

    @Test
    void spansTheWindowEndingAtTheQuery() {
      assertThat(config.attentionStartPosition(SLIDING_LAYER, 700))
          .isEqualTo(700 - SLIDING_WINDOW + 1);
    }

    @Test
    void neverLooksPastTheQuery() {
      assertThat(config.attentionEndPosition(SLIDING_LAYER, 700, 940)).isEqualTo(700);
      assertThat(config.attentionEndPosition(FULL_LAYER, 700, 940)).isEqualTo(700);
    }

    @Test
    void isNotBidirectional() {
      assertThat(config.usesBidirectionalAttention()).isFalse();
    }
  }

  @Nested
  class GemmaFamilyLayout {

    @Test
    void embeddingGemmaSharesGemma3BlockStructure() {
      LlamaConfig gemma3 = config(DecoderArchitecture.GEMMA3);
      LlamaConfig embedding = config(DecoderArchitecture.GEMMA_EMBEDDING);

      assertThat(embedding.usesNeoxRope()).isEqualTo(gemma3.usesNeoxRope());
      assertThat(embedding.usesGeluFfn()).isEqualTo(gemma3.usesGeluFfn());
      assertThat(embedding.usesPostAttentionNorm()).isEqualTo(gemma3.usesPostAttentionNorm());
      assertThat(embedding.usesPostFfnNorm()).isEqualTo(gemma3.usesPostFfnNorm());
      assertThat(embedding.embeddingScale()).isEqualTo(gemma3.embeddingScale());
      assertThat(embedding.usesStandardLlamaLayerSemantics()).isFalse();
    }

    @Test
    void onlyEmbeddingGemmaAttendsBidirectionally() {
      assertThat(config(DecoderArchitecture.GEMMA_EMBEDDING).usesBidirectionalAttention()).isTrue();
      for (DecoderArchitecture architecture : DecoderArchitecture.values()) {
        if (architecture != DecoderArchitecture.GEMMA_EMBEDDING) {
          assertThat(config(architecture).usesBidirectionalAttention())
              .describedAs("%s", architecture)
              .isFalse();
        }
      }
    }

    @Test
    void slidingWindowLayersUseTheirOwnRotaryBase() {
      LlamaConfig config = config(DecoderArchitecture.GEMMA_EMBEDDING);

      assertThat(config.ropeTheta(SLIDING_LAYER)).isEqualTo(10_000.0f);
      assertThat(config.ropeTheta(FULL_LAYER)).isEqualTo(1_000_000.0f);
    }
  }
}
