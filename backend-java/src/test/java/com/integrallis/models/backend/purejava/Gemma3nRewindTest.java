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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gemma3n.Gemma3nTestAccess;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Rewinding a Gemma 3n sequence, which the adapter does by replaying the tokens it retained.
 *
 * <p>Written before the decoder ever reached the fleet, because this exact defect did reach it for
 * LFM2: that adapter refused {@code rewind}, and the RAG harness rewinds to reuse a shared prompt
 * prefix before the first question, so the model failed every case rather than answering any.
 *
 * <p>Each test compares a rewound decoder against a second decoder that reached the same position
 * without ever rewinding, and requires the logits to be <b>identical</b> -- replay is
 * deterministic, so any difference at all is a difference in state.
 */
@Tag("unit")
class Gemma3nRewindTest {

  private static final int[] PROMPT = {0, 1, 2, 3};
  private static final int[] CONTINUATION = {2, 0};

  @Test
  void aRewoundDecoderProducesExactlyWhatAFreshOneDoes() {
    Gemma3nDecoderAdapter rewound = adapter();
    Gemma3nDecoderAdapter fresh = adapter();

    rewound.prefill(PROMPT, 0);
    rewound.rewind(2);
    assertThat(rewound.checkpoint()).isEqualTo(2);
    float[] afterRewind = rewound.prefill(new int[] {PROMPT[2], PROMPT[3]}, 2).clone();

    assertThat(afterRewind)
        .describedAs("replay must reconstruct the AltUp streams and the key-value cache exactly")
        .containsExactly(fresh.prefill(PROMPT, 0));
  }

  @Test
  void aSharedPrefixCanBeReusedAcrossTwoContinuations() {
    Gemma3nDecoderAdapter reused = adapter();
    reused.prefill(PROMPT, 0);
    int prefix = reused.checkpoint();
    float[] first = reused.prefill(CONTINUATION, prefix).clone();

    reused.rewind(prefix);
    float[] second = reused.prefill(CONTINUATION, prefix).clone();

    assertThat(second).containsExactly(first);

    Gemma3nDecoderAdapter fresh = adapter();
    fresh.prefill(PROMPT, 0);
    assertThat(fresh.prefill(CONTINUATION, prefix)).containsExactly(first);
  }

  /**
   * A rewind that crosses the sliding window is the interesting one.
   *
   * <p>The window is two positions, so rewinding to 1 from position 4 asks for a state whose
   * evicted ring entries are gone. Truncating the cache could not serve that; replaying rebuilds
   * it.
   */
  @Test
  void aRewindPastTheSlidingWindowStillReconstructsTheState() {
    Gemma3nDecoderAdapter rewound = adapter();
    rewound.prefill(new int[] {0, 1, 2, 3, 4}, 0);
    rewound.rewind(1);

    Gemma3nDecoderAdapter fresh = adapter();
    fresh.forward(0, 0);

    assertThat(rewound.checkpoint()).isEqualTo(1);
    assertThat(rewound.forward(1, 1)).containsExactly(fresh.forward(1, 1));
  }

  @Test
  void rewindingToTheStartIsTheSameAsAReset() {
    Gemma3nDecoderAdapter rewound = adapter();
    rewound.prefill(PROMPT, 0);
    rewound.rewind(0);

    assertThat(rewound.checkpoint()).isZero();
    assertThat(rewound.prefill(PROMPT, 0)).containsExactly(adapter().prefill(PROMPT, 0));
  }

  @Test
  void rewindingForwardIsRefusedRatherThanInventingTokens() {
    Gemma3nDecoderAdapter decoder = adapter();
    decoder.prefill(PROMPT, 0);

    assertThatThrownBy(() -> decoder.rewind(PROMPT.length + 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("checkpoint");
    assertThatThrownBy(() -> decoder.rewind(-1)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aSecondSessionIsRefusedRatherThanSharingOneAltUpState() {
    Gemma3nDecoderAdapter decoder = adapter();
    decoder.openSession();

    assertThatThrownBy(decoder::openSession)
        .isInstanceOf(UnsupportedOperationException.class)
        .hasMessageContaining("single session");
  }

  @Test
  void aResetDiscardsTheHistoryItWouldOtherwiseReplay() {
    Gemma3nDecoderAdapter decoder = adapter();
    decoder.prefill(PROMPT, 0);
    decoder.reset();

    assertThat(decoder.checkpoint()).isZero();
    assertThatThrownBy(() -> decoder.rewind(2)).isInstanceOf(IllegalArgumentException.class);
    assertThat(decoder.prefill(PROMPT, 0)).containsExactly(adapter().prefill(PROMPT, 0));
  }

  private static Gemma3nDecoderAdapter adapter() {
    return new Gemma3nDecoderAdapter(Gemma3nTestAccess.toyForwardPass());
  }
}
