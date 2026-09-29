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

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.lfm2.Lfm2Config;
import com.integrallis.models.backend.purejava.lfm2.Lfm2ForwardPass;
import com.integrallis.models.backend.purejava.lfm2.Lfm2ToyModel;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Rewinding an LFM2 sequence, which the decoder does by replaying the tokens it retained.
 *
 * <p>The adapter used to refuse outright. That refusal was correct about the state -- a short
 * convolution's shift register cannot be truncated the way a key-value cache can -- but it took
 * LFM2 out of the RAG harness entirely: reusing a shared prompt prefix begins by rewinding to it,
 * so the model failed at the first question with {@code UnsupportedOperationException} rather than
 * answering any.
 *
 * <p>What has to hold is that a replayed state is <b>the same state</b>, not merely a plausible
 * one. Every test here therefore compares a rewound decoder against a second decoder that reached
 * the same position without ever rewinding, and requires the logits to be identical -- not close.
 * Replaying is deterministic, so any difference at all is a difference in state.
 */
@Tag("unit")
class Lfm2RewindTest {

  private static final int[] PROMPT = {0, 1, 2, 3};

  /** Within the toy vocabulary, which is only four tokens wide. */
  private static final int[] CONTINUATION = {2, 0};

  @Test
  void aRewoundDecoderProducesExactlyWhatAFreshOneDoes() {
    Lfm2DecoderAdapter rewound = adapter();
    Lfm2DecoderAdapter fresh = adapter();

    // Take the long path: run the whole prompt, then go back to position 2 and run the rest again.
    rewound.prefill(PROMPT, 0);
    rewound.rewind(2);
    assertThat(rewound.checkpoint()).isEqualTo(2);
    float[] afterRewind = rewound.prefill(new int[] {PROMPT[2], PROMPT[3]}, 2).clone();

    // The short path: the same four tokens, never rewound.
    float[] direct = fresh.prefill(PROMPT, 0).clone();

    assertThat(afterRewind)
        .describedAs("replay must reconstruct the convolution state exactly")
        .containsExactly(direct);
  }

  @Test
  void rewindingToTheStartIsTheSameAsAReset() {
    Lfm2DecoderAdapter rewound = adapter();
    Lfm2DecoderAdapter fresh = adapter();

    rewound.prefill(PROMPT, 0);
    rewound.rewind(0);

    assertThat(rewound.checkpoint()).isZero();
    assertThat(rewound.prefill(PROMPT, 0)).containsExactly(fresh.prefill(PROMPT, 0));
  }

  @Test
  void aSharedPrefixCanBeReusedAcrossTwoDifferentContinuations() {
    // The harness's actual pattern: prefill the evidence once, answer, rewind to it, answer again.
    Lfm2DecoderAdapter reused = adapter();
    reused.prefill(PROMPT, 0);
    int prefix = reused.checkpoint();
    float[] firstAnswer = reused.prefill(CONTINUATION, prefix).clone();

    reused.rewind(prefix);
    float[] secondAnswer = reused.prefill(CONTINUATION, prefix).clone();

    assertThat(secondAnswer)
        .describedAs("the same continuation on the same prefix must give the same logits")
        .containsExactly(firstAnswer);

    // And against a decoder that never reused anything.
    Lfm2DecoderAdapter fresh = adapter();
    fresh.prefill(PROMPT, 0);
    assertThat(fresh.prefill(CONTINUATION, prefix)).containsExactly(firstAnswer);
  }

  @Test
  void rewindingToWhereTheDecoderAlreadyIsDoesNothing() {
    Lfm2DecoderAdapter decoder = adapter();
    float[] before = decoder.prefill(PROMPT, 0).clone();

    decoder.rewind(PROMPT.length);

    assertThat(decoder.checkpoint()).isEqualTo(PROMPT.length);
    // No replay happened, so the next token continues from the same state.
    Lfm2DecoderAdapter fresh = adapter();
    fresh.prefill(PROMPT, 0);
    assertThat(decoder.forward(1, PROMPT.length)).containsExactly(fresh.forward(1, PROMPT.length));
    assertThat(before).isNotNull();
  }

  @Test
  void rewindingForwardIsRefusedRatherThanInventingTokens() {
    Lfm2DecoderAdapter decoder = adapter();
    decoder.prefill(PROMPT, 0);

    assertThatThrownBy(() -> decoder.rewind(PROMPT.length + 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("checkpoint");
    assertThatThrownBy(() -> decoder.rewind(-1)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aResetDiscardsTheHistoryItWouldOtherwiseReplay() {
    Lfm2DecoderAdapter decoder = adapter();
    decoder.prefill(PROMPT, 0);
    decoder.reset();

    assertThat(decoder.checkpoint()).isZero();
    // A rewind past the new position must be refused, not satisfied from the discarded history.
    assertThatThrownBy(() -> decoder.rewind(2)).isInstanceOf(IllegalArgumentException.class);

    Lfm2DecoderAdapter fresh = adapter();
    assertThat(decoder.prefill(PROMPT, 0)).containsExactly(fresh.prefill(PROMPT, 0));
  }

  private static Lfm2DecoderAdapter adapter() {
    GgufFile file = Lfm2ToyModel.file(30, 40);
    Lfm2Config config = Lfm2Config.fromMetadata(file.metadata());
    return new Lfm2DecoderAdapter(Lfm2ForwardPass.fromGgufFile(file, config, 32));
  }
}
