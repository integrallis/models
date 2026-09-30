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

import com.integrallis.models.backend.purejava.deepseek2.Deepseek2TestAccess;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Rewinding a DeepSeek-V2 sequence, which this adapter does by truncating rather than replaying.
 *
 * <p>The cache is indexed by absolute position, so moving the position back is exact and free. That
 * is a claim about correctness, not just about speed, so it is checked the same way the replaying
 * decoders are: against a second decoder that reached the same position without ever rewinding,
 * requiring <b>identical</b> logits.
 */
@Tag("unit")
class Deepseek2RewindTest {

  private static final int[] PROMPT = {0, 1, 2, 3};
  private static final int[] CONTINUATION = {2, 0};

  @Test
  void aRewoundDecoderProducesExactlyWhatAFreshOneDoes() {
    Deepseek2DecoderAdapter rewound = adapter();
    rewound.prefill(PROMPT, 0);
    rewound.rewind(2);
    assertThat(rewound.checkpoint()).isEqualTo(2);
    float[] afterRewind = rewound.prefill(new int[] {PROMPT[2], PROMPT[3]}, 2).clone();

    assertThat(afterRewind).containsExactly(adapter().prefill(PROMPT, 0));
  }

  @Test
  void aSharedPrefixCanBeReusedAcrossTwoContinuations() {
    Deepseek2DecoderAdapter reused = adapter();
    reused.prefill(PROMPT, 0);
    int prefix = reused.checkpoint();
    float[] first = reused.prefill(CONTINUATION, prefix).clone();

    reused.rewind(prefix);
    float[] second = reused.prefill(CONTINUATION, prefix).clone();

    assertThat(second).containsExactly(first);
  }

  /**
   * Stale cache entries above the checkpoint must not be read.
   *
   * <p>Truncation leaves them in place. Continuing with a <b>different</b> token after the rewind
   * is what proves attention reads only up to the current position: if it read further, the old
   * entries would contribute and the result would not match a decoder that never saw them.
   */
  @Test
  void entriesAboveTheCheckpointAreNotRead() {
    Deepseek2DecoderAdapter rewound = adapter();
    rewound.prefill(new int[] {0, 1, 2, 3, 4}, 0);
    rewound.rewind(2);
    float[] different = rewound.prefill(new int[] {4, 4}, 2).clone();

    Deepseek2DecoderAdapter fresh = adapter();
    assertThat(different)
        .describedAs("the discarded tokens 2,3,4 must not influence the new continuation")
        .containsExactly(fresh.prefill(new int[] {0, 1, 4, 4}, 0));
  }

  @Test
  void rewindingToTheStartIsTheSameAsAReset() {
    Deepseek2DecoderAdapter rewound = adapter();
    rewound.prefill(PROMPT, 0);
    rewound.rewind(0);

    assertThat(rewound.checkpoint()).isZero();
    assertThat(rewound.prefill(PROMPT, 0)).containsExactly(adapter().prefill(PROMPT, 0));
  }

  @Test
  void rewindingForwardIsRefused() {
    Deepseek2DecoderAdapter decoder = adapter();
    decoder.prefill(PROMPT, 0);

    assertThatThrownBy(() -> decoder.rewind(PROMPT.length + 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("checkpoint");
    assertThatThrownBy(() -> decoder.rewind(-1)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aSecondSessionIsRefused() {
    Deepseek2DecoderAdapter decoder = adapter();
    decoder.openSession();

    assertThatThrownBy(decoder::openSession)
        .isInstanceOf(UnsupportedOperationException.class)
        .hasMessageContaining("single session");
  }

  private static Deepseek2DecoderAdapter adapter() {
    return new Deepseek2DecoderAdapter(Deepseek2TestAccess.toyForwardPass());
  }
}
