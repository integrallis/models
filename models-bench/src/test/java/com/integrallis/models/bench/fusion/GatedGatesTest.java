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
package com.integrallis.models.bench.fusion;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.Tokenizer;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class GatedGatesTest {
  private final ExecutorService executor = Executors.newFixedThreadPool(3);
  private final DecodeSettings settings = new DecodeSettings(0f, 0, 1f, 7L, 40, 1);
  private static final int[] PROMPT = {3, 1, 4, 1, 5, 9, 2};

  @AfterEach
  void shutdown() {
    executor.shutdownNow();
  }

  private static List<FusionMember> toys() {
    return List.of(
        new ToyMember("L", 1.3, 0.0), new ToyMember("M", 2.1, 0.4), new ToyMember("I", 0.7, -0.2));
  }

  @Test
  void g1GatedHoldsOnToyMembers() {
    GatedGates.Identity lead =
        GatedGates.leadIdentity(toys(), ToyMemberBackend::new, PROMPT, settings);
    assertThat(lead.identical()).isTrue();
    assertThat(lead.firstDivergence()).isEqualTo(-1);
    GatedGates.Identity every =
        GatedGates.everyTokenIdentity(toys(), ToyMemberBackend::new, PROMPT, settings, executor);
    assertThat(every.identical()).isTrue();
    assertThat(every.gated().length).isGreaterThan(3);
  }

  @Test
  void g8HoldsOnAnExactMemberAndCatchesOneWhosePrefillDropsTheLastToken() {
    ToyMember exact = new ToyMember("M", 2.1, 0.4);
    // Any token sequence will do for G8; a fixed one keeps the deltas independent of where the
    // toy lead happens to emit its end token.
    int[] continuation = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2};
    for (int delta : new int[] {1, 2, 7}) {
      GatedGates.CatchUp row =
          GatedGates.catchUpEqualsStepping(exact, PROMPT, continuation, delta, 1e-6);
      assertThat(row.within()).as("delta " + delta).isTrue();
      assertThat(row.argmaxEqual()).isTrue();
      assertThat(row.maxAbsDiff()).isEqualTo(0.0);
    }
    GatedGates.CatchUp broken =
        GatedGates.catchUpEqualsStepping(
            new DroppingMember(new ToyMember("M", 2.1, 0.4)), PROMPT, continuation, 7, 1e-6);
    assertThat(broken.within()).isFalse();
    assertThat(broken.maxAbsDiff()).isGreaterThan(1e-6);
  }

  /** A member whose multi-token prefill forgets the last token of the delta: G8 must catch it. */
  private static final class DroppingMember implements FusionMember {
    private final FusionMember inner;

    DroppingMember(FusionMember inner) {
      this.inner = inner;
    }

    @Override
    public String name() {
      return inner.name();
    }

    @Override
    public Tokenizer tokenizer() {
      return inner.tokenizer();
    }

    @Override
    public void reset() {
      inner.reset();
    }

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      if (startPosition == 0 || tokens.length == 1) {
        return inner.prefill(tokens, startPosition);
      }
      // Consume all but the last token, then pretend the last one was seen.
      float[] logits =
          inner.prefill(java.util.Arrays.copyOf(tokens, tokens.length - 1), startPosition);
      inner.forward(tokens[tokens.length - 1], startPosition + tokens.length - 1);
      return logits;
    }

    @Override
    public float[] forward(int token, int position) {
      return inner.forward(token, position);
    }

    @Override
    public int checkpoint() {
      return inner.checkpoint();
    }

    @Override
    public void rewind(int checkpoint) {
      inner.rewind(checkpoint);
    }

    @Override
    public void close() {
      inner.close();
    }
  }
}
