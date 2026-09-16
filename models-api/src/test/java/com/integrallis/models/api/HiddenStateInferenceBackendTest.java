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
package com.integrallis.models.api;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class HiddenStateInferenceBackendTest {

  @Test
  void defaultPrefillReturnsTheFinalHiddenStateWithoutRequestingLogits() {
    StubBackend backend = new StubBackend();

    float[] hidden = backend.prefillHiddenState(new int[] {2, 3, 5}, 0);

    assertThat(hidden).containsExactly(5, 2);
    assertThat(backend.position).isEqualTo(3);
  }

  private static final class StubBackend implements HiddenStateInferenceBackend {
    private int position;

    @Override
    public boolean supportsHiddenState() {
      return true;
    }

    @Override
    public float[] forwardHiddenState(int token, int position) {
      this.position = position + 1;
      return new float[] {token, position};
    }

    @Override
    public String name() {
      return "stub";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("stub", "stub", 12, 12, 2, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      throw new UnsupportedOperationException();
    }

    @Override
    public float[] forward(int token, int position) {
      throw new AssertionError("hidden-state prefill must not project logits");
    }

    @Override
    public void close() {}
  }
}
