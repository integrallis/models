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
package com.integrallis.models.audio;

import com.integrallis.models.api.BackendDiagnostics;
import java.util.Objects;

interface SopranoEngine extends AutoCloseable {

  final class Step {
    private final float[] logits;
    private final float[] hiddenState;

    Step(float[] logits, float[] hiddenState) {
      this.logits = Objects.requireNonNull(logits, "logits");
      this.hiddenState = Objects.requireNonNull(hiddenState, "hiddenState");
    }

    float[] logits() {
      return logits;
    }

    float[] hiddenState() {
      return hiddenState;
    }
  }

  int[] encodePrompt(String text);

  Step begin(int[] prompt);

  Step advance(int token);

  float[] decode(float[] features, int frames);

  String modelName();

  BackendDiagnostics diagnostics();

  int eosToken();

  int vocabularySize();

  int hiddenSize();

  int contextLength();

  int checkpoint();

  int sampleRate();

  int samplesPerToken();

  @Override
  void close();
}
