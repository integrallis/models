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

import java.util.Objects;

/** Optional trained heads exposed by a text-generation model in addition to token generation. */
public interface AuxiliaryTextGenerationModel extends TextGenerationModel {

  /** Whether this model can encode text with its own contrastive head. */
  boolean supportsContrastiveEncoding();

  /** Fixed output dimension of the contrastive head. */
  int contrastiveDimension();

  /** Encodes one segmented sequence with the model's trained contrastive head. */
  float[] encodeContrastive(ModelPrompt prompt);

  /** Encodes ordinary text with the model's trained contrastive head. */
  default float[] encodeContrastive(String text) {
    return encodeContrastive(ModelPrompt.text(Objects.requireNonNull(text, "text")));
  }

  /** Whether this model carries a calibrated post-hoc confidence head. */
  boolean supportsConfidenceScoring();

  /** Scores one complete segmented prompt-and-output sequence. */
  float scoreConfidence(ModelPrompt sequence);

  /** Scores one complete ordinary-text sequence. */
  default float scoreConfidence(String sequence) {
    return scoreConfidence(ModelPrompt.text(Objects.requireNonNull(sequence, "sequence")));
  }
}
