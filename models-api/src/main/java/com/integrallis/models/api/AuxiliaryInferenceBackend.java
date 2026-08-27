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

/** Optional in-model heads that complement autoregressive generation. */
public interface AuxiliaryInferenceBackend extends InferenceBackend {

  /** Whether this model can encode sequences with its trained contrastive head. */
  boolean supportsContrastiveEncoding();

  /** Dimension produced by {@link #encodeContrastive(int[])}. */
  int contrastiveDimension();

  /** Encodes one complete token sequence with the model's trained contrastive head. */
  float[] encodeContrastive(int[] tokens);

  /** Whether this model carries a calibrated post-hoc confidence head. */
  boolean supportsConfidenceScoring();

  /** Returns the confidence probability for one complete prompt-and-output token sequence. */
  float scoreConfidence(int[] tokens);
}
