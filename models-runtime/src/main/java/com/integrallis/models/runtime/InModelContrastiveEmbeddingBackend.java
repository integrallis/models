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
package com.integrallis.models.runtime;

import com.integrallis.models.api.AuxiliaryInferenceBackend;
import com.integrallis.models.api.EmbeddingBackend;
import com.integrallis.models.api.Tokenizer;
import java.util.Objects;

/** Exposes a generative model's trained contrastive head through the embedding SPI. */
public final class InModelContrastiveEmbeddingBackend implements EmbeddingBackend {

  private final AuxiliaryInferenceBackend model;
  private final Tokenizer tokenizer;

  public InModelContrastiveEmbeddingBackend(AuxiliaryInferenceBackend model) {
    this.model = Objects.requireNonNull(model, "model");
    if (!model.supportsContrastiveEncoding()) {
      throw new IllegalArgumentException("model does not expose a contrastive encoding head");
    }
    this.tokenizer = model.tokenizer();
  }

  @Override
  public int dimension() {
    return model.contrastiveDimension();
  }

  @Override
  public float[] embed(String text) {
    return model.encodeContrastive(tokenizer.encode(Objects.requireNonNull(text, "text")));
  }

  /** The adapter borrows the model and therefore does not close it. */
  @Override
  public void close() {}
}
