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

/**
 * How per-position hidden states are reduced to one sentence vector.
 *
 * <p>A property of the embedding model, not of the forward pass: the same weights produce different
 * vectors under different pooling, and using the wrong one silently degrades retrieval rather than
 * failing. The choice must therefore travel with the model, not be guessed by the caller.
 */
public enum Pooling {

  /**
   * Take the final position's state.
   *
   * <p>Correct for causal decoder-only embedding models, where only the last position has attended
   * to the whole input. Qwen3-Embedding and most GGUF embedding models built on generative
   * architectures use this.
   */
  LAST_TOKEN,

  /**
   * Average across every position.
   *
   * <p>Correct for encoder models with bidirectional attention, where every position sees the whole
   * input. Costs a hidden state per token rather than just the last.
   */
  MEAN
}
