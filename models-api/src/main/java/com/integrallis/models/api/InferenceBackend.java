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

/** SPI interface for inference backends capable of running a model's forward pass. */
public interface InferenceBackend extends AutoCloseable {

  /** Returns the backend implementation name. */
  String name();

  /** Returns metadata about the loaded model. */
  ModelMetadata metadata();

  /** Returns the tokenizer for this model. */
  Tokenizer tokenizer();

  /** Runs a single forward pass for the given token at the given position, returning logits. */
  float[] forward(int token, int position);

  /**
   * Clears request-specific state before a new generation. Stateless backends may keep the default
   * no-op implementation.
   */
  default void reset() {}

  @Override
  void close();
}
