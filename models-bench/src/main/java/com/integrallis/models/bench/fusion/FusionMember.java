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

import com.integrallis.models.api.Tokenizer;

/**
 * One independently loaded model taking part in an arm: its own weights and its own KV lineage.
 *
 * <p>Implementations are driven by one thread at a time.
 */
public interface FusionMember extends AutoCloseable {

  /** Member name used in arm specs and reports. */
  String name();

  /** The member's tokenizer; fused members must have identical tokenizers (gate G0). */
  Tokenizer tokenizer();

  /** Clears the KV lineage so the next position is zero. */
  void reset();

  /** Prefills tokens at {@code startPosition} and returns stable logits for the last one. */
  float[] prefill(int[] tokens, int startPosition);

  /** Runs one token at {@code position} and returns stable logits. */
  float[] forward(int token, int position);

  /** Next token position of the KV lineage. */
  int checkpoint();

  /** Discards KV state at and after {@code checkpoint}. */
  void rewind(int checkpoint);

  /**
   * Consumes {@code tokens} from {@code startPosition} and returns the logits after each token, or
   * {@code null} when the backend cannot return all positions from one call.
   */
  default float[][] verify(int[] tokens, int startPosition) {
    return null;
  }

  @Override
  void close();
}
