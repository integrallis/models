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

/**
 * Stateful SPI for running one loaded model's forward pass.
 *
 * <p>Backend instances are not thread-safe. A caller must serialize {@link #forward}, {@link
 * #prefill}, {@link #reset}, rewind/verification capabilities, and {@link #close}. Begin each
 * independent sequence with {@code reset()}, use contiguous zero-based positions, and do not invoke
 * the backend after closing it. Returned metadata, diagnostics, and tokenizer references are
 * non-null.
 */
public interface InferenceBackend extends AutoCloseable {

  /** Returns the backend implementation name. */
  String name();

  /** Returns metadata about the loaded model. */
  ModelMetadata metadata();

  /** Returns the deterministic runtime decisions exposed by this backend. */
  default BackendDiagnostics diagnostics() {
    return BackendDiagnostics.unavailable(name());
  }

  /** Returns the tokenizer for this model. */
  Tokenizer tokenizer();

  /**
   * Runs a forward pass and returns a stable, caller-owned logits array.
   *
   * @param token token ID in {@code [0, metadata().vocabSize())}
   * @param position zero-based sequence position
   * @return a new or otherwise stable array of {@code metadata().vocabSize()} logits
   */
  float[] forward(int token, int position);

  /**
   * Runs a single forward pass whose returned storage may be reused by the next backend call.
   *
   * <p>The default preserves the stable snapshot returned by {@link #forward(int, int)}. Stateful
   * backends may override this method to avoid copying logits when the caller consumes them before
   * the next backend invocation. A transient result must remain valid until that next invocation.
   */
  default float[] forwardTransient(int token, int position) {
    return forward(token, position);
  }

  /**
   * Processes a contiguous prompt and returns logits for its final token.
   *
   * <p>The returned storage follows {@link #forwardTransient(int, int)} semantics. Backends with a
   * batched prompt path or the ability to suppress intermediate logits should override this method.
   * Implementations must not modify {@code tokens}.
   *
   * @param tokens nonempty token IDs in sequence order
   * @param startPosition zero-based position of the first token
   * @return logits produced after the final token
   */
  default float[] prefill(int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition < 0) {
      throw new IllegalArgumentException("startPosition must be >= 0");
    }

    float[] logits = null;
    for (int index = 0; index < tokens.length; index++) {
      logits = forwardTransient(tokens[index], Math.addExact(startPosition, index));
    }
    return logits;
  }

  /**
   * Clears request-specific state before a new generation. Stateless backends may keep the default
   * no-op implementation. After this method returns, the next forward position is zero.
   */
  default void reset() {}

  /** Releases all backend-owned resources. No other method may be called afterward. */
  @Override
  void close();
}
