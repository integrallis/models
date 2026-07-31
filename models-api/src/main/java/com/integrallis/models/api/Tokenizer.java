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
 * Converts model text and trusted template controls to token IDs and back.
 *
 * <p>Tokenizer instances returned by a backend are read-only and safe for concurrent encode/decode
 * calls. Inputs and returned strings/arrays are non-null. Returned arrays are caller-owned, and
 * implementations do not mutate array inputs.
 */
public interface Tokenizer {

  /**
   * Encodes ordinary text into token IDs.
   *
   * <p>Text that spells a registered special token is treated as ordinary input. Use {@link
   * #encode(ModelPrompt)} when applying a model template with trusted control-token segments.
   */
  int[] encode(String text);

  /**
   * Encodes a segmented model prompt, recognizing special tokens only in control segments.
   *
   * <p>Tokenizers without a special-token vocabulary may inherit the flattening implementation. A
   * tokenizer with special tokens must override this method.
   */
  default int[] encode(ModelPrompt prompt) {
    return encode(Objects.requireNonNull(prompt, "prompt").text());
  }

  /** Encodes trusted template text in which registered tokenizer control tokens are recognized. */
  default int[] encodeControl(String text) {
    return encode(ModelPrompt.control(Objects.requireNonNull(text, "text")));
  }

  /**
   * Decodes token IDs in sequence order.
   *
   * @param tokens token IDs in {@code [0, vocabSize())}
   * @return decoded text
   */
  String decode(int[] tokens);

  /**
   * Decodes one token ID.
   *
   * @param token token ID in {@code [0, vocabSize())}
   * @return decoded token text, which may be empty
   */
  String decode(int token);

  /** Returns the vocabulary size. */
  int vocabSize();

  /** Returns the beginning-of-sequence token ID. */
  int bosToken();

  /** Returns the end-of-sequence token ID. */
  int eosToken();

  /**
   * Returns whether a token ends generation.
   *
   * <p>Modern chat tokenizers can define several terminal tokens, such as end-of-sequence,
   * end-of-turn, and end-of-message. Implementations that only have one terminal token inherit the
   * end-of-sequence behavior.
   */
  default boolean isEndOfGeneration(int token) {
    return token == eosToken();
  }
}
