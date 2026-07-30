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

/** Tokenizer interface for encoding text to token IDs and decoding back. */
public interface Tokenizer {

  /**
   * Encodes text into token IDs, recognizing any special-token text defined by the tokenizer.
   *
   * <p>Callers should use this method for trusted, fully rendered prompts. Use {@link
   * #encodeOrdinary(String)} for untrusted user content that must not be interpreted as control
   * tokens.
   */
  int[] encode(String text);

  /**
   * Encodes ordinary text without interpreting embedded text as tokenizer control tokens.
   *
   * <p>Tokenizers without a special-token vocabulary can inherit this implementation.
   */
  default int[] encodeOrdinary(String text) {
    return encode(text);
  }

  /** Decodes an array of token IDs back into a string. */
  String decode(int[] tokens);

  /** Decodes a single token ID to its string representation. */
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
