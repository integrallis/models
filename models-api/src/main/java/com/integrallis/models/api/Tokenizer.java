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

  /** Encodes a string into an array of token IDs. */
  int[] encode(String text);

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
}
