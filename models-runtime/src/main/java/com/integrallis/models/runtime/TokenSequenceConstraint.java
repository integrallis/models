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

import java.util.Arrays;
import java.util.Objects;

/** Constrains generation to one exact token sequence. */
public final class TokenSequenceConstraint implements TokenConstraint {

  private final int[] tokens;
  private int position;

  private TokenSequenceConstraint(int[] tokens) {
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    this.tokens = tokens.clone();
  }

  /** Creates a constraint that accepts exactly {@code tokens}. */
  public static TokenSequenceConstraint of(int... tokens) {
    Objects.requireNonNull(tokens, "tokens");
    return new TokenSequenceConstraint(tokens);
  }

  @Override
  public boolean allows(int token) {
    return position < tokens.length && token == tokens[position];
  }

  @Override
  public void accept(int token) {
    if (!allows(token)) {
      String expected =
          position < tokens.length ? Integer.toString(tokens[position]) : "<complete>";
      throw new IllegalArgumentException(
          "token " + token + " does not match expected token " + expected);
    }
    position++;
  }

  @Override
  public boolean isComplete() {
    return position == tokens.length;
  }

  @Override
  public String toString() {
    return "TokenSequenceConstraint" + Arrays.toString(tokens) + "@" + position;
  }
}
