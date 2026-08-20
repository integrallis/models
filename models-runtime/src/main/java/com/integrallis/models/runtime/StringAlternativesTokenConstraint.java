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

import com.integrallis.models.api.Tokenizer;
import java.util.List;
import java.util.Objects;

/** Constrains generation to one of a finite set of decoded strings. */
public final class StringAlternativesTokenConstraint implements TokenConstraint {

  private final Tokenizer tokenizer;
  private final List<String> alternatives;
  private final StringBuilder generated = new StringBuilder();

  public StringAlternativesTokenConstraint(Tokenizer tokenizer, List<String> alternatives) {
    this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    this.alternatives = List.copyOf(Objects.requireNonNull(alternatives, "alternatives"));
    if (this.alternatives.isEmpty()) {
      throw new IllegalArgumentException("alternatives must not be empty");
    }
    for (String alternative : this.alternatives) {
      if (alternative == null || alternative.isEmpty()) {
        throw new IllegalArgumentException("alternatives must not contain null or empty strings");
      }
    }
  }

  @Override
  public boolean allows(int token) {
    if (isComplete()) {
      return false;
    }
    String fragment = tokenizer.decode(token);
    if (fragment.isEmpty()) {
      return false;
    }
    String candidate = generated + fragment;
    for (String alternative : alternatives) {
      if (alternative.startsWith(candidate)) {
        return true;
      }
    }
    return false;
  }

  @Override
  public void accept(int token) {
    if (!allows(token)) {
      throw new IllegalArgumentException("token " + token + " violates the string alternatives");
    }
    generated.append(tokenizer.decode(token));
  }

  @Override
  public boolean isComplete() {
    for (String alternative : alternatives) {
      if (alternative.contentEquals(generated)) {
        return true;
      }
    }
    return false;
  }
}
