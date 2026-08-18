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

/**
 * Stateful token-level constraint for generation.
 *
 * <p>Constraints are consulted before sampling and advanced only after a generated token is
 * accepted. Implementations are request-scoped and need not be thread-safe.
 */
public interface TokenConstraint {

  /** Returns whether {@code token} is valid as the next generated token. */
  boolean allows(int token);

  /** Advances this constraint after {@code token} was generated and accepted. */
  void accept(int token);

  /** Returns whether generation can stop without emitting another token. */
  default boolean isComplete() {
    return false;
  }

  /** Returns a constraint that accepts every generated token and never completes early. */
  static TokenConstraint unrestricted() {
    return UnrestrictedTokenConstraint.INSTANCE;
  }

  enum UnrestrictedTokenConstraint implements TokenConstraint {
    INSTANCE;

    @Override
    public boolean allows(int token) {
      return true;
    }

    @Override
    public void accept(int token) {}
  }
}
