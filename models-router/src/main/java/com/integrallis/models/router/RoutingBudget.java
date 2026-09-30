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
package com.integrallis.models.router;

import java.math.BigDecimal;
import java.util.Objects;

/**
 * Thread-safe shared spending account in one application-selected currency. Reuse an account across
 * a session, tenant or process. Candidate prices and this account must use the same currency.
 * Unknown usage is charged at the reserved bound. This is an in-process ledger, not a payment API.
 */
public final class RoutingBudget {
  private final BigDecimal limit;
  private BigDecimal spent = BigDecimal.ZERO;
  private BigDecimal reserved = BigDecimal.ZERO;

  public RoutingBudget(BigDecimal limit) {
    this.limit = nonNegative(limit, "limit");
  }

  public synchronized Snapshot snapshot() {
    return new Snapshot(limit, spent, reserved);
  }

  synchronized Reservation reserve(BigDecimal amount) {
    nonNegative(amount, "amount");
    if (spent.add(reserved).add(amount).compareTo(limit) > 0) {
      throw new RoutingBudgetExceededException("insufficient routing budget");
    }
    reserved = reserved.add(amount);
    return new Reservation(amount);
  }

  static BigDecimal cost(ModelCandidate candidate, long input, long output) {
    return BigDecimal.valueOf(candidate.costPerMillionInputTokens())
        .multiply(BigDecimal.valueOf(input))
        .add(
            BigDecimal.valueOf(candidate.costPerMillionOutputTokens())
                .multiply(BigDecimal.valueOf(output)))
        .movePointLeft(6);
  }

  private static BigDecimal nonNegative(BigDecimal value, String name) {
    Objects.requireNonNull(value, name);
    if (value.signum() < 0) throw new IllegalArgumentException(name + " must not be negative");
    return value;
  }

  public record Snapshot(BigDecimal limit, BigDecimal spent, BigDecimal reserved) {
    public BigDecimal available() {
      return limit.subtract(spent).subtract(reserved);
    }
  }

  final class Reservation {
    private final BigDecimal amount;
    private boolean settled;

    Reservation(BigDecimal amount) {
      this.amount = amount;
    }

    void settle(BigDecimal actual) {
      synchronized (RoutingBudget.this) {
        if (settled) return;
        settled = true;
        reserved = reserved.subtract(amount);
        spent = spent.add(actual == null ? amount : actual);
      }
    }

    void release() {
      settle(BigDecimal.ZERO);
    }
  }
}
