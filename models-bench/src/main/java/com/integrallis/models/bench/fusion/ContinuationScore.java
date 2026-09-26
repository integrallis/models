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

/**
 * Teacher-forced scores of one continuation under one member.
 *
 * @param tokenLogProbabilities log p(token_i | prompt, token_<i) at T = 1
 * @param tokenRawLogits the raw logit of each continuation token (for the article rule)
 * @param usedVerify whether one multi-position backend call produced the logits
 */
public record ContinuationScore(
    double[] tokenLogProbabilities, double[] tokenRawLogits, boolean usedVerify) {

  public ContinuationScore {
    tokenLogProbabilities = tokenLogProbabilities.clone();
    tokenRawLogits = tokenRawLogits.clone();
  }

  @Override
  public double[] tokenLogProbabilities() {
    return tokenLogProbabilities.clone();
  }

  @Override
  public double[] tokenRawLogits() {
    return tokenRawLogits.clone();
  }

  public double sum() {
    double total = 0;
    for (double value : tokenLogProbabilities) {
      total += value;
    }
    return total;
  }

  public double mean() {
    return tokenLogProbabilities.length == 0 ? 0 : sum() / tokenLogProbabilities.length;
  }

  public double rawSum() {
    double total = 0;
    for (double value : tokenRawLogits) {
      total += value;
    }
    return total;
  }
}
