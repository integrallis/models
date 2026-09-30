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

import java.util.List;
import java.util.Map;

/**
 * One generated output.
 *
 * @param tokens generated token ids, excluding the end-of-generation token
 * @param text decoded text of {@code tokens}
 * @param truncated whether the token cap was reached
 * @param stoppedOnEndOfGeneration whether an end-of-generation token ended decoding
 * @param tokenLogProbabilities per generated token, log-probability under the producing
 *     distribution at T = 1 (the member for single arms, the fused distribution for fusion)
 * @param prefillNanos wall time of the prompt prefill (all members)
 * @param wallNanos wall time of the whole decode including prefill
 * @param memberForwardNanos per member, summed prefill and forward time on its worker
 * @param agreement G3 counters; {@code null} for single-member decoding
 * @param tokenLog sampled per-token log; empty for single-member decoding
 */
public record DecodeResult(
    int[] tokens,
    String text,
    boolean truncated,
    boolean stoppedOnEndOfGeneration,
    double[] tokenLogProbabilities,
    long prefillNanos,
    long wallNanos,
    Map<String, Long> memberForwardNanos,
    AgreementStats agreement,
    List<TokenLogEntry> tokenLog) {

  public DecodeResult {
    tokens = tokens.clone();
    tokenLogProbabilities = tokenLogProbabilities.clone();
    memberForwardNanos = Map.copyOf(memberForwardNanos);
    tokenLog = List.copyOf(tokenLog);
  }

  @Override
  public int[] tokens() {
    return tokens.clone();
  }

  @Override
  public double[] tokenLogProbabilities() {
    return tokenLogProbabilities.clone();
  }
}
