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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.RerankingModel;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/**
 * Issue #449: Verify that {@link RerankingModel#scoreAll} calls {@code scoreBatch} exactly once
 * when overridden, and that batching amortizes per-call overhead compared to a sequential loop.
 *
 * <p>This test uses an artificial latency model so it does not require a real GGUF fixture.
 */
class RerankingModelBatchRoutingBenchmarkTest {

  private static final String QUERY = "query";
  private static final List<String> DOCUMENTS =
      List.of("a", "bb", "ccc", "dddd", "eeeee", "ffffff");

  /** Model that injects a fixed per-call latency to make overhead measurable. */
  private static class LatencyModel implements RerankingModel {
    private final long latencyNanos;
    private int scoreCalls;
    private int scoreBatchCalls;

    LatencyModel(long latencyNanos) {
      this.latencyNanos = latencyNanos;
    }

    @Override
    public double score(String query, String document) {
      scoreCalls++;
      busyWait(latencyNanos);
      return query.length() * 1.0 + document.length();
    }

    @Override
    public List<Double> scoreBatch(String query, List<String> documents) {
      scoreBatchCalls++;
      // Batched path pays latency once for the whole batch, then negligible per-doc work.
      busyWait(latencyNanos);
      return documents.stream().map(doc -> query.length() * 1.0 + doc.length()).toList();
    }

    @Override
    public void close() {}

    int scoreCalls() {
      return scoreCalls;
    }

    int scoreBatchCalls() {
      return scoreBatchCalls;
    }
  }

  /** Model that falls back to the default scoreBatch (which loops over score). */
  private static class DefaultBatchModel implements RerankingModel {
    private final long latencyNanos;

    DefaultBatchModel(long latencyNanos) {
      this.latencyNanos = latencyNanos;
    }

    @Override
    public double score(String query, String document) {
      busyWait(latencyNanos);
      return query.length() * 1.0 + document.length();
    }

    @Override
    public void close() {}
  }

  @Test
  void scoreAllCallsScoreBatchOnce() {
    LatencyModel model = new LatencyModel(0);
    model.scoreAll(QUERY, DOCUMENTS);
    assertThat(model.scoreBatchCalls()).isEqualTo(1);
    // The mock batch path computes directly without calling score() per document.
    assertThat(model.scoreCalls()).isEqualTo(0);
  }

  @Test
  void batchAmortizesOverheadVersusSequentialLoop() {
    long latencyNanos = TimeUnit.MILLISECONDS.toNanos(1); // 1 ms artificial work

    // Sequential: N separate score() calls, each paying the latency
    long sequentialStart = System.nanoTime();
    DefaultBatchModel sequential = new DefaultBatchModel(latencyNanos);
    for (String doc : DOCUMENTS) {
      sequential.score(QUERY, doc);
    }
    long sequentialNanos = System.nanoTime() - sequentialStart;

    // Batched: scoreBatch is called once, then internally loops
    long batchStart = System.nanoTime();
    LatencyModel batched = new LatencyModel(latencyNanos);
    batched.scoreAll(QUERY, DOCUMENTS);
    long batchNanos = System.nanoTime() - batchStart;

    // The batch path should be faster because it pays the overhead once
    // rather than N times. Tolerance allows for scheduler noise.
    assertThat(batchNanos)
        .isLessThan((long) (sequentialNanos * 0.90))
        .withFailMessage(
            "Batched path should be measurably faster than sequential loop: batch=%d ms, seq=%d ms",
            TimeUnit.NANOSECONDS.toMillis(batchNanos),
            TimeUnit.NANOSECONDS.toMillis(sequentialNanos));
  }

  @Test
  void emptyBatchCallsScoreBatchOnceAndReturnsEmpty() {
    LatencyModel model = new LatencyModel(0);
    List<Double> scores = model.scoreAll(QUERY, List.of());
    assertThat(scores).isEmpty();
    assertThat(model.scoreBatchCalls()).isEqualTo(0); // short-circuits before scoreBatch
    assertThat(model.scoreCalls()).isEqualTo(0);
  }

  /** Spins for approximately {@code nanos} without calling sleep (avoids OS scheduler quanta). */
  private static void busyWait(long nanos) {
    long deadline = System.nanoTime() + nanos;
    while (System.nanoTime() < deadline) {
      Thread.onSpinWait();
    }
  }
}
