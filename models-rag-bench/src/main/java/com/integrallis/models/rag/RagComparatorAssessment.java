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
package com.integrallis.models.rag;

import java.util.List;

/**
 * Relative performance of a Models run against one same-host local inference engine.
 *
 * <p>Every ratio here is computed from timings taken by this harness on both arms. None is taken
 * from an engine's self-reported counters. That distinction is load-bearing: Ollama's {@code
 * prompt_eval_count / prompt_eval_duration} reports the whole prompt over the duration of only the
 * tokens left after prefix-cache reuse, which on 14 of 31 certified models implies a throughput
 * above the host's peak FLOP rate. A ratio built on it measures caching policy, not speed. See
 * {@code benchmark-results/2026-10-08-prefill-metric-audit/}.
 *
 * <p>{@code timeToFirstTokenRatio} is reported but does not gate. It is the metric a user actually
 * perceives, and it is wall clock on our own timer for both arms, so it is the right candidate for
 * a future gate; the measured p95 distribution across the certified catalogue is 1.66x median, so a
 * ceiling cannot be chosen without deciding which entries may drop.
 */
public record RagComparatorAssessment(
    String comparatorBackend,
    double decodeThroughputRatio,
    double endToEndLatencyRatio,
    double timeToFirstTokenRatio,
    double minimumDecodeThroughputRatio,
    double maximumEndToEndLatencyRatio,
    boolean qualified,
    List<String> failures) {
  public RagComparatorAssessment {
    failures = List.copyOf(failures);
  }
}
