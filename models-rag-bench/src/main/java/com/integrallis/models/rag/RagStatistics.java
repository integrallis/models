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
import java.util.Map;
import java.util.Objects;
import java.util.function.ToDoubleFunction;

/** Aggregates measured RAG trials without hiding failed attempts. */
public final class RagStatistics {
  private RagStatistics() {}

  public static RagBenchmarkSummary summarize(
      List<RagRun> runs, int totalAttempts, Map<String, RagCase> casesById) {
    Objects.requireNonNull(runs, "runs");
    Objects.requireNonNull(casesById, "casesById");
    if (totalAttempts < runs.size() || totalAttempts < 1) {
      throw new IllegalArgumentException("totalAttempts must include every successful run");
    }

    List<RagRun> answerable =
        runs.stream().filter(run -> caseFor(run, casesById).answerable()).toList();
    List<RagRun> unanswerable =
        runs.stream().filter(run -> !caseFor(run, casesById).answerable()).toList();

    return new RagBenchmarkSummary(
        totalAttempts,
        runs.size(),
        maximum(runs, run -> run.generation().loadMillis()),
        percentiles(runs, RagRun::retrievalMillis),
        percentiles(runs, RagRun::frameworkOverheadMillis),
        percentiles(runs, run -> run.generation().ttftMillis()),
        percentiles(runs, run -> run.generation().tpotMillis()),
        percentiles(runs, RagRun::endToEndMillis),
        percentile(runs, run -> run.generation().prefillTokensPerSecond(), 0.50),
        percentile(runs, run -> run.generation().decodeTokensPerSecond(), 0.50),
        runs.stream().mapToLong(run -> run.generation().peakRssBytes()).max().orElse(0),
        runs.stream().mapToDouble(run -> run.generation().cpuMillis()).sum(),
        average(answerable, run -> run.evaluation().retrievalRecall()),
        average(answerable, run -> run.evaluation().reciprocalRank()),
        average(answerable, run -> run.evaluation().factCoverage()),
        average(answerable, run -> run.evaluation().citationRecall()),
        average(answerable, run -> run.evaluation().citationPrecision()),
        average(unanswerable, run -> run.evaluation().correct() ? 1 : 0),
        average(runs, run -> run.evaluation().correct() ? 1 : 0));
  }

  private static RagCase caseFor(RagRun run, Map<String, RagCase> casesById) {
    RagCase testCase = casesById.get(run.caseId());
    if (testCase == null) {
      throw new IllegalArgumentException("missing case metadata: " + run.caseId());
    }
    return testCase;
  }

  private static LatencyPercentiles percentiles(List<RagRun> runs, ToDoubleFunction<RagRun> value) {
    return new LatencyPercentiles(percentile(runs, value, 0.50), percentile(runs, value, 0.95));
  }

  private static double percentile(
      List<RagRun> runs, ToDoubleFunction<RagRun> value, double percentile) {
    double[] sorted = runs.stream().mapToDouble(value).sorted().toArray();
    if (sorted.length == 0) {
      return 0;
    }
    double index = percentile * (sorted.length - 1);
    int lower = (int) Math.floor(index);
    int upper = (int) Math.ceil(index);
    if (lower == upper) {
      return sorted[lower];
    }
    double fraction = index - lower;
    return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
  }

  private static double average(List<RagRun> runs, ToDoubleFunction<RagRun> value) {
    return runs.stream().mapToDouble(value).average().orElse(0);
  }

  private static double maximum(List<RagRun> runs, ToDoubleFunction<RagRun> value) {
    return runs.stream().mapToDouble(value).max().orElse(0);
  }
}
