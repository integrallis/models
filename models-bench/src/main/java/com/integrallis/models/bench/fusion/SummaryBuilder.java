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

import com.integrallis.models.bench.fusion.ReportModel.G3;
import com.integrallis.models.bench.fusion.ReportModel.Item;
import com.integrallis.models.bench.fusion.ReportModel.Output;
import com.integrallis.models.bench.fusion.ReportModel.Summary;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Per-arm summary statistics, including gate G3 counters and the G5 truncation flag. */
final class SummaryBuilder {
  static final double G5_TRUNCATION_LIMIT = 0.03;

  private SummaryBuilder() {}

  static Summary summarize(List<Item> items) {
    int correct = 0;
    int outputs = 0;
    int truncated = 0;
    int failures = 0;
    long generatedTokens = 0;
    double generationSeconds = 0;
    double core = 0;
    double reusedCore = 0;
    List<Integer> traceTokens = new ArrayList<>();
    Map<String, int[]> aggregation = new LinkedHashMap<>();
    Map<String, int[]> mismatch = new LinkedHashMap<>();
    long steps = 0;
    long agree = 0;
    long noMember = 0;
    double entropy = 0;
    Map<String, Long> equals = new LinkedHashMap<>();
    Map<String, Double> kl = new LinkedHashMap<>();
    for (Item item : items) {
      correct += item.correct() ? 1 : 0;
      failures += item.extractionFailure() ? 1 : 0;
      core += Math.max(0, item.coreSeconds());
      reusedCore += item.reusedCoreSeconds();
      item.aggregationCorrect()
          .forEach(
              (key, value) -> {
                int[] counts = aggregation.computeIfAbsent(key, k -> new int[2]);
                counts[0] += value ? 1 : 0;
                counts[1]++;
              });
      for (Output output : item.outputs()) {
        outputs++;
        truncated += output.truncated() ? 1 : 0;
        traceTokens.add(output.tokens());
        if (!output.reused()) {
          generatedTokens += output.tokens();
          generationSeconds += output.wallMillis() / 1000.0;
        }
        if (output.reasoningAnswer() != null && output.statedAnswer() != null) {
          int[] counts = mismatch.computeIfAbsent(output.producer(), k -> new int[2]);
          counts[0] += output.consistent() ? 0 : 1;
          counts[1]++;
        }
        AgreementStats stats = output.agreement();
        if (stats != null) {
          steps += stats.steps();
          agree += stats.allMembersAgree();
          noMember += stats.fusedEqualsNoMember();
          entropy += stats.sumFusedEntropy();
          stats
              .fusedEqualsMember()
              .forEach((name, count) -> equals.merge(name, (long) count, Long::sum));
          stats.sumKlFusedToMember().forEach((name, value) -> kl.merge(name, value, Double::sum));
        }
      }
    }
    Map<String, Double> aggregationAccuracy = new LinkedHashMap<>();
    aggregation.forEach((key, counts) -> aggregationAccuracy.put(key, ratio(counts[0], counts[1])));
    Map<String, Double> mismatchRate = new LinkedHashMap<>();
    mismatch.forEach((key, counts) -> mismatchRate.put(key, ratio(counts[0], counts[1])));
    G3 g3 = null;
    if (steps > 0) {
      Map<String, Double> equalsRate = new LinkedHashMap<>();
      final long totalSteps = steps;
      equals.forEach((name, count) -> equalsRate.put(name, count / (double) totalSteps));
      Map<String, Double> meanKl = new LinkedHashMap<>();
      kl.forEach((name, value) -> meanKl.put(name, value / totalSteps));
      g3 =
          new G3(
              steps,
              agree,
              agree / (double) steps,
              equalsRate,
              noMember / (double) steps,
              entropy / steps,
              meanKl,
              agree == steps);
    }
    double truncationRate = ratio(truncated, outputs);
    return new Summary(
        items.size(),
        correct,
        ratio(correct, items.size()),
        outputs,
        truncated,
        truncationRate,
        failures,
        ratio(failures, items.size()),
        generatedTokens,
        generationSeconds,
        generationSeconds > 0 ? generatedTokens / generationSeconds : 0,
        core,
        items.isEmpty() ? 0 : core / items.size(),
        reusedCore,
        percentile95(traceTokens),
        aggregationAccuracy,
        mismatchRate,
        g3,
        truncationRate > G5_TRUNCATION_LIMIT);
  }

  /** Nearest-rank 95th percentile. */
  static double percentile95(List<Integer> values) {
    if (values.isEmpty()) {
      return 0;
    }
    int[] sorted = values.stream().mapToInt(Integer::intValue).toArray();
    Arrays.sort(sorted);
    int rank = (int) Math.ceil(0.95 * sorted.length);
    return sorted[Math.max(0, rank - 1)];
  }

  private static double ratio(long numerator, long denominator) {
    return denominator == 0 ? 0 : numerator / (double) denominator;
  }
}
