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
package com.integrallis.models.bench;

import java.util.Locale;

/** Stable Markdown renderer for documentation comparison tables. */
public final class ComparisonMarkdown {

  private ComparisonMarkdown() {}

  public static String render(ComparisonReport report) {
    StringBuilder markdown = new StringBuilder();
    markdown
        .append("| Backend | Version | Load ms | p95 TTFT ms | p95 TPOT ms | Prefill tok/s | ")
        .append(
            "Decode tok/s | Peak RSS GiB | vs llama.cpp | Output match | Latency tier | Relative |\n")
        .append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n");
    for (BackendComparison backend : report.backends()) {
      markdown.append(
          String.format(
              Locale.ROOT,
              "| %s | %s | %.1f | %.1f | %.1f | %.2f | %.2f | %.2f | %.1f%% | %.1f%% | %s | %s |%n",
              backend.backend(),
              backend.backendVersion(),
              backend.loadMillis(),
              backend.p95TtftMillis(),
              backend.p95TpotMillis(),
              backend.prefillTokensPerSecond(),
              backend.decodeTokensPerSecond(),
              backend.peakRssBytes() / (double) (1L << 30),
              backend.decodeRatioToLlamaCpp() * 100,
              backend.outputMatchRateToLlamaCpp() * 100,
              backend.performanceTier(),
              backend.relativePerformance()));
    }
    return markdown.toString();
  }
}
