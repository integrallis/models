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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.decisions.Choice;
import com.integrallis.models.decisions.LetterLogitScorer;
import com.integrallis.models.decisions.Verdict;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Runs a labelled triage corpus through the typed-decision readout and reports accuracy and
 * per-decision latency.
 *
 * <p>Exists to answer a comparison rather than to produce a number in isolation: external work
 * publishes category accuracy and seconds-per-message for this shape of task over public
 * customer-support corpora, through an HTTP server that returns log-probabilities. This runs the
 * same corpus, the same outcomes and the same rules in-process, so the two can be put side by side
 * without either side's harness being the variable.
 *
 * <p>What it does <b>not</b> do is claim the result transfers. Point it at the corpus the other
 * numbers were measured on, or the comparison is between two datasets.
 *
 * <pre>{@code
 * ./gradlew :models-bench:run -PmainClass=...TriageDecisionBenchCli --args="\
 *   --model /path/to/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
 *   --messages corpus/messages.tsv --rules corpus/rules.tsv --gate 0.9 --output out.json"
 * }</pre>
 */
final class TriageDecisionBenchCli {

  private static final Set<String> OPTIONS =
      Set.of("model", "messages", "rules", "question", "limit", "gate", "temperature", "output");

  private static final String DEFAULT_QUESTION = "Which team should handle this customer message?";

  private TriageDecisionBenchCli() {}

  static void run(String[] args) throws Exception {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    Path model = requirePath(values, "model");
    TriageTask task =
        TriageTask.load(
            requirePath(values, "messages"),
            requirePath(values, "rules"),
            values.getOrDefault("question", DEFAULT_QUESTION),
            BenchmarkCliArguments.integer(values, "limit", 0));
    double gate = doubleValue(values, "gate", 0.9);
    double temperature = doubleValue(values, "temperature", 1.0);

    Choice space = task.space();
    // Rubric before the criterion, letters after it. MEASURED over 120 JevBench items: that layout
    // scored 0.9000 where no rubric scored 0.7500 and rubric-and-letters-together scored 0.8000
    // (LetterLogitScorer#renderCriteria). The layout is part of the measurement, not formatting.
    String rubric = LetterLogitScorer.renderCriteria(task.outcomes(), task.criteria());
    String tail = task.question() + "\n" + LetterLogitScorer.renderOptions(task.outcomes());

    long openedAt = System.nanoTime();
    try (InferenceBackend backend = PureJavaModelSource.resolve(model.toString()).load()) {
      double openMs = millis(openedAt);
      Tokenizer tokenizer = backend.tokenizer();
      int[] letterTokens = resolveLetterTokens(tokenizer, task.outcomes().size());
      LetterLogitScorer scorer = new LetterLogitScorer(temperature);

      // Print the route rather than inferring it: the same artifact can resolve to different
      // kernels, and a latency figure whose backend is unnamed is not comparable to anything.
      System.out.printf(
          "model=%s backend=%s outcomes=%d cases=%d gate=%.2f temperature=%.3f%n",
          model.getFileName(),
          backend.name(),
          task.outcomes().size(),
          task.messages().size(),
          gate,
          temperature);
      System.out.printf("open: %.0f ms%n", openMs);

      List<String> chosen = new ArrayList<>();
      List<Double> confidence = new ArrayList<>();
      List<Double> latencies = new ArrayList<>();
      double firstMs = Double.NaN;
      for (TriageTask.Message message : task.messages()) {
        String prompt = message.text() + "\n\n" + rubric + "\n\n" + tail;
        int[] tokens = tokenizer.encode(prompt);
        backend.reset();
        long at = System.nanoTime();
        float[] logits = backend.prefill(tokens, 0);
        Verdict verdict = scorer.score(space, logits, letterTokens);
        double took = millis(at);
        if (Double.isNaN(firstMs)) {
          // Reported separately because it carries JIT warm-up; folding it into the distribution
          // would make a cold run look like a slow model.
          firstMs = took;
        } else {
          latencies.add(took);
        }
        chosen.add(verdict.winner());
        confidence.add(verdict.confidence());
      }

      TriageTask.Scored scored = task.score(chosen, confidence, gate);
      report(scored, openMs, firstMs, latencies);
      String output = values.get("output");
      if (output != null) {
        Path destination = Path.of(output);
        Files.writeString(
            destination,
            json(model, backend.name(), task, scored, openMs, firstMs, latencies, temperature),
            StandardCharsets.UTF_8);
        System.out.println("wrote " + destination.toAbsolutePath());
      }
    }
  }

  /**
   * The token id of each answer letter, verified to exist in this vocabulary.
   *
   * <p>A letter resolved to the wrong id reads the wrong logit and fails silently -- the run
   * completes and the accuracy is simply wrong -- so an unresolvable letter is fatal here. The
   * leading-space forms are tried because families disagree: SentencePiece vocabularies carry
   * {@code U+2581} before a word-initial token and byte-level BPE carries {@code U+0120}, and which
   * one a model uses cannot be assumed from its name.
   */
  private static int[] resolveLetterTokens(Tokenizer tokenizer, int outcomes) {
    int[] letters = new int[outcomes];
    for (int index = 0; index < outcomes; index++) {
      String letter = String.valueOf((char) ('A' + index));
      int id = firstResolvable(tokenizer, letter, "▁" + letter, "Ġ" + letter);
      if (id < 0) {
        throw new IllegalStateException(
            "this vocabulary has no single token for answer letter '"
                + letter
                + "'; the decision readout would score the wrong logit");
      }
      letters[index] = id;
    }
    return letters;
  }

  private static int firstResolvable(Tokenizer tokenizer, String... candidates) {
    for (String candidate : candidates) {
      int id = tokenizer.tokenId(candidate);
      if (id >= 0) {
        return id;
      }
    }
    return -1;
  }

  private static void report(
      TriageTask.Scored scored, double openMs, double firstMs, List<Double> latencies) {
    double[] sorted = latencies.stream().mapToDouble(Double::doubleValue).sorted().toArray();
    System.out.printf(
        "first decision: %.0f ms (includes JIT warm-up)%nwarm: p50 %.0f ms, p95 %.0f ms, mean %.0f ms%n",
        firstMs,
        percentile(sorted, 0.50),
        percentile(sorted, 0.95),
        Arrays.stream(sorted).average().orElse(Double.NaN));
    System.out.printf("accuracy: %.4f over %d cases%n", scored.accuracy(), scored.cases());
    System.out.printf(
        "at confidence >= %.2f: decided %.1f%% of cases, %.4f accurate%n",
        scored.gate(), 100.0 * scored.coverageAtGate(), scored.accuracyAtGate());
    System.out.println("per outcome (correct/seen):");
    scored
        .perOutcome()
        .forEach(
            (outcome, tally) -> System.out.printf("  %-16s %d/%d%n", outcome, tally[0], tally[1]));
    if (!scored.confusions().isEmpty()) {
      System.out.println("confusions:");
      scored.confusions().entrySet().stream()
          .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
          .limit(10)
          .forEach(entry -> System.out.printf("  %-34s %d%n", entry.getKey(), entry.getValue()));
    }
  }

  private static String json(
      Path model,
      String backend,
      TriageTask task,
      TriageTask.Scored scored,
      double openMs,
      double firstMs,
      List<Double> latencies,
      double temperature) {
    double[] sorted = latencies.stream().mapToDouble(Double::doubleValue).sorted().toArray();
    StringBuilder out = new StringBuilder("{\n");
    out.append("  \"model\": \"").append(model.getFileName()).append("\",\n");
    out.append("  \"backend\": \"").append(backend).append("\",\n");
    out.append("  \"question\": \"").append(task.question().replace("\"", "'")).append("\",\n");
    out.append("  \"outcomes\": ").append(task.outcomes().size()).append(",\n");
    out.append("  \"cases\": ").append(scored.cases()).append(",\n");
    out.append(String.format(Locale.ROOT, "  \"temperature\": %.4f,%n", temperature));
    out.append(String.format(Locale.ROOT, "  \"accuracy\": %.6f,%n", scored.accuracy()));
    out.append(String.format(Locale.ROOT, "  \"gate\": %.4f,%n", scored.gate()));
    out.append(
        String.format(Locale.ROOT, "  \"coverageAtGate\": %.6f,%n", scored.coverageAtGate()));
    out.append(
        String.format(Locale.ROOT, "  \"accuracyAtGate\": %.6f,%n", scored.accuracyAtGate()));
    out.append(String.format(Locale.ROOT, "  \"openMs\": %.1f,%n", openMs));
    out.append(String.format(Locale.ROOT, "  \"firstDecisionMs\": %.1f,%n", firstMs));
    out.append(String.format(Locale.ROOT, "  \"warmP50Ms\": %.2f,%n", percentile(sorted, 0.50)));
    out.append(String.format(Locale.ROOT, "  \"warmP95Ms\": %.2f,%n", percentile(sorted, 0.95)));
    out.append(
        String.format(
            Locale.ROOT,
            "  \"warmMeanMs\": %.2f%n",
            Arrays.stream(sorted).average().orElse(Double.NaN)));
    return out.append("}\n").toString();
  }

  private static double percentile(double[] sorted, double quantile) {
    if (sorted.length == 0) {
      return Double.NaN;
    }
    int index = (int) Math.ceil(quantile * sorted.length) - 1;
    return sorted[Math.max(0, Math.min(sorted.length - 1, index))];
  }

  private static double millis(long startNanos) {
    return (System.nanoTime() - startNanos) / 1_000_000.0;
  }

  private static Path requirePath(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return Path.of(value);
  }

  private static double doubleValue(Map<String, String> values, String name, double fallback) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      return fallback;
    }
    try {
      return Double.parseDouble(value);
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("--" + name + " must be a number: " + value, failure);
    }
  }
}
