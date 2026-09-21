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
package com.integrallis.models.decisions;

import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Runs a prepared JevBench cohort through the shared-prefix candidate evaluator.
 *
 * <p>Reads the flat task file, scores every candidate against a state evaluated once, and writes
 * one line of label probabilities per task. Nothing is generated: the model is asked for
 * distributions over candidate tokens, so the probabilities are native in the sense the harness
 * means.
 *
 * <p>Every answer space is built as a {@code Choice} over the task's own labels, whatever the
 * question type. The label set is what the harness scores against, and a Choice carries it exactly;
 * the ordinal reading of a score question is recovered by the harness from the label values.
 */
public final class JevBenchRunner {

  private JevBenchRunner() {}

  /** Arguments: model path, prepared TSV, output TSV, temperature. */
  public static void main(String[] args) throws IOException {
    Path model = Path.of(args[0]);
    Path tasks = Path.of(args[1]);
    Path out = Path.of(args[2]);
    double temperature = args.length > 3 ? Double.parseDouble(args[3]) : 1.0;

    List<String[]> rows = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(tasks, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (!line.isBlank()) {
          rows.add(line.split("\t", -1));
        }
      }
    }
    System.out.printf("tasks=%d temperature=%.3f%n", rows.size(), temperature);

    long wall0 = System.nanoTime();
    int done = 0;
    int sharedProven = 0;
    long candidateTokens = 0;

    try (PureJavaBackend backend =
            PureJavaBackend.load(model, RustGgufBatchedMatrixKernel.openBundled());
        BufferedWriter writer = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {

      SharedPrefixCandidateEvaluator evaluator =
          new SharedPrefixCandidateEvaluator(backend, temperature);

      for (String[] row : rows) {
        String id = row[0];
        List<String> labels = List.of(row[2].split("\\|", -1));
        String prompt = row[3].replace("\\n", "\n");

        int[] state = backend.tokenizer().encode(prompt);
        List<int[]> candidates = new ArrayList<>(labels.size());
        for (String label : labels) {
          candidates.add(backend.tokenizer().encode(" " + label));
        }

        long t0 = System.nanoTime();
        CandidateBatch batch = evaluator.evaluateBatch(state, new Choice(id, labels), candidates);
        double latency = (System.nanoTime() - t0) / 1e9;

        candidateTokens += batch.candidateTokensEvaluated();
        sharedProven += batch.sharedPrefixProven() ? 1 : 0;

        StringBuilder sb = new StringBuilder(id).append('\t').append(latency);
        double[] p = batch.verdict().probabilities();
        for (int i = 0; i < labels.size(); i++) {
          sb.append('\t').append(labels.get(i)).append('=').append(p[i]);
        }
        writer.write(sb.toString());
        writer.newLine();

        if (++done % 10 == 0) {
          writer.flush();
          System.out.printf(
              "  %d/%d  %.1f s%n", done, rows.size(), (System.nanoTime() - wall0) / 1e9);
        }
      }
    }
    System.out.printf(
        "done: %d tasks, wall %.1f s, candidate-token forwards %d, prefix sharing proven on %d/%d%n",
        done, (System.nanoTime() - wall0) / 1e9, candidateTokens, sharedProven, done);
  }
}
