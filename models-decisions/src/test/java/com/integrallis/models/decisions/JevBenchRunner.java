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

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.backend.cuda.CudaGgufBatchedMatrixKernel;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
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

    // Kernel selection is explicit and printed. An unreported fallback has invalidated three
    // measurements in this campaign, so the run says what it actually ran on.
    String requested = System.getProperty("decisions.kernel", "auto");
    GgufBatchedMatrixKernel selected;
    String kernelName;
    CudaGgufBatchedMatrixKernel.Status cuda =
        requested.equals("cuda") || requested.equals("auto")
            ? CudaGgufBatchedMatrixKernel.open()
            : null;
    if (cuda != null && cuda.kernel().isPresent()) {
      selected = cuda.kernel().orElseThrow();
      kernelName = "cuda [" + cuda.deviceName() + ", cc " + cuda.computeCapability() + "]";
    } else if (requested.equals("cuda")) {
      throw new IllegalStateException(
          "cuda requested and unavailable: " + (cuda == null ? "not attempted" : cuda.reason()));
    } else {
      selected = RustGgufBatchedMatrixKernel.openBundled();
      kernelName = "rust-ffm";
    }
    RuntimeReport runtime =
        new RuntimeReport()
            .with("model", model.getFileName().toString())
            .with(
                "mode",
                Boolean.parseBoolean(System.getProperty("decisions.letterLogits", "false"))
                    ? "letter-logits"
                    : "candidate-scoring")
            .kernel(
                kernelName.startsWith("cuda") ? "cuda" : kernelName,
                cuda != null && cuda.kernel().isPresent() ? cuda.deviceName() : "",
                cuda != null && cuda.kernel().isPresent()
                    ? "cc " + cuda.computeCapability() + ", ptx " + cuda.ptxTarget()
                    : "")
            .with("prefillBatchSize", System.getProperty("models.purejava.prefillBatchSize", "32"))
            .with(
                "maxContextLength",
                System.getProperty("models.purejava.maxContextLength", "default"));
    // A caller that demands a configuration gets it or gets an exception, never a quiet fallback.
    String demanded = System.getProperty("decisions.requireKernel", "");
    if (!demanded.isBlank()) {
      runtime.require("kernel", demanded);
    }
    System.out.println(runtime);

    try (PureJavaBackend backend = PureJavaBackend.load(model, selected);
        BufferedWriter writer = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {

      SharedPrefixCandidateEvaluator evaluator =
          new SharedPrefixCandidateEvaluator(backend, temperature);
      // The letter arm asks the model which letter comes next instead of scoring each option, so
      // one forward answers the whole question and the option text is read from the prompt.
      boolean letters = Boolean.parseBoolean(System.getProperty("decisions.letterLogits", "false"));
      // SemIf's arm (MIT, github.com/TheoLeeCJ/openjev) scores the same frozen Qwen3.5-4B 9
      // Intelligence points above ours. Two differences are testable here: it applies the model's
      // chat template with a system instruction and a JSON payload, and its answer slots are bare
      // single-token letters rather than " A". Under -Ddecisions.rawPrompt the fourth column is
      // already the complete prompt and is fed verbatim; the option block is not appended.
      boolean rawPrompt = Boolean.parseBoolean(System.getProperty("decisions.rawPrompt", "false"));
      LetterLogitScorer letterScorer = new LetterLogitScorer(temperature);
      System.out.printf("mode=%s%n", letters ? "letter-logits" : "candidate-scoring");

      // Warmup, and it is not a nicety. Measured on an A40 host 2026-09-22: the same item scored
      // repeatedly in one JVM gives 0.9110, 0.9399, then 0.9316622781200229 and that value
      // thereafter, bit-identical and reproducible in a second JVM. Without this the first two
      // items of every run are scored on partly-interpreted code, are not reproducible, and are
      // also the slowest, which moves the p50/p95 the Speed axis is computed from.
      int warmupItems = Integer.getInteger("decisions.warmupItems", 3);
      List<String[]> schedule = new ArrayList<>();
      for (int i = 0; i < warmupItems && !rows.isEmpty(); i++) {
        schedule.add(rows.get(0));
      }
      int warmupRemaining = schedule.size();
      schedule.addAll(rows);
      if (warmupRemaining > 0) {
        System.out.printf("warmup=%d items (discarded)%n", warmupRemaining);
      }

      for (String[] row : schedule) {
        String id = row[0];
        List<String> labels = List.of(row[2].split("\\|", -1));
        String prompt = row[3].replace("\\n", "\n");

        int[] state = backend.tokenizer().encode(prompt);
        List<int[]> candidates = new ArrayList<>(labels.size());
        for (String label : labels) {
          candidates.add(backend.tokenizer().encode(" " + label));
        }

        double[] p;
        double latency;
        if (letters) {
          Choice space = new Choice(id, labels);
          int[] lettered =
              rawPrompt
                  ? backend.tokenizer().encode(prompt)
                  : backend
                      .tokenizer()
                      .encode(prompt + "\n" + LetterLogitScorer.renderOptions(labels));
          int[] letterTokens = new int[labels.size()];
          for (int i = 0; i < labels.size(); i++) {
            int[] encoded =
                backend
                    .tokenizer()
                    .encode(rawPrompt ? String.valueOf((char) ('A' + i)) : " " + (char) ('A' + i));
            letterTokens[i] = encoded[encoded.length - 1];
          }
          if (rawPrompt) {
            // SemIf refuses any slot that is not one exact round-trip token, because a multi-token
            // or merged slot silently reads the wrong logit. Fail closed rather than score it.
            for (int i = 0; i < labels.size(); i++) {
              int[] encoded = backend.tokenizer().encode(String.valueOf((char) ('A' + i)));
              if (encoded.length != 1) {
                throw new IllegalStateException(
                    "answer slot " + (char) ('A' + i) + " is not one token: " + encoded.length);
              }
            }
          }
          long t0 = System.nanoTime();
          try (InferenceSession session = backend.openSession()) {
            int last = lettered.length - 1;
            if (last > 0) {
              int[] head = new int[last];
              System.arraycopy(lettered, 0, head, 0, last);
              backend.prefill(session, head, 0);
            }
            float[] logits = backend.forward(session, lettered[last], last);
            p = letterScorer.score(space, logits, letterTokens).probabilities();
          }
          latency = (System.nanoTime() - t0) / 1e9;
          candidateTokens += 1; // one forward, whatever the option count
        } else {
          long t0 = System.nanoTime();
          CandidateBatch batch = evaluator.evaluateBatch(state, new Choice(id, labels), candidates);
          latency = (System.nanoTime() - t0) / 1e9;
          candidateTokens += batch.candidateTokensEvaluated();
          sharedProven += batch.sharedPrefixProven() ? 1 : 0;
          p = batch.verdict().probabilities();
        }

        if (warmupRemaining > 0) {
          // Discarded: no row written, and the tallies restart so warmup cannot leak into them.
          warmupRemaining--;
          candidateTokens = 0;
          sharedProven = 0;
          if (warmupRemaining == 0) {
            // Restart the clock too, or the reported wall time bills the measured items for
            // warmup they did not pay.
            wall0 = System.nanoTime();
          }
          continue;
        }

        StringBuilder sb = new StringBuilder(id).append('\t').append(latency);
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
    if (cuda != null && cuda.counters().isPresent()) {
      var counters = cuda.counters().orElseThrow();
      // nvidia-smi samples instantaneously and says nothing about whether our kernel was asked to
      // do anything. These counters do: an inert kernel loaded, reported itself, and refused
      // every operation.
      System.out.printf("  cuda inert            %s%n", counters.inert());
      System.out.printf("  cuda accelerated ops  %d%n", counters.totalAcceleratedOperations());
      System.out.printf("  cuda by tensor type   %s%n", counters.acceleratedOperations());
      System.out.printf("  cuda refusals         %s%n", counters.refusals());
      // The launch/transfer terms are what decide whether a device run can beat the CPU at all.
      // One launch per batch row means a 512-row prefill projection is 512 serialized launches.
      System.out.printf("  cuda launches         %d%n", counters.kernelLaunches());
      System.out.printf(
          "  cuda h2d              %d copies, %.1f MiB%n",
          counters.hostToDeviceTransfers(), counters.hostToDeviceBytes() / 1048576.0);
      System.out.printf(
          "  cuda d2h              %d copies, %.1f MiB%n",
          counters.deviceToHostTransfers(), counters.deviceToHostBytes() / 1048576.0);
      System.out.printf(
          "  cuda weight uploads   %d tensors, %.1f MiB%n",
          counters.weightUploads(), counters.weightUploadBytes() / 1048576.0);
      runtime
          .with("cudaAcceleratedOps", String.valueOf(counters.totalAcceleratedOperations()))
          .with("cudaInert", String.valueOf(counters.inert()));
    }
    Files.writeString(
        Path.of(out.toString().replaceAll("\\.tsv$", "") + "-runtime.json"),
        runtime.toJson() + System.lineSeparator(),
        StandardCharsets.UTF_8);
    System.out.printf(
        "done: %d tasks, wall %.1f s, candidate-token forwards %d, prefix sharing proven on %d/%d%n",
        done, (System.nanoTime() - wall0) / 1e9, candidateTokens, sharedProven, done);
  }
}
