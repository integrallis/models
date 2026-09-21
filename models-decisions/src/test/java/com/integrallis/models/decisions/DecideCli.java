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
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Runs a released artifact over a JSONL file of questions, one verdict per line.
 *
 * <p>Input lines are {@code {"id":..., "context":..., "question":...}}. Output lines carry the
 * calibrated probability, the decision, and the wall time for that item, so a run of this is
 * directly comparable with any other system scored on the same file.
 *
 * <p>The prompt built here is the same one the harvest used. A released head reads the hidden state
 * that a specific prompt produces, and changing the wording at inference time would feed the head a
 * distribution it was never fitted on — which shows up as a quietly worse model rather than an
 * error.
 */
public final class DecideCli {

  private static final int WINDOW_TOKENS = 4000;

  private DecideCli() {}

  /** Arguments: artifact path, GGUF base path, input JSONL, output JSONL. */
  public static void main(String[] args) throws IOException {
    DecisionArtifact artifact = DecisionArtifact.read(Path.of(args[0]));
    Path model = Path.of(args[1]);
    Path input = Path.of(args[2]);
    Path output = Path.of(args[3]);

    System.out.printf(
        "artifact: %s, base %s, width %d, temperature %.4f%n",
        args[0], artifact.baseModel(), artifact.width(), artifact.temperature());

    long start = System.nanoTime();
    int done = 0;
    // The Rust kernel is the qualified fast path, but it is only bundled for the platforms it was
    // built for. Falling back keeps the artifact testable on a laptop; which path ran is printed,
    // because a silent fallback would make a latency number mean two different things.
    // Checked before any inference: a wrong base wastes the whole run and looks like a bad model.
    artifact.requireBase(model);

    PureJavaBackend backend;
    String kernel;
    try {
      backend = PureJavaBackend.load(model, RustGgufBatchedMatrixKernel.openBundled());
      kernel = "rust-ffm";
    } catch (RuntimeException | UnsatisfiedLinkError unavailable) {
      backend = PureJavaBackend.load(model);
      kernel = "pure-java (no native kernel for this platform: " + unavailable.getMessage() + ")";
    }
    System.out.printf("kernel:   %s%n", kernel);

    try (PureJavaBackend held = backend;
        BufferedReader reader = Files.newBufferedReader(input, StandardCharsets.UTF_8);
        BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
      SharedPrefixInferenceBackend sharing = held;

      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) {
          continue;
        }
        String id = between(line, "\"id\":\"", "\"");
        String context = unescape(between(line, "\"context\":\"", "\","));
        String question = unescape(between(line, "\"question\":\"", "\""));

        int[] prompt = held.tokenizer().encode(buildPrompt(context, question));
        boolean truncated = prompt.length > WINDOW_TOKENS;
        if (truncated) {
          int[] windowed = new int[WINDOW_TOKENS];
          System.arraycopy(prompt, 0, windowed, 0, WINDOW_TOKENS);
          prompt = windowed;
        }

        long itemStart = System.nanoTime();
        double probability;
        try (InferenceSession session = sharing.openSession()) {
          int last = prompt.length - 1;
          if (last > 0) {
            int[] head = new int[last];
            System.arraycopy(prompt, 0, head, 0, last);
            held.prefill(session, head, 0);
          }
          float[] hidden = sharing.forwardHiddenState(session, prompt[last], last);
          probability = artifact.decide(hidden).probabilityOfTrue();
        }
        double seconds = (System.nanoTime() - itemStart) / 1e9;

        writer.write(
            String.format(
                "{\"id\":\"%s\",\"probability\":%.6f,\"decision\":%s,\"truncated\":%s,"
                    + "\"latency_s\":%.4f,\"prompt_tokens\":%d}",
                id, probability, probability >= 0.5, truncated, seconds, prompt.length));
        writer.newLine();

        if (++done % 10 == 0) {
          writer.flush();
          System.out.printf("  %d  %.1f s%n", done, (System.nanoTime() - start) / 1e9);
        }
      }
    }
    System.out.printf(
        "done: %d items, wall %.1f s, %.3f s/item%n",
        done, (System.nanoTime() - start) / 1e9, (System.nanoTime() - start) / 1e9 / Math.max(done, 1));
  }

  private static String buildPrompt(String context, String question) {
    return context
        + "\n\nQuestion: "
        + question
        + "\nIs the question answerable from the text above? Answer yes or no.\nAnswer:";
  }

  private static String unescape(String s) {
    return s.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\");
  }

  private static String between(String s, String open, String close) {
    int a = s.indexOf(open);
    if (a < 0) {
      throw new IllegalArgumentException("field " + open + " missing from input line");
    }
    a += open.length();
    int b = s.indexOf(close, a);
    if (b < 0) {
      throw new IllegalArgumentException("field " + open + " unterminated in input line");
    }
    return s.substring(a, b);
  }
}
