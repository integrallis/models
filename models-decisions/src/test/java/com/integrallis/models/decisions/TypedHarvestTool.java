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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Harvests one hidden state per item for a fixed question over varying state.
 *
 * <p>This is the shape a Choice or Score head wants and the shape the answerability harvest is
 * not: there, the question changed per item and had to ride in the prompt. Here the question is
 * the same for every item, so it is a property of the head rather than of the input, and the state
 * is encoded once with no per-question cost.
 *
 * <p>The first rows are reported as they land so a malformed corpus shows up in the first minute
 * rather than after an hour of compute. Two corpora have already been harvested to completion in
 * this project before anyone read a row back, and one of them was void.
 */
public final class TypedHarvestTool {

  private static final int WINDOW_TOKENS = 1024;

  private TypedHarvestTool() {}

  /** Arguments: GGUF base, corpus JSONL, output JSONL, sample size, corpus name, prompt suffix. */
  public static void main(String[] args) throws IOException {
    Path model = Path.of(args[0]);
    Path corpus = Path.of(args[1]);
    Path output = Path.of(args[2]);
    int sampleSize = Integer.parseInt(args[3]);
    String corpusName = args[4];
    String suffix = args.length > 5 ? args[5] : "";

    List<Row> rows = readCorpus(corpus);
    if (rows.size() < sampleSize) {
      throw new IllegalArgumentException(
          "corpus holds " + rows.size() + " rows, fewer than the requested " + sampleSize);
    }
    // Seeded so the sample is the same every time the harvest is re-run for the same size.
    Collections.shuffle(rows, new Random(20260921L));
    rows = new ArrayList<>(rows.subList(0, sampleSize));

    PureJavaBackend backend;
    String kernel;
    try {
      backend = PureJavaBackend.load(model, RustGgufBatchedMatrixKernel.openBundled());
      kernel = "rust-ffm";
    } catch (RuntimeException | UnsatisfiedLinkError unavailable) {
      backend = PureJavaBackend.load(model);
      kernel = "pure-java";
    }

    long start = System.nanoTime();
    int done = 0;
    int truncatedCount = 0;
    try (PureJavaBackend held = backend;
        BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
      SharedPrefixInferenceBackend sharing = held;
      System.out.printf("  %s: %d items, kernel %s%n", corpusName, rows.size(), kernel);

      for (int index = 0; index < rows.size(); index++) {
        Row row = rows.get(index);
        int[] prompt = held.tokenizer().encode(row.state + suffix);
        boolean truncated = prompt.length > WINDOW_TOKENS;
        if (truncated) {
          int[] windowed = new int[WINDOW_TOKENS];
          System.arraycopy(prompt, 0, windowed, 0, WINDOW_TOKENS);
          prompt = windowed;
          truncatedCount++;
        }

        float[] state;
        try (InferenceSession session = sharing.openSession()) {
          int last = prompt.length - 1;
          if (last > 0) {
            int[] head = new int[last];
            System.arraycopy(prompt, 0, head, 0, last);
            held.prefill(session, head, 0);
          }
          state = sharing.forwardHiddenState(session, prompt[last], last).clone();
        }

        writer.write(encode(row.id, splitFor(index), row.label, truncated, state));
        writer.newLine();

        if (++done <= 3 || done % 100 == 0) {
          writer.flush();
          System.out.printf(
              "  %d/%d  label %d  width %d  %.1f s%n",
              done, rows.size(), row.label, state.length, (System.nanoTime() - start) / 1e9);
        }
      }
    }
    System.out.printf(
        "done %s: items=%d truncated=%d wall=%.1f s%n",
        corpusName, done, truncatedCount, (System.nanoTime() - start) / 1e9);
  }

  /** Deterministic 60/20/20, by position in the already-shuffled sample. */
  private static String splitFor(int index) {
    int bucket = index % 10;
    if (bucket < 6) {
      return "train";
    }
    return bucket < 8 ? "calibration" : "sealed";
  }

  private static String encode(
      String id, String split, int outcome, boolean truncated, float[] state) {
    StringBuilder line = new StringBuilder(state.length * 8 + 64);
    line.append("{\"id\":\"").append(id).append("\",\"split\":\"").append(split);
    line.append("\",\"outcome\":").append(outcome);
    line.append(",\"truncated\":").append(truncated).append(",\"state\":[");
    for (int index = 0; index < state.length; index++) {
      if (index > 0) {
        line.append(',');
      }
      // Raw bits, so the round trip is exact rather than merely close.
      line.append(Float.floatToIntBits(state[index]));
    }
    return line.append("]}").toString();
  }

  private record Row(String id, int label, String state) {}

  private static List<Row> readCorpus(Path path) throws IOException {
    List<Row> rows = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) {
          continue;
        }
        rows.add(new Row(field(line, "id"), intField(line, "label"), field(line, "state")));
      }
    }
    return rows;
  }

  private static String field(String line, String name) {
    // JSON permits whitespace after the colon and most writers emit it, so a reader that demands
    // its absence rejects perfectly ordinary input.
    int a = valueStart(line, name);
    if (line.charAt(a) != '"') {
      throw new IllegalArgumentException("field " + name + " is not a string");
    }
    a++;
    StringBuilder value = new StringBuilder();
    for (int index = a; index < line.length(); index++) {
      char character = line.charAt(index);
      if (character == '\\' && index + 1 < line.length()) {
        char next = line.charAt(++index);
        value.append(
            switch (next) {
              case 'n' -> '\n';
              case 't' -> '\t';
              case 'r' -> '\r';
              default -> next;
            });
      } else if (character == '"') {
        return value.toString();
      } else {
        value.append(character);
      }
    }
    throw new IllegalArgumentException("field " + name + " unterminated");
  }

  /** Index of the first non-space character after {@code "name":}. */
  private static int valueStart(String line, String name) {
    String open = "\"" + name + "\":";
    int a = line.indexOf(open);
    if (a < 0) {
      throw new IllegalArgumentException("field " + name + " missing");
    }
    a += open.length();
    while (a < line.length() && Character.isWhitespace(line.charAt(a))) {
      a++;
    }
    if (a >= line.length()) {
      throw new IllegalArgumentException("field " + name + " has no value");
    }
    return a;
  }

  private static int intField(String line, String name) {
    int a = valueStart(line, name);
    int b = a;
    while (b < line.length() && (Character.isDigit(line.charAt(b)) || line.charAt(b) == '-')) {
      b++;
    }
    return Integer.parseInt(line.substring(a, b));
  }
}
