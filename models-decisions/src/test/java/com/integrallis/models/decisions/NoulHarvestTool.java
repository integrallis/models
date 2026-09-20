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
import java.util.List;

/**
 * Harvests, for every item of a drawn split, the verdict the base reaches through its ordinary
 * decode loop and the hidden state a head would read instead.
 *
 * <p>Both come from one prompt evaluation. The prompt is prefilled through the batched path, the
 * final token is evaluated once for its hidden state, the session is rewound one position, and the
 * same token is evaluated again for its logits. That yields a hidden state and a decode verdict at
 * the identical position, which is the comparison the pre-registration gates on, without paying for
 * the prompt twice.
 *
 * <p>This is a tool rather than a test. It writes evidence; it asserts nothing.
 */
public final class NoulHarvestTool {

  private static final int WINDOW_TOKENS = 4000;

  private NoulHarvestTool() {}

  /** Runs the harvest. Arguments: model, corpus JSONL, output JSONL, sample size, corpus name. */
  public static void main(String[] args) throws IOException {
    Path model = Path.of(args[0]);
    Path corpus = Path.of(args[1]);
    Path output = Path.of(args[2]);
    int sampleSize = Integer.parseInt(args[3]);
    String corpusName = args[4];

    List<Row> rows = readCorpus(corpus);
    System.out.printf("%s: pool=%d%n", corpusName, rows.size());

    List<CorpusItem> pool = new ArrayList<>(rows.size());
    for (Row row : rows) {
      pool.add(new CorpusItem(row.id, row.passageId, row.question, row.answerable));
    }
    int train = sampleSize / 2;
    int calibration = sampleSize / 5;
    SplitPlan plan =
        SplitPlanner.plan(
            pool, sampleSize, 20260920L, train, calibration, sampleSize - train - calibration);
    System.out.printf(
        "split: train=%d calibration=%d sealed=%d sealedFloor=%.4f%n",
        plan.train().size(),
        plan.calibration().size(),
        plan.sealedTest().size(),
        plan.sealedTestMajorityFloor());

    java.util.Map<String, Row> byId = new java.util.HashMap<>();
    for (Row row : rows) {
      byId.put(row.id, row);
    }

    long start = System.nanoTime();
    long cpuStart = cpuMicros();
    int done = 0;
    int outOfWindow = 0;

    try (PureJavaBackend backend =
            PureJavaBackend.load(model, RustGgufBatchedMatrixKernel.openBundled());
        BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
      SharedPrefixInferenceBackend sharing = backend;
      int yesToken = firstToken(backend, " yes");
      int noToken = firstToken(backend, " no");

      for (String splitName : List.of("train", "calibration", "sealed")) {
        List<CorpusItem> split =
            switch (splitName) {
              case "train" -> plan.train();
              case "calibration" -> plan.calibration();
              default -> plan.sealedTest();
            };
        for (CorpusItem item : split) {
          Row row = byId.get(item.id());
          int[] prompt = backend.tokenizer().encode(buildPrompt(row));
          boolean truncated = prompt.length > WINDOW_TOKENS;
          if (truncated) {
            int[] windowed = new int[WINDOW_TOKENS];
            System.arraycopy(prompt, 0, windowed, 0, WINDOW_TOKENS);
            prompt = windowed;
            if (item.answerable()) {
              outOfWindow++;
            }
          }

          long itemCpu = cpuMicros();
          try (InferenceSession session = sharing.openSession()) {
            int last = prompt.length - 1;
            if (last > 0) {
              int[] head = new int[last];
              System.arraycopy(prompt, 0, head, 0, last);
              backend.prefill(session, head, 0);
            }
            float[] hidden = sharing.forwardHiddenState(session, prompt[last], last).clone();
            backend.rewind(session, last);
            float[] logits = backend.forward(session, prompt[last], last);
            boolean decodeSaysYes = logits[yesToken] >= logits[noToken];
            long itemMicros = cpuMicros() - itemCpu;

            writer.write(
                encode(
                    item.id(),
                    splitName,
                    item.answerable(),
                    decodeSaysYes,
                    truncated,
                    itemMicros,
                    hidden));
            writer.newLine();
          }
          if (++done % 25 == 0) {
            writer.flush();
            System.out.printf(
                "  %s %d/%d  %.1f s elapsed%n",
                corpusName, done, sampleSize, (System.nanoTime() - start) / 1e9);
          }
        }
      }
    }
    System.out.printf(
        "done %s: items=%d outOfWindowAnswerable=%d wall=%.1f s cpu=%.1f s%n",
        corpusName,
        done,
        outOfWindow,
        (System.nanoTime() - start) / 1e9,
        (cpuMicros() - cpuStart) / 1e6);
  }

  private static String buildPrompt(Row row) {
    return row.context
        + "\n\nQuestion: "
        + row.question
        + "\nIs the question answerable from the text above? Answer yes or no.\nAnswer:";
  }

  private static int firstToken(PureJavaBackend backend, String text) {
    int[] tokens = backend.tokenizer().encode(text);
    return tokens[tokens.length - 1];
  }

  private static String encode(
      String id,
      String split,
      boolean label,
      boolean decode,
      boolean truncated,
      long cpuMicros,
      float[] hidden) {
    StringBuilder builder = new StringBuilder(hidden.length * 9 + 128);
    builder
        .append("{\"id\":\"")
        .append(id.replace("\"", "'"))
        .append("\",\"split\":\"")
        .append(split)
        .append("\",\"label\":")
        .append(label)
        .append(",\"decode\":")
        .append(decode)
        .append(",\"truncated\":")
        .append(truncated)
        .append(",\"cpuMicros\":")
        .append(cpuMicros)
        .append(",\"hidden\":[");
    for (int index = 0; index < hidden.length; index++) {
      if (index > 0) {
        builder.append(',');
      }
      builder.append(Float.floatToRawIntBits(hidden[index]));
    }
    return builder.append("]}").toString();
  }

  /** Process CPU time in microseconds, the energy-proportional quantity this host can measure. */
  private static long cpuMicros() {
    try {
      String stat = Files.readString(Path.of("/sys/fs/cgroup/cpu.stat"));
      for (String line : stat.split("\n")) {
        if (line.startsWith("usage_usec")) {
          return Long.parseLong(line.substring("usage_usec ".length()).trim());
        }
      }
    } catch (IOException | RuntimeException ignored) {
      // Not a cgroup host; CPU accounting is simply unavailable and reported as zero.
    }
    return 0L;
  }

  private static List<Row> readCorpus(Path path) throws IOException {
    List<Row> rows = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) {
          continue;
        }
        rows.add(
            new Row(
                field(line, "id"),
                field(line, "passageId"),
                field(line, "question"),
                line.contains("\"answerable\": true") || line.contains("\"answerable\":true"),
                field(line, "context")));
      }
    }
    return rows;
  }

  /** A deliberately small JSON string-field reader; the corpus files are machine-written. */
  private static String field(String line, String name) {
    String key = "\"" + name + "\": \"";
    int start = line.indexOf(key);
    if (start < 0) {
      key = "\"" + name + "\":\"";
      start = line.indexOf(key);
    }
    if (start < 0) {
      return "";
    }
    start += key.length();
    StringBuilder value = new StringBuilder();
    for (int index = start; index < line.length(); index++) {
      char character = line.charAt(index);
      if (character == '\\') {
        char next = line.charAt(++index);
        value.append(
            switch (next) {
              case 'n' -> '\n';
              case 't' -> '\t';
              case 'r' -> '\r';
              case 'u' -> (char) Integer.parseInt(line.substring(index + 1, index + 5), 16);
              default -> next;
            });
        if (next == 'u') {
          index += 4;
        }
      } else if (character == '"') {
        break;
      } else {
        value.append(character);
      }
    }
    return value.toString();
  }

  private record Row(
      String id, String passageId, String question, boolean answerable, String context) {}
}
