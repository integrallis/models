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

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;

/** Fits a typed head on a harvest, reads the sealed split once, and writes the artifact. */
public final class TypedReleaseTool {

  private TypedReleaseTool() {}

  /**
   * Arguments: harvest JSONL, corpus name, space kind (choice|score), comma-separated labels,
   * question, base model id, base GGUF path (or empty), L2, output artifact path.
   */
  public static void main(String[] args) throws IOException {
    Path harvest = Path.of(args[0]);
    String corpus = args[1];
    String kind = args[2];
    List<String> labels = List.of(args[3].split(","));
    String question = args[4];
    String baseModel = args[5];
    String baseFile = args[6];
    double l2 = Double.parseDouble(args[7]);
    Path out = Path.of(args[8]);

    AnswerSpace space =
        switch (kind) {
          case "choice" -> new Choice(question, labels);
          case "score" -> new Score(question, labels);
          default -> throw new IllegalArgumentException("unknown space kind " + kind);
        };

    List<TypedRecord> records = new ArrayList<>();
    int skipped = 0;
    try (BufferedReader reader = Files.newBufferedReader(harvest, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) {
          continue;
        }
        try {
          records.add(parse(line));
        } catch (RuntimeException incomplete) {
          skipped++;
        }
      }
    }

    String baseDigest = baseFile.isBlank() ? "" : DecisionArtifact.digestOf(Path.of(baseFile));
    TypedEvaluation evaluation =
        new TypedEvaluation(
            corpus, space, records, new LogisticHeadTrainer(400, 0.1, l2), baseModel, baseDigest);
    TypedReport report = evaluation.scoreSealedOnce();

    System.out.printf("%n=== %s RELEASE (sealed split read once) ===%n", corpus);
    System.out.printf("  rows %d, skipped %d%n", records.size(), skipped);
    System.out.printf("  space                 %s over %d labels%n", kind, report.outcomes());
    System.out.printf("  sealed items          %d%n", report.sealedSize());
    System.out.printf("  majority-class floor  %.4f%n", report.majorityFloor());
    System.out.printf("  accuracy              %.4f%n", report.accuracy());
    System.out.printf("  beats floor by 0.05   %s%n", report.beatsFloor(0.05) ? "yes" : "NO");
    System.out.printf("  ECE                   %.4f%n", report.expectedCalibrationError());
    System.out.printf("  Brier (multi-class)   %.4f%n", report.brier());
    if (!Double.isNaN(report.ordinalMeanAbsoluteError())) {
      System.out.printf("  ordinal MAE (levels)  %.4f%n", report.ordinalMeanAbsoluteError());
    }
    System.out.printf("  temperature           %.4f%n", report.temperature());
    System.out.printf("  truncated in sealed   %.4f%n", report.truncatedRate());

    DecisionArtifact artifact = evaluation.artifact();
    artifact.write(out);
    byte[] bytes = Files.readAllBytes(out);
    System.out.printf("%n  artifact              %s%n", out);
    System.out.printf("  bytes                 %d%n", bytes.length);
    System.out.printf("  sha256                %s%n", sha256(bytes));
    System.out.printf(
        "  base digest           %s%n", baseDigest.isEmpty() ? "NOT RECORDED" : baseDigest);

    DecisionArtifact reloaded = DecisionArtifact.read(out);
    int mismatches = 0;
    for (TypedRecord record : records) {
      if (!record.split().equals("sealed")) {
        continue;
      }
      double[] a = artifact.decide(record.state()).probabilities();
      double[] b = reloaded.decide(record.state()).probabilities();
      for (int index = 0; index < a.length; index++) {
        if (a[index] != b[index]) {
          mismatches++;
          break;
        }
      }
    }
    System.out.printf(
        "  reload reproduces     %s%n%n",
        mismatches == 0 ? "yes, bit for bit" : mismatches + " DIFFER");
    if (mismatches != 0) {
      throw new IllegalStateException("the written artifact does not reproduce the scored one");
    }
  }

  private static String sha256(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException(impossible);
    }
  }

  private static TypedRecord parse(String line) {
    String id = between(line, "\"id\":\"", "\"");
    String split = between(line, "\"split\":\"", "\"");
    int outcome = Integer.parseInt(between(line, "\"outcome\":", ",").trim());
    boolean truncated = line.contains("\"truncated\":true");
    String array = between(line, "\"state\":[", "]");
    String[] parts = array.split(",");
    float[] state = new float[parts.length];
    for (int index = 0; index < parts.length; index++) {
      state[index] = Float.intBitsToFloat(Integer.parseInt(parts[index].trim()));
    }
    return new TypedRecord(id, split, outcome, truncated, state);
  }

  private static String between(String line, String open, String close) {
    int a = line.indexOf(open);
    if (a < 0) {
      throw new IllegalArgumentException("incomplete line");
    }
    a += open.length();
    int b = line.indexOf(close, a);
    if (b < 0) {
      throw new IllegalArgumentException("incomplete line");
    }
    return line.substring(a, b);
  }
}
