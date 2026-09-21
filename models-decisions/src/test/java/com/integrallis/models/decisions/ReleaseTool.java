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

/**
 * Cuts a release: fits on the train split, calibrates on the calibration split, reads the sealed
 * split once, and writes the artifact that produced those numbers.
 *
 * <p>The sealed reading and the artifact come from the same object, so the file that ships is the
 * file that was scored. There is no step here where a better configuration could be selected after
 * seeing the sealed result, because the sealed split is opened after the fitting is finished and
 * refuses to be opened twice.
 */
public final class ReleaseTool {

  private ReleaseTool() {}

  /** Arguments: harvest JSONL, corpus name, base model id, output artifact path, base file. */
  public static void main(String[] args) throws IOException {
    Path harvest = Path.of(args[0]);
    String corpus = args[1];
    String baseModel = args[2];
    Path out = Path.of(args[3]);
    // The base file itself, so the artifact records what it was actually fitted against.
    String baseDigest = args.length > 4 && !args[4].isBlank() ? DecisionArtifact.digestOf(Path.of(args[4])) : "";

    List<HarvestRecord> records = new ArrayList<>();
    int skipped = 0;
    try (BufferedReader reader = Files.newBufferedReader(harvest, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        try {
          records.add(parse(line));
        } catch (RuntimeException incomplete) {
          skipped++;
        }
      }
    }

    Noul space = new Noul("the state answers the question");
    NoulEvaluation evaluation =
        new NoulEvaluation(
            corpus, space, records, new LogisticHeadTrainer(400, 0.1, 1.0), baseModel, baseDigest);
    NoulReport report = evaluation.scoreSealedOnce();

    DecisionArtifact artifact = evaluation.artifact();
    artifact.write(out);

    System.out.printf("=== %s RELEASE (sealed split read once) ===%n", corpus);
    System.out.printf("  rows %d, skipped %d partial lines%n", records.size(), skipped);
    System.out.printf("  sealed items          %d%n", report.sealedSize());
    System.out.printf("  majority-class floor  %.4f%n", report.majorityFloor());
    System.out.printf("  head accuracy         %.4f%n", report.headAccuracy());
    System.out.printf("  base decode accuracy  %.4f%n", report.decodeAccuracy());
    System.out.printf("  agreement with decode %.4f%n", report.agreementWithDecode());
    System.out.printf("  ECE                   %.4f%n", report.expectedCalibrationError());
    System.out.printf("  Brier                 %.4f%n", report.brier());
    System.out.printf("  temperature           %.4f%n", report.temperature());
    System.out.printf("  truncated in sealed   %.4f%n", report.sealedTruncatedRate());
    System.out.printf(
        "  base informative      %s%n", report.teacherIsInformative(0.05) ? "yes" : "NO");
    System.out.println();

    // A release is quoted by its numbers and pinned by its digest, so print both together.
    byte[] bytes = Files.readAllBytes(out);
    System.out.printf("  artifact              %s%n", out);
    System.out.printf("  bytes                 %d%n", bytes.length);
    System.out.printf("  sha256                %s%n", sha256(bytes));
    System.out.printf("  base digest           %s%n", baseDigest.isEmpty() ? "NOT RECORDED" : baseDigest);

    // The released file must reproduce the decisions just reported, or the report describes
    // something that was never written to disk.
    DecisionArtifact reloaded = DecisionArtifact.read(out);
    int mismatches = 0;
    for (HarvestRecord record : records) {
      if (!record.split().equals("sealed")) {
        continue;
      }
      double a = artifact.decide(record.hidden()).probabilityOfTrue();
      double b = reloaded.decide(record.hidden()).probabilityOfTrue();
      if (a != b) {
        mismatches++;
      }
    }
    System.out.printf(
        "  reload reproduces     %s%n", mismatches == 0 ? "yes, bit for bit" : mismatches + " DIFFER");
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

  private static HarvestRecord parse(String line) {
    String id = between(line, "\"id\":\"", "\"");
    String split = between(line, "\"split\":\"", "\"");
    boolean label = line.contains("\"label\":true");
    boolean decode = line.contains("\"decode\":true");
    boolean truncated = line.contains("\"truncated\":true");
    String arr = between(line, "\"hidden\":[", "]");
    String[] parts = arr.split(",");
    float[] hidden = new float[parts.length];
    for (int i = 0; i < parts.length; i++) {
      hidden[i] = Float.intBitsToFloat(Integer.parseInt(parts[i].trim()));
    }
    return new HarvestRecord(id, split, label, decode, truncated, 0L, hidden);
  }

  private static String between(String s, String open, String close) {
    int a = s.indexOf(open);
    if (a < 0) {
      throw new IllegalArgumentException("incomplete line");
    }
    a += open.length();
    int b = s.indexOf(close, a);
    if (b < 0) {
      throw new IllegalArgumentException("incomplete line");
    }
    return s.substring(a, b);
  }
}
