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
package com.integrallis.models.backend.purejava.bert;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.bert.DecisionCalibrationQualification.Outcome;
import com.integrallis.models.backend.purejava.bert.DecisionCalibrationQualification.Report;
import com.integrallis.models.backend.purejava.bert.ModernBertConfig.QuestionType;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Gate 8: Laya over JevBench, scored by {@code decision-calibration-v1}.
 *
 * <p>The cohort is {@code cohort-jevbench-120.jsonl} — JevBench v1.1, MIT, **authored and reviewed
 * outside this project**, which is the only reason it can qualify anything. A corpus written here
 * would score our own assumptions back to us.
 *
 * <p>Thresholds were registered in {@code experiments/laya-modern-bert/README.md} before this ran,
 * together with the cohort's chance accuracy (0.3004 overall, 0.2090 on {@code choice}, 0.5000 on
 * the binary {@code noul} subset). A miss is published as a null.
 *
 * <p>Everything in the path is the shipped one: {@link ModernBertDecisionModel} renders the
 * artifact's own prompt template, tokenizes with the artifact's own tokenizer, runs the Java
 * decoder, and applies the artifact's own temperature table. Nothing is stubbed and nothing is
 * tuned per case.
 */
@Tag("model-fixture")
class LayaJevBenchQualificationIT {

  private static final Path ARTIFACT =
      Path.of(
          System.getProperty(
              "models.fixtures.laya",
              Path.of(System.getProperty("user.home"), ".jvllm", "models", "Laya-Q8_0.gguf")
                  .toString()));

  private static final Path COHORT =
      Path.of(
          System.getProperty(
              "models.fixtures.jevbench",
              "benchmark-results/2026-09-24-decision-latency/clm-arm/cohort-jevbench-120.jsonl"));

  private static final Path REPORT =
      Path.of(System.getProperty("laya.report.dir", "build"))
          .resolve("jevbench-120-decision-calibration-v2.json");

  @Test
  void qualifiesOrPublishesTheNull() throws IOException {
    assumeTrue(Files.isRegularFile(ARTIFACT), "Laya-Q8_0.gguf is not installed at " + ARTIFACT);
    assumeTrue(Files.isRegularFile(COHORT), "JevBench cohort not found at " + COHORT);

    List<Case> cases = cohort();
    assertThat(cases).hasSize(120);

    List<Outcome> outcomes = new ArrayList<>();
    List<String> perCase = new ArrayList<>();
    long startedAt = System.nanoTime();

    try (ModernBertDecisionModel model = ModernBertDecisionModel.open(ARTIFACT)) {
      for (Case one : cases) {
        List<ModernBertDecisionPrompt.Option> options = new ArrayList<>();
        for (String label : one.labels) {
          options.add(new ModernBertDecisionPrompt.Option(label, one.criteria.get(label)));
        }
        ModernBertDecisionAnswer answer =
            model.decide(one.type, one.instructions, options, one.state);

        String chosen = one.labels.get(answer.chosen());
        boolean correct = chosen.equals(one.expected);
        outcomes.add(
            new Outcome(
                one.id,
                one.type,
                one.labels.size(),
                correct,
                answer.confidence(),
                answer.uncalibrated()));
        // The RAW logits are recorded, not just the confidence. Temperature scaling is refitted
        // from logits, and without them a refit would need another full pass over the cohort --
        // eight minutes to recover a number that was already computed and thrown away.
        StringBuilder logits = new StringBuilder();
        for (int option = 0; option < answer.logits().length; option++) {
          logits
              .append(option == 0 ? "" : ",")
              .append(String.format(Locale.ROOT, "%.6f", answer.logits()[option]));
        }
        perCase.add(
            String.format(
                Locale.ROOT,
                "    {\"id\":\"%s\",\"type\":\"%s\",\"options\":%d,\"expected\":\"%s\","
                    + "\"chosen\":\"%s\",\"expectedIndex\":%d,\"correct\":%s,"
                    + "\"confidence\":%.6f,\"uncalibrated\":%s,\"temperature\":%.6f,"
                    + "\"bucket\":\"%s\",\"rawLogits\":[%s]}",
                one.id,
                one.type.key(),
                one.labels.size(),
                one.expected,
                chosen,
                one.labels.indexOf(one.expected),
                correct,
                answer.confidence(),
                answer.uncalibrated(),
                answer.temperature(),
                answer.temperatureBucket(),
                logits));
      }
    }
    double seconds = (System.nanoTime() - startedAt) / 1e9;

    Report report = DecisionCalibrationQualification.assess(outcomes);
    System.out.printf(
        "JEVBENCH policy=%s cases=%d accuracy=%.4f chance=%.4f ece=%.4f uncalibrated=%d"
            + " exercised=%d seconds=%.1f%n",
        report.policyId(),
        report.cases(),
        report.accuracy(),
        report.chanceAccuracy(),
        report.calibrationError(),
        report.uncalibratedAnswers(),
        report.uncalibratedBucketsExercised(),
        seconds);
    report
        .accuracyByType()
        .forEach(
            (type, accuracy) ->
                System.out.printf(
                    "JEVBENCH   %-7s n=%-4d accuracy=%.4f chance=%.4f%n",
                    type,
                    report.casesByType().get(type),
                    accuracy,
                    report.chanceByType().get(type)));
    report.failures().forEach(failure -> System.out.println("JEVBENCH   FAIL " + failure));
    report.limitations().forEach(l -> System.out.println("JEVBENCH   LIMITATION " + l));
    System.out.println(
        "JEVBENCH tier="
            + report.tier()
            + " publishable="
            + report.tier().publishable()
            + " exportsConfidenceQuality="
            + report.tier().exportsConfidenceQuality());

    Files.createDirectories(REPORT.getParent());
    Files.writeString(REPORT, json(report, perCase, seconds), StandardCharsets.UTF_8);

    // The report is written whichever way the gate goes -- a null is a result and gets an artifact
    // like any other. The assertion is on the policy, not on a number chosen after the fact.
    // The assertion is on the ACCURACY gates, which decide publishability. A calibration miss is
    // a tier, recorded in the report and in the limitations, not a reason to withhold the model --
    // see DecisionUseCaseTier for why, and for the double standard v1 created.
    assertThat(report.failures())
        .as(
            "decision-calibration-v2 over JevBench 120; thresholds registered before this ran,"
                + " report at %s",
            REPORT)
        .isEmpty();
    assertThat(report.tier().publishable()).as("tier %s", report.tier()).isTrue();
  }

  private static String sha256(Path path) throws IOException {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      try (InputStream in = Files.newInputStream(path)) {
        byte[] buffer = new byte[1 << 16];
        for (int read = in.read(buffer); read > 0; read = in.read(buffer)) {
          digest.update(buffer, 0, read);
        }
      }
      StringBuilder hex = new StringBuilder(64);
      for (byte b : digest.digest()) {
        hex.append(String.format(Locale.ROOT, "%02x", b));
      }
      return hex.toString();
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 is required by every JVM", e);
    }
  }

  private static long sizeOf(Path path) throws IOException {
    return Files.size(path);
  }

  /** Machine specs, because a result without them is not reproducible. */
  private static String environment() {
    return "  \"environment\": {\n"
        + "    \"osName\": \""
        + System.getProperty("os.name")
        + "\",\n"
        + "    \"osArch\": \""
        + System.getProperty("os.arch")
        + "\",\n"
        + "    \"javaVersion\": \""
        + System.getProperty("java.version")
        + "\",\n"
        + "    \"availableProcessors\": "
        + Runtime.getRuntime().availableProcessors()
        + "\n"
        + "  },\n";
  }

  private static String json(Report report, List<String> perCase, double seconds)
      throws IOException {
    StringBuilder out = new StringBuilder();
    out.append("{\n");
    out.append("  \"policy\": \"").append(report.policyId()).append("\",\n");
    out.append("  \"cohort\": \"JevBench v1.1, MIT, 120 cases\",\n");
    out.append("  \"artifact\": \"").append(ARTIFACT.getFileName()).append("\",\n");
    // The digest and size, not just the file name. A report that names its artifact by file name
    // binds to nothing: the catalog entry downstream would have to copy the digest from itself
    // rather than check it against what was actually measured.
    out.append("  \"artifactSha256\": \"").append(sha256(ARTIFACT)).append("\",\n");
    out.append("  \"artifactSizeBytes\": ").append(sizeOf(ARTIFACT)).append(",\n");
    out.append("  \"cohortSha256\": \"").append(sha256(COHORT)).append("\",\n");
    out.append("  \"cases\": ").append(report.cases()).append(",\n");
    out.append(
        String.format(Locale.ROOT, "  \"chanceAccuracy\": %.6f,%n", report.chanceAccuracy()));
    out.append(String.format(Locale.ROOT, "  \"accuracy\": %.6f,%n", report.accuracy()));
    out.append(
        String.format(Locale.ROOT, "  \"calibrationError\": %.6f,%n", report.calibrationError()));
    out.append("  \"uncalibratedAnswers\": ").append(report.uncalibratedAnswers()).append(",\n");
    out.append("  \"uncalibratedBucketsExercised\": ")
        .append(report.uncalibratedBucketsExercised())
        .append(",\n");
    out.append(String.format(Locale.ROOT, "  \"wallClockSeconds\": %.2f,%n", seconds));
    out.append(environment());
    out.append("  \"byType\": {\n");
    List<String> types = new ArrayList<>(report.accuracyByType().keySet());
    for (int index = 0; index < types.size(); index++) {
      String type = types.get(index);
      out.append(
          String.format(
              Locale.ROOT,
              "    \"%s\": {\"cases\": %d, \"accuracy\": %.6f, \"chance\": %.6f}%s%n",
              type,
              report.casesByType().get(type),
              report.accuracyByType().get(type),
              report.chanceByType().get(type),
              index + 1 < types.size() ? "," : ""));
    }
    out.append("  },\n");
    out.append("  \"tier\": \"").append(report.tier()).append("\",\n");
    out.append("  \"publishable\": ").append(report.tier().publishable()).append(",\n");
    out.append("  \"exportsConfidenceQuality\": ")
        .append(report.tier().exportsConfidenceQuality())
        .append(",\n");
    out.append("  \"limitations\": [");
    for (int index = 0; index < report.limitations().size(); index++) {
      out.append(index == 0 ? "\n" : ",\n");
      out.append("    \"").append(report.limitations().get(index).replace("\"", "'")).append("\"");
    }
    out.append(report.limitations().isEmpty() ? "],\n" : "\n  ],\n");
    out.append("  \"failures\": [");
    for (int index = 0; index < report.failures().size(); index++) {
      out.append(index == 0 ? "\n" : ",\n");
      out.append("    \"").append(report.failures().get(index).replace("\"", "'")).append("\"");
    }
    out.append(report.failures().isEmpty() ? "],\n" : "\n  ],\n");
    out.append("  \"cases_detail\": [\n");
    out.append(String.join(",\n", perCase));
    out.append("\n  ]\n}\n");
    return out.toString();
  }

  private record Case(
      String id,
      QuestionType type,
      String instructions,
      List<String> labels,
      Map<String, String> criteria,
      String state,
      String expected) {}

  /**
   * Reads the cohort.
   *
   * <p>Hand-parsed rather than via Jackson because {@code backend-java} ships and a test cohort is
   * not a reason to put a JSON library in its dependency graph. The file is one object per line
   * with no nested arrays beyond {@code labels}, which is what makes that tractable.
   */
  private static List<Case> cohort() throws IOException {
    List<Case> cases = new ArrayList<>();
    for (String line : Files.readAllLines(COHORT, StandardCharsets.UTF_8)) {
      if (line.isBlank()) {
        continue;
      }
      String id = string(line, "id");
      // Two shapes, and the cohort really does carry both. All 108 `choice` and `noul` cases name
      // their answer as a string label; all 12 `score` cases name it as an integer index, and
      // their `labels` are "0".."n" so the index IS the label. Assuming one shape failed loudly on
      // the first run, which is better than scoring twelve cases against the wrong key.
      String expected = stringOrNumber(line, "expected");
      String state = string(line, "state");
      String instructions = string(line, "instructions");
      String type = string(line, "type");
      List<String> labels = new ArrayList<>();
      Matcher labelBlock =
          Pattern.compile("\"labels\":\\s*\\[(.*?)]", Pattern.DOTALL).matcher(line);
      if (labelBlock.find()) {
        Matcher label = Pattern.compile("\"((?:[^\"\\\\]|\\\\.)*)\"").matcher(labelBlock.group(1));
        while (label.find()) {
          labels.add(unescape(label.group(1)));
        }
      }
      // `criteria` is an object keyed by label for `choice` and `noul`, and a positional array for
      // `score` -- which matches laya's own template, whose `score` branch renders
      // "level {key}: {description}" and so wants the description by position.
      Map<String, String> criteria = new LinkedHashMap<>();
      Matcher criteriaObject =
          Pattern.compile("\"criteria\":\\s*\\{(.*?)}", Pattern.DOTALL).matcher(line);
      Matcher criteriaArray =
          Pattern.compile("\"criteria\":\\s*\\[(.*?)]", Pattern.DOTALL).matcher(line);
      if (criteriaObject.find()) {
        Matcher entry =
            Pattern.compile("\"((?:[^\"\\\\]|\\\\.)*)\":\\s*\"((?:[^\"\\\\]|\\\\.)*)\"")
                .matcher(criteriaObject.group(1));
        while (entry.find()) {
          criteria.put(unescape(entry.group(1)), unescape(entry.group(2)));
        }
      } else if (criteriaArray.find()) {
        Matcher entry =
            Pattern.compile("\"((?:[^\"\\\\]|\\\\.)*)\"").matcher(criteriaArray.group(1));
        int position = 0;
        while (entry.find() && position < labels.size()) {
          criteria.put(labels.get(position++), unescape(entry.group(1)));
        }
      }
      cases.add(
          new Case(
              id,
              QuestionType.valueOf(type.toUpperCase(Locale.ROOT)),
              instructions,
              List.copyOf(labels),
              Map.copyOf(criteria),
              state,
              expected));
    }
    return cases;
  }

  /** A value that may be quoted or a bare number; the cohort uses both for {@code expected}. */
  private static String stringOrNumber(String line, String key) {
    Matcher quoted =
        Pattern.compile("\"" + key + "\":\\s*\"((?:[^\"\\\\]|\\\\.)*)\"").matcher(line);
    if (quoted.find()) {
      return unescape(quoted.group(1));
    }
    Matcher bare = Pattern.compile("\"" + key + "\":\\s*(-?\\d+)").matcher(line);
    if (bare.find()) {
      return bare.group(1);
    }
    throw new IllegalStateException("cohort line has no " + key + ": " + line);
  }

  private static String string(String line, String key) {
    Matcher matcher =
        Pattern.compile("\"" + key + "\":\\s*\"((?:[^\"\\\\]|\\\\.)*)\"").matcher(line);
    if (!matcher.find()) {
      throw new IllegalStateException("cohort line has no " + key + ": " + line);
    }
    return unescape(matcher.group(1));
  }

  private static String unescape(String raw) {
    return raw.replace("\\\"", "\"")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\\", "\\");
  }
}
