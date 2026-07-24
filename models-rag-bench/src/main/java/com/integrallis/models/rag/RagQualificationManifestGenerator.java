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
package com.integrallis.models.rag;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

/** Derives launch qualification data from raw RAG reports and the production policy. */
public final class RagQualificationManifestGenerator {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private RagQualificationManifestGenerator() {}

  public static RagQualificationManifest generate(
      Path reportsDirectory, String modelsRevision, int targetQualifiedModels) throws IOException {
    if (!Files.isDirectory(reportsDirectory)) {
      throw new IllegalArgumentException("reports directory does not exist: " + reportsDirectory);
    }
    if (modelsRevision == null || modelsRevision.isBlank()) {
      throw new IllegalArgumentException("modelsRevision must not be blank");
    }
    if (targetQualifiedModels < 1) {
      throw new IllegalArgumentException("targetQualifiedModels must be positive");
    }

    List<Path> reportPaths;
    try (Stream<Path> paths = Files.list(reportsDirectory)) {
      reportPaths =
          paths
              .filter(Files::isRegularFile)
              .filter(path -> path.getFileName().toString().endsWith(".json"))
              .sorted(Comparator.comparing(path -> path.getFileName().toString()))
              .toList();
    }
    if (reportPaths.isEmpty()) {
      throw new IllegalArgumentException("reports directory contains no JSON reports");
    }

    Set<String> modelIds = new HashSet<>();
    List<RagQualificationManifestEntry> entries = new ArrayList<>();
    for (Path path : reportPaths) {
      RagBenchmarkReport report = MAPPER.readValue(path.toFile(), RagBenchmarkReport.class);
      if (!modelIds.add(report.modelId())) {
        throw new IllegalArgumentException("duplicate model ID in reports: " + report.modelId());
      }
      RagProductionQualification qualification =
          RagProductionQualificationPolicy.assessLocalEngine(report);
      entries.add(entry(reportsDirectory, path, report, qualification));
    }

    int qualified =
        Math.toIntExact(entries.stream().filter(RagQualificationManifestEntry::qualified).count());
    return new RagQualificationManifest(
        RagQualificationManifest.CURRENT_SCHEMA_VERSION,
        Instant.now().toString(),
        RagQualificationManifest.POLICY_VERSION,
        modelsRevision.trim(),
        targetQualifiedModels,
        qualified,
        entries.size() - qualified,
        entries);
  }

  private static RagQualificationManifestEntry entry(
      Path reportsDirectory,
      Path path,
      RagBenchmarkReport report,
      RagProductionQualification qualification)
      throws IOException {
    RagBenchmarkSummary summary = report.summary();
    int runCount = report.runs().size();
    long rawCorrect = report.runs().stream().filter(run -> run.rawEvaluation().correct()).count();
    long modelAnswers =
        report.runs().stream()
            .filter(run -> run.grounding().decision() == GroundingDecision.MODEL_ANSWER)
            .count();
    long extractiveFallbacks =
        report.runs().stream()
            .filter(run -> run.grounding().decision() == GroundingDecision.EXTRACTIVE_FALLBACK)
            .count();
    return new RagQualificationManifestEntry(
        report.modelId(),
        report.model(),
        report.backend(),
        report.backendVersion(),
        report.artifactSha256(),
        report.artifactSizeBytes(),
        reportsDirectory.relativize(path).toString(),
        sha256(path),
        qualification.absoluteTier(),
        qualification.verdict(),
        qualification.qualified(),
        summary.totalAttempts(),
        summary.retrievalMillis().p95(),
        summary.ttftMillis().p95(),
        summary.tpotMillis().p95(),
        summary.endToEndMillis().p95(),
        summary.p50PrefillTokensPerSecond(),
        summary.p50DecodeTokensPerSecond(),
        summary.peakRssBytes(),
        summary.correctAnswerRate(),
        rate(rawCorrect, runCount),
        summary.abstentionAccuracy(),
        rate(modelAnswers, runCount),
        rate(extractiveFallbacks, runCount),
        report.environment());
  }

  private static double rate(long count, int total) {
    return total == 0 ? 0 : count / (double) total;
  }

  private static String sha256(Path path) throws IOException {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      try (InputStream input = Files.newInputStream(path);
          DigestInputStream hashing = new DigestInputStream(input, digest)) {
        hashing.transferTo(java.io.OutputStream.nullOutputStream());
      }
      return HexFormat.of().formatHex(digest.digest());
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is unavailable", impossible);
    }
  }
}
