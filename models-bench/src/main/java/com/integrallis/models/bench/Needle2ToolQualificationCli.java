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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.GenerationLoop;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Runs the pinned Needle 2 tool-calling conformance suite through the pure-Java backend. */
final class Needle2ToolQualificationCli {
  static final String MODEL_ID = "cactus_compute_needle2_cact_cq2_mixed";
  static final String MODEL_NAME = "Cactus Compute Needle 2 CACT CQ2 Mixed";
  static final String ARTIFACT_SHA256 =
      "b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba";

  private static final int PASS = 0;
  private static final int FAIL = 1;
  private static final Set<String> OPTIONS =
      Set.of("model", "report", "max-tokens", "models-revision");

  private Needle2ToolQualificationCli() {}

  record Configuration(Path model, Path report, int maxTokens, String modelsRevision) {}

  record GenerationControls(double temperature, int maxTokens, String promptTemplate) {}

  record SuiteIdentity(
      String id,
      String sha256,
      String sourceRepository,
      String sourceRevision,
      String sourcePath) {}

  record Report(
      int schemaVersion,
      Instant createdAt,
      String policyVersion,
      String modelsRevision,
      String modelId,
      String model,
      String backend,
      String backendVersion,
      String artifactSha256,
      long artifactSizeBytes,
      SuiteIdentity suite,
      GenerationControls generation,
      BenchmarkEnvironment environment,
      BackendDiagnostics backendDiagnostics,
      Needle2ToolQualification.Summary summary,
      List<Needle2ToolQualification.CaseResult> cases) {
    Report {
      cases = List.copyOf(cases);
    }
  }

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    String configuredModel = values.get("model");
    if (configuredModel == null || configuredModel.isBlank()) {
      throw new IllegalArgumentException("--model is required");
    }
    Path model = Path.of(configuredModel);
    if (!Files.isRegularFile(model)) {
      throw new IllegalArgumentException("artifact does not exist: " + model);
    }
    int maxTokens = BenchmarkCliArguments.integer(values, "max-tokens", 256);
    if (maxTokens < 32 || maxTokens > 512) {
      throw new IllegalArgumentException("--max-tokens must be between 32 and 512");
    }
    String revision = values.get("models-revision");
    if (revision == null || !revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact 40-character Git SHA");
    }
    Path report =
        Path.of(
            values.getOrDefault(
                "report", "build/reports/tool-qualification/needle2-cact-pure-java.json"));
    return new Configuration(model, report, maxTokens, revision);
  }

  static int run(String[] args) throws IOException {
    Configuration configuration = parse(args);
    String artifactSha256 = Hashing.sha256(configuration.model());
    if (!ARTIFACT_SHA256.equals(artifactSha256)) {
      throw new IllegalArgumentException(
          "artifact digest does not match the pinned Needle 2 release: " + artifactSha256);
    }

    ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    Needle2ToolQualification.Suite suite = Needle2ToolQualification.loadSuite(mapper);
    SamplingOptions options =
        SamplingOptions.builder().temperature(0.0f).maxTokens(configuration.maxTokens()).build();
    List<Needle2ToolQualification.CaseResult> results = new ArrayList<>();
    BackendDiagnostics diagnostics;
    long loadStart = System.nanoTime();
    try (PureJavaBackend backend = PureJavaBackend.load(configuration.model())) {
      double loadMillis = (System.nanoTime() - loadStart) / 1_000_000.0;
      System.out.printf("loaded %s in %.1f ms%n", MODEL_ID, loadMillis);
      GenerationLoop generation = new GenerationLoop(backend);
      for (Needle2ToolQualification.Case item : suite.cases()) {
        long start = System.nanoTime();
        String output =
            generation.generate(
                ChatTemplate.NEEDLE2.render(
                    List.of(ChatMessage.user(item.query())), item.toolSpecs(mapper)),
                options);
        long elapsedMillis = (System.nanoTime() - start) / 1_000_000L;
        Needle2ToolQualification.CaseResult result =
            Needle2ToolQualification.evaluate(mapper, item, output, elapsedMillis);
        results.add(result);
        System.out.printf(
            "%-10s %s %6d ms  selection=%s schema=%s arguments=%d/%d%n",
            item.id(),
            result.passed() ? "PASS" : "FAIL",
            elapsedMillis,
            result.selectionExact(),
            result.schemaValid() && result.declaredArgumentsOnly(),
            result.expectedArgumentMatches(),
            result.expectedArguments());
      }
      diagnostics = backend.diagnostics();
    }

    Needle2ToolQualification.Summary summary = Needle2ToolQualification.summarize(results);
    Report report =
        new Report(
            1,
            Instant.now(),
            Needle2ToolQualification.POLICY_VERSION,
            configuration.modelsRevision(),
            MODEL_ID,
            MODEL_NAME,
            "pure-java",
            implementationVersion(),
            artifactSha256,
            Files.size(configuration.model()),
            suiteIdentity(suite),
            new GenerationControls(0.0, configuration.maxTokens(), "needle2"),
            BenchmarkEnvironment.capture(),
            diagnostics,
            summary,
            results);
    write(configuration.report(), mapper, report);
    System.out.printf(
        "%s structured=%.3f selection=%.3f schema=%.3f declared-only=%.3f "
            + "arguments=%.3f refusal=%.3f report=%s%n",
        summary.verdict(),
        summary.structuredOutputRate(),
        summary.toolSelectionExactRate(),
        summary.schemaValidityRate(),
        summary.declaredArgumentsOnlyRate(),
        summary.expectedArgumentAccuracy(),
        summary.refusalAccuracy(),
        configuration.report().toAbsolutePath());
    return summary.qualified() ? PASS : FAIL;
  }

  private static SuiteIdentity suiteIdentity(Needle2ToolQualification.Suite suite)
      throws IOException {
    String suiteText;
    try (InputStream input =
        Needle2ToolQualificationCli.class
            .getClassLoader()
            .getResourceAsStream(Needle2ToolQualification.SUITE_RESOURCE)) {
      if (input == null) {
        throw new IOException(
            "Missing qualification suite: " + Needle2ToolQualification.SUITE_RESOURCE);
      }
      suiteText = new String(input.readAllBytes(), StandardCharsets.UTF_8);
    }
    return new SuiteIdentity(
        suite.suiteId(),
        Hashing.sha256(suiteText),
        suite.sourceRepository(),
        suite.sourceRevision(),
        suite.sourcePath());
  }

  private static String implementationVersion() {
    String version = PureJavaBackend.class.getPackage().getImplementationVersion();
    return version == null || version.isBlank() ? "development" : version;
  }

  private static void write(Path output, ObjectMapper mapper, Report report) throws IOException {
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    Files.writeString(output, mapper.writeValueAsString(report) + System.lineSeparator());
  }
}
