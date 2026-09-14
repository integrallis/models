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
import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolDecisionPolicy;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.ToolDecisionScore;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Screens a frozen hybrid decision rule on the separate BFCL-live development window. */
final class ActivatedHybridLiveDecisionCli {
  static final String EXPERIMENT = "qwen3-17b-hybrid-live-decision-v21";
  static final ActivatedToolDecisionPolicy POLICY =
      new ActivatedToolDecisionPolicy(4913, 19536, 4.5f, 1.3521204f, 33.705593f, 39.42147f);

  private static final Set<String> OPTIONS =
      Set.of("model", "adapter", "records", "manifest", "output", "models-revision");

  private ActivatedHybridLiveDecisionCli() {}

  record Configuration(
      Path model, Path adapter, Path records, Path manifest, Path output, String modelsRevision) {}

  record Observation(
      String id,
      String kind,
      boolean callExpected,
      ActivatedToolDecisionPolicy.Decision decision,
      float baseMargin,
      float specialistMargin,
      boolean physicallyShared,
      long elapsedMillis) {}

  record Summary(
      int total,
      int calls,
      int noCalls,
      int correctCallDecisions,
      int correctNoCalls,
      int physicallyShared,
      boolean passed,
      String verdict) {}

  record Report(
      int schemaVersion,
      String experiment,
      String createdAt,
      String modelsRevision,
      String sourceSha256,
      String sourceManifestSha256,
      String modelSha256,
      ActivatedAdapterMetadata adapter,
      BenchmarkEnvironment environment,
      BackendDiagnostics backend,
      ActivatedToolDecisionPolicy policy,
      boolean developmentDataPreviouslyExposed,
      Summary summary,
      List<Observation> observations) {
    Report {
      observations = List.copyOf(observations);
    }
  }

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    String revision = required(values, "models-revision");
    if (!revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact 40-character Git SHA");
    }
    return new Configuration(
        requiredFile(values, "model"),
        requiredDirectory(values, "adapter"),
        requiredFile(values, "records"),
        requiredFile(values, "manifest"),
        Path.of(values.getOrDefault("output", "build/reports/tool-qualification/v21-live.json")),
        revision);
  }

  static int run(String[] args) throws IOException {
    Configuration configuration = parse(args);
    requireHash(
        configuration.model(), ActivatedHybridDevelopmentCli.MODEL_SHA256, "base GGUF model");
    requireHash(
        configuration.adapter().resolve("adapter_model.safetensors"),
        ActivatedHiddenDecisionProfileCli.ADAPTER_SHA256,
        "activated adapter");
    requireHash(
        configuration.records(), ActivatedQ4TransferCli.LIVE_RECORDS_SHA256, "live records");
    requireHash(
        configuration.manifest(), ActivatedQ4TransferCli.LIVE_MANIFEST_SHA256, "live manifest");

    ObjectMapper mapper = ActivatedDecisionProfileCli.mapper();
    List<ActivatedDecisionProfileCli.SourceCase> cases =
        ActivatedQ4TransferCli.loadLiveCases(mapper, configuration.records());
    requireShape(cases);

    List<Observation> observations = new ArrayList<>();
    PureJavaBackend backend =
        PureJavaBackend.loadActivatedAdapter(configuration.model(), configuration.adapter());
    ActivatedToolCallingModel activatedModel;
    try {
      activatedModel = new ActivatedToolCallingModel(backend, 1);
    } catch (RuntimeException | Error failure) {
      try {
        backend.close();
      } catch (RuntimeException | Error closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }

    ActivatedAdapterMetadata adapter;
    BackendDiagnostics diagnostics;
    try (ActivatedToolCallingModel model = activatedModel) {
      adapter = model.adapter();
      diagnostics = backend.diagnostics();
      int ordinal = 0;
      for (ActivatedDecisionProfileCli.SourceCase item : cases) {
        ordinal++;
        long started = System.nanoTime();
        ActivatedToolDecisionPolicy.Decision decision;
        float baseMargin;
        float specialistMargin;
        boolean physicallyShared;
        try (ActivatedToolTurn turn =
            model.openToolTurn(item.prompt(), ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
          ToolDecisionScore specialist =
              turn.scoreToolDecision(POLICY.callTokenId(), POLICY.noCallTokenId());
          ToolDecisionScore base =
              turn.scoreBaseToolDecision(POLICY.callTokenId(), POLICY.noCallTokenId());
          decision = POLICY.decide(base, specialist);
          baseMargin = base.callMargin();
          specialistMargin = specialist.callMargin();
          physicallyShared = turn.physicallySharesPrefix();
        }
        Observation observation =
            new Observation(
                item.id(),
                item.kind(),
                item.callExpected(),
                decision,
                baseMargin,
                specialistMargin,
                physicallyShared,
                (System.nanoTime() - started) / 1_000_000L);
        observations.add(observation);
        System.out.printf(
            "%2d/%d %-26s expected=%-5s decision=%-12s shared=%-5s %d ms%n",
            ordinal,
            cases.size(),
            item.id(),
            item.callExpected(),
            decision,
            physicallyShared,
            observation.elapsedMillis());
      }
    }

    Summary summary = summarize(observations);
    Report report =
        new Report(
            1,
            EXPERIMENT,
            Instant.now().toString(),
            configuration.modelsRevision(),
            ActivatedQ4TransferCli.LIVE_RECORDS_SHA256,
            ActivatedQ4TransferCli.LIVE_MANIFEST_SHA256,
            ActivatedHybridDevelopmentCli.MODEL_SHA256,
            adapter,
            BenchmarkEnvironment.capture(),
            diagnostics,
            POLICY,
            true,
            summary,
            observations);
    Path parent = configuration.output().toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    mapper.writeValue(configuration.output().toFile(), report);
    System.out.printf(
        "%s calls=%d/%d no-calls=%d/%d shared=%d/%d report=%s%n",
        summary.verdict(),
        summary.correctCallDecisions(),
        summary.calls(),
        summary.correctNoCalls(),
        summary.noCalls(),
        summary.physicallyShared(),
        summary.total(),
        configuration.output().toAbsolutePath());
    return summary.passed() ? 0 : 1;
  }

  static Summary summarize(List<Observation> observations) {
    Objects.requireNonNull(observations, "observations");
    int total = observations.size();
    int calls = Math.toIntExact(observations.stream().filter(Observation::callExpected).count());
    int noCalls = total - calls;
    int correctCalls =
        Math.toIntExact(
            observations.stream()
                .filter(Observation::callExpected)
                .filter(item -> item.decision().shouldCall())
                .count());
    int correctNoCalls =
        Math.toIntExact(
            observations.stream()
                .filter(item -> !item.callExpected())
                .filter(item -> !item.decision().shouldCall())
                .count());
    int physical =
        Math.toIntExact(observations.stream().filter(Observation::physicallyShared).count());
    boolean passed =
        total == 75
            && calls == 50
            && noCalls == 25
            && correctCalls >= 48
            && correctNoCalls >= 24
            && physical == total;
    return new Summary(
        total,
        calls,
        noCalls,
        correctCalls,
        correctNoCalls,
        physical,
        passed,
        passed ? "PASS" : "FAIL");
  }

  private static void requireShape(List<ActivatedDecisionProfileCli.SourceCase> cases) {
    long simple = cases.stream().filter(item -> "simple".equals(item.kind())).count();
    long multiple = cases.stream().filter(item -> "multiple".equals(item.kind())).count();
    long irrelevant = cases.stream().filter(item -> "irrelevance".equals(item.kind())).count();
    if (cases.size() != 75
        || simple != 25
        || multiple != 25
        || irrelevant != 25
        || cases.stream().filter(ActivatedDecisionProfileCli.SourceCase::callExpected).count()
            != 50) {
      throw new IllegalArgumentException("V21 live window must contain 25 cases per kind");
    }
  }

  private static void requireHash(Path path, String expected, String label) throws IOException {
    String actual = Hashing.sha256(path);
    if (!expected.equals(actual)) {
      throw new IllegalArgumentException(label + " digest does not match: " + actual);
    }
  }

  private static Path requiredFile(Map<String, String> values, String name) {
    Path path = Path.of(required(values, name));
    if (!Files.isRegularFile(path)) {
      throw new IllegalArgumentException("--" + name + " is not a regular file: " + path);
    }
    return path;
  }

  private static Path requiredDirectory(Map<String, String> values, String name) {
    Path path = Path.of(required(values, name));
    if (!Files.isDirectory(path)) {
      throw new IllegalArgumentException("--" + name + " is not a directory: " + path);
    }
    return path;
  }

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }
}
