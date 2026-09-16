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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolDecisionPolicy;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.TokenConstraint;
import com.integrallis.models.runtime.ToolDecisionScore;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Runs the frozen, exposed Java development gate before opening a sealed hybrid qualification set.
 */
final class ActivatedHybridDevelopmentCli {
  static final String EXPERIMENT = "qwen3-17b-hybrid-generation-v20";
  static final String SOURCE_SHA256 =
      "944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c";
  static final String MODEL_SHA256 =
      "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a";
  static final int CALL_TOKEN_ID = 4913;
  static final int NO_CALL_TOKEN_ID = 19536;
  static final int REQUIRED_CALL_DECISIONS = 48;
  static final int REQUIRED_NO_CALL_DECISIONS = 24;
  static final int REQUIRED_EXACT_TOOL_CALLS = 43;
  static final ActivatedToolDecisionPolicy POLICY =
      new ActivatedToolDecisionPolicy(
          CALL_TOKEN_ID, NO_CALL_TOKEN_ID, 4.8179874f, 1.3521204f, 33.705593f, 39.42147f);

  private static final Set<String> OPTIONS =
      Set.of("model", "adapter", "records", "output", "models-revision", "max-tokens", "case");

  private ActivatedHybridDevelopmentCli() {}

  record Configuration(
      Path model,
      Path adapter,
      Path records,
      Path output,
      String modelsRevision,
      int maxTokens,
      String caseId) {}

  record SourceCase(
      String id,
      String kind,
      boolean callExpected,
      ModelPrompt prompt,
      List<ToolSpec> tools,
      JsonNode expected) {
    SourceCase {
      Objects.requireNonNull(id, "id");
      Objects.requireNonNull(kind, "kind");
      Objects.requireNonNull(prompt, "prompt");
      tools = List.copyOf(Objects.requireNonNull(tools, "tools"));
      expected = Objects.requireNonNull(expected, "expected").deepCopy();
    }
  }

  record Observation(
      String id,
      String kind,
      boolean callExpected,
      ActivatedToolDecisionPolicy.Decision decision,
      float baseMargin,
      float specialistMargin,
      boolean physicallyShared,
      String output,
      boolean syntaxValid,
      boolean schemaValid,
      boolean exact,
      long elapsedMillis) {}

  record Summary(
      int total,
      int calls,
      int noCalls,
      int correctCallDecisions,
      int correctNoCalls,
      int strictSyntax,
      int schemaValid,
      int exactToolCalls,
      int physicallyShared,
      boolean passed,
      String verdict) {}

  record GenerationControls(double temperature, int maxTokens) {}

  record Report(
      int schemaVersion,
      String experiment,
      String createdAt,
      String modelsRevision,
      String sourceSha256,
      String modelSha256,
      ActivatedAdapterMetadata adapter,
      BenchmarkEnvironment environment,
      BackendDiagnostics backend,
      ActivatedToolDecisionPolicy policy,
      GenerationControls generation,
      boolean developmentDataExposed,
      String selectedCase,
      Summary summary,
      List<Observation> observations) {
    Report {
      observations = List.copyOf(observations);
    }
  }

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    Path model = requiredFile(values, "model");
    Path adapter = requiredDirectory(values, "adapter");
    Path records = requiredFile(values, "records");
    String revision = required(values, "models-revision");
    if (!revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact 40-character Git SHA");
    }
    int maxTokens = BenchmarkCliArguments.integer(values, "max-tokens", 128);
    if (maxTokens < 32 || maxTokens > 512) {
      throw new IllegalArgumentException("--max-tokens must be between 32 and 512");
    }
    String caseId = values.get("case");
    if (caseId != null && caseId.isBlank()) {
      throw new IllegalArgumentException("--case must not be blank");
    }
    Path output =
        Path.of(
            values.getOrDefault(
                "output", "build/reports/tool-qualification/activated-hybrid-v20.json"));
    return new Configuration(model, adapter, records, output, revision, maxTokens, caseId);
  }

  static int run(String[] args) throws IOException {
    Configuration configuration = parse(args);
    requireDigest(configuration.records(), SOURCE_SHA256, "exposed development records");
    requireDigest(configuration.model(), MODEL_SHA256, "Qwen3 1.7B Q4_K_M model");

    ObjectMapper mapper = ActivatedDecisionProfileCli.mapper();
    List<SourceCase> allCases = loadCases(mapper, configuration.records());
    List<SourceCase> selected = select(allCases, configuration.caseId());
    SamplingOptions options =
        SamplingOptions.builder().temperature(0).maxTokens(configuration.maxTokens()).build();
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
      for (SourceCase item : selected) {
        ordinal++;
        long started = System.nanoTime();
        ActivatedToolDecisionPolicy.Decision decision;
        float baseMargin;
        float specialistMargin;
        boolean physicallyShared;
        String output = "";
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
          if (decision.shouldCall()) {
            output = turn.generateToolCall(options, TokenConstraint.unrestricted());
          }
        }
        Observation observation =
            evaluate(
                mapper,
                item,
                decision,
                baseMargin,
                specialistMargin,
                physicallyShared,
                output,
                (System.nanoTime() - started) / 1_000_000L);
        observations.add(observation);
        System.out.printf(
            "%2d/%d %-18s expected=%-5s decision=%-12s exact=%-5s shared=%-5s %d ms%n",
            ordinal,
            selected.size(),
            item.id(),
            item.callExpected(),
            decision,
            observation.exact(),
            observation.physicallyShared(),
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
            SOURCE_SHA256,
            MODEL_SHA256,
            adapter,
            BenchmarkEnvironment.capture(),
            diagnostics,
            POLICY,
            new GenerationControls(0, configuration.maxTokens()),
            true,
            configuration.caseId(),
            summary,
            observations);
    Path parent = configuration.output().toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    mapper.writeValue(configuration.output().toFile(), report);
    System.out.printf(
        "%s calls=%d/%d no-calls=%d/%d exact=%d/%d syntax=%d/%d schema=%d/%d shared=%d/%d report=%s%n",
        summary.verdict(),
        summary.correctCallDecisions(),
        summary.calls(),
        summary.correctNoCalls(),
        summary.noCalls(),
        summary.exactToolCalls(),
        summary.calls(),
        summary.strictSyntax(),
        summary.total(),
        summary.schemaValid(),
        summary.total(),
        summary.physicallyShared(),
        summary.total(),
        configuration.output().toAbsolutePath());

    // A selected case is a mechanics diagnostic, not a qualification verdict.
    if (configuration.caseId() != null) {
      return observations.stream().allMatch(Observation::physicallyShared) ? 0 : 1;
    }
    return summary.passed() ? 0 : 1;
  }

  static Observation evaluate(
      ObjectMapper mapper,
      SourceCase item,
      ActivatedToolDecisionPolicy.Decision decision,
      float baseMargin,
      float specialistMargin,
      boolean physicallyShared,
      String output,
      long elapsedMillis) {
    Objects.requireNonNull(mapper, "mapper");
    Objects.requireNonNull(item, "item");
    Objects.requireNonNull(decision, "decision");
    if (!decision.shouldCall()) {
      return new Observation(
          item.id(),
          item.kind(),
          item.callExpected(),
          decision,
          baseMargin,
          specialistMargin,
          physicallyShared,
          output,
          true,
          true,
          !item.callExpected(),
          elapsedMillis);
    }

    var parsed = BfclCallMatcher.parseStrictCompletion(mapper, output);
    boolean syntaxValid = parsed.isPresent();
    List<ToolCall> calls = parsed.orElseGet(List::of);
    boolean schemaValid = syntaxValid && BfclCallMatcher.callsValidate(mapper, calls, item.tools());
    boolean exact =
        schemaValid
            && item.callExpected()
            && BfclCallMatcher.exactCallsMatch(mapper, calls, item.expected(), item.tools());
    return new Observation(
        item.id(),
        item.kind(),
        item.callExpected(),
        decision,
        baseMargin,
        specialistMargin,
        physicallyShared,
        output,
        syntaxValid,
        schemaValid,
        exact,
        elapsedMillis);
  }

  static Summary summarize(List<Observation> observations) {
    Objects.requireNonNull(observations, "observations");
    int total = observations.size();
    int calls = Math.toIntExact(observations.stream().filter(Observation::callExpected).count());
    int noCalls = total - calls;
    int correctCallDecisions =
        count(observations, item -> item.callExpected() && item.decision().shouldCall());
    int correctNoCalls =
        count(observations, item -> !item.callExpected() && !item.decision().shouldCall());
    int strictSyntax = count(observations, Observation::syntaxValid);
    int schemaValid = count(observations, Observation::schemaValid);
    int exactToolCalls = count(observations, item -> item.callExpected() && item.exact());
    int physicallyShared = count(observations, Observation::physicallyShared);
    boolean passed =
        total == 75
            && calls == 50
            && noCalls == 25
            && correctCallDecisions >= REQUIRED_CALL_DECISIONS
            && correctNoCalls >= REQUIRED_NO_CALL_DECISIONS
            && exactToolCalls >= REQUIRED_EXACT_TOOL_CALLS
            && strictSyntax == total
            && schemaValid == total
            && physicallyShared == total;
    return new Summary(
        total,
        calls,
        noCalls,
        correctCallDecisions,
        correctNoCalls,
        strictSyntax,
        schemaValid,
        exactToolCalls,
        physicallyShared,
        passed,
        passed ? "PASS" : "FAIL");
  }

  private static int count(
      List<Observation> observations, java.util.function.Predicate<Observation> predicate) {
    return Math.toIntExact(observations.stream().filter(predicate).count());
  }

  private static List<SourceCase> loadCases(ObjectMapper mapper, Path records) throws IOException {
    ActivatedDecisionProfileCli.LoadedCases loaded =
        ActivatedDecisionProfileCli.loadCases(mapper, records);
    Map<String, JsonNode> source = new LinkedHashMap<>();
    try (var lines = Files.lines(records)) {
      for (String line : lines.toList()) {
        JsonNode item = mapper.readTree(line);
        if ("adapter".equals(item.path("mode").asText())) {
          String id = requiredText(item, "id");
          if (source.put(id, item) != null) {
            throw new IllegalArgumentException("duplicate adapter record: " + id);
          }
        }
      }
    }

    List<SourceCase> cases = new ArrayList<>();
    for (ActivatedDecisionProfileCli.SourceCase item : loaded.cases()) {
      JsonNode raw = source.get(item.id());
      if (raw == null) {
        throw new IllegalArgumentException("missing adapter record: " + item.id());
      }
      JsonNode expected = raw.path("expected");
      if (!expected.isArray()) {
        throw new IllegalArgumentException("expected calls must be an array for " + item.id());
      }
      List<ToolSpec> tools = parseTools(mapper, raw.path("tools"));
      cases.add(
          new SourceCase(
              item.id(), item.kind(), item.callExpected(), item.prompt(), tools, expected));
    }
    return List.copyOf(cases);
  }

  private static List<SourceCase> select(List<SourceCase> cases, String caseId) {
    if (caseId == null) {
      return cases;
    }
    List<SourceCase> selected = cases.stream().filter(item -> item.id().equals(caseId)).toList();
    if (selected.isEmpty()) {
      throw new IllegalArgumentException("unknown --case: " + caseId);
    }
    return selected;
  }

  private static List<ToolSpec> parseTools(ObjectMapper mapper, JsonNode value) {
    if (!value.isArray() || value.isEmpty()) {
      throw new IllegalArgumentException("tools must be a nonempty array");
    }
    List<ToolSpec> tools = new ArrayList<>();
    for (JsonNode item : value) {
      JsonNode function = item.path("function");
      JsonNode parameters = function.path("parameters");
      if (!parameters.isObject()) {
        throw new IllegalArgumentException("tool parameters must be a JSON object");
      }
      try {
        tools.add(
            new ToolSpec(
                requiredText(function, "name"),
                function.path("description").asText(""),
                mapper.writeValueAsString(parameters)));
      } catch (com.fasterxml.jackson.core.JsonProcessingException failure) {
        throw new IllegalArgumentException("cannot serialize tool schema", failure);
      }
    }
    return List.copyOf(tools);
  }

  private static void requireDigest(Path path, String expected, String label) throws IOException {
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

  private static String requiredText(JsonNode item, String field) {
    String value = item.path(field).asText();
    if (value.isBlank()) {
      throw new IllegalArgumentException(field + " must not be blank");
    }
    return value;
  }
}
