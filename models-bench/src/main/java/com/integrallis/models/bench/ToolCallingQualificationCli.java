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
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.GenerationLoop;
import com.integrallis.models.runtime.InModelContrastiveEmbeddingBackend;
import com.integrallis.models.runtime.ToolCallTokenConstraints;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ToolCallScanner;
import com.integrallis.models.runtime.chat.ToolSpecRetriever;
import com.integrallis.models.runtime.chat.ToolSyntax;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Runs the shared small-model tool-calling gate through the pure-Java backend. */
final class ToolCallingQualificationCli {
  static final String POLICY_VERSION = "small-model-tool-conformance-v1";

  private static final int PASS = 0;
  private static final int FAIL = 1;
  private static final Set<String> OPTIONS =
      Set.of("candidate", "model", "report", "max-tokens", "models-revision", "case");

  private ToolCallingQualificationCli() {}

  record Configuration(
      ToolCallingCandidate candidate,
      Path model,
      Path report,
      int maxTokens,
      String modelsRevision,
      String caseId) {}

  record GenerationControls(double temperature, int maxTokens, String promptTemplate) {}

  record SuiteIdentity(
      String id,
      String sha256,
      String sourceRepository,
      String sourceRevision,
      String sourcePath) {}

  record FollowUpResult(
      String id,
      boolean applicable,
      long endToEndMillis,
      String output,
      boolean conversational,
      boolean grounded,
      boolean repeatedToolCall,
      boolean passed,
      List<String> diagnostics) {
    FollowUpResult {
      diagnostics = List.copyOf(diagnostics);
    }
  }

  record Report(
      int schemaVersion,
      String createdAt,
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
      int plannedAttempts,
      List<String> inapplicableCases,
      boolean complete,
      Needle2ToolQualification.Summary summary,
      FollowUpResult followUp,
      boolean qualified,
      String verdict,
      List<Needle2ToolQualification.CaseResult> cases) {
    Report {
      inapplicableCases = List.copyOf(inapplicableCases);
      cases = List.copyOf(cases);
    }
  }

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    String candidateValue = values.get("candidate");
    if (candidateValue == null || candidateValue.isBlank()) {
      throw new IllegalArgumentException("--candidate is required");
    }
    ToolCallingCandidate candidate = ToolCallingCandidate.parse(candidateValue);
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
                "report", "build/reports/tool-qualification/" + candidate.key() + ".json"));
    String caseId = values.get("case");
    if (caseId != null && caseId.isBlank()) {
      throw new IllegalArgumentException("--case must not be blank");
    }
    return new Configuration(candidate, model, report, maxTokens, revision, caseId);
  }

  static int run(String[] args) throws IOException {
    Configuration configuration = parse(args);
    ToolCallingCandidate candidate = configuration.candidate();
    String artifactSha256 = Hashing.sha256(configuration.model());
    if (!candidate.artifactSha256().equals(artifactSha256)) {
      throw new IllegalArgumentException(
          "artifact digest does not match the pinned "
              + candidate.key()
              + " release: "
              + artifactSha256);
    }

    ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    Needle2ToolQualification.Suite suite = Needle2ToolQualification.loadSuite(mapper);
    List<String> inapplicable = inapplicableCases(suite, candidate);
    List<Needle2ToolQualification.Case> selected =
        selectCases(suite, candidate, configuration.caseId());
    String createdAt = Instant.now().toString();
    BenchmarkEnvironment environment = BenchmarkEnvironment.capture();
    SamplingOptions options =
        SamplingOptions.builder().temperature(0.0f).maxTokens(configuration.maxTokens()).build();
    List<Needle2ToolQualification.CaseResult> results = new ArrayList<>();
    BackendDiagnostics diagnostics;
    FollowUpResult followUp;
    long loadStart = System.nanoTime();
    try (PureJavaBackend backend = PureJavaBackend.load(configuration.model())) {
      System.out.printf(
          "loaded %s in %.1f ms%n",
          candidate.modelId(), (System.nanoTime() - loadStart) / 1_000_000.0);
      GenerationLoop generation = new GenerationLoop(backend);
      diagnostics = backend.diagnostics();
      for (Needle2ToolQualification.Case item : selected) {
        List<ToolSpec> declaredTools = item.toolSpecs(mapper);
        List<ToolSpec> tools = selectedTools(candidate, backend, item.query(), declaredTools);
        ModelPrompt prompt =
            candidate.template().render(List.of(ChatMessage.user(item.query())), tools);
        System.out.printf(
            "%-24s prompt=%d tools=%s%n",
            item.id(),
            backend.tokenizer().encode(prompt).length,
            tools.stream().map(ToolSpec::name).toList());
        long start = System.nanoTime();
        String output = generate(candidate, backend, generation, prompt, options, tools);
        long elapsedMillis = (System.nanoTime() - start) / 1_000_000L;
        Needle2ToolQualification.CaseResult result =
            Needle2ToolQualification.evaluate(
                mapper, item, output, elapsedMillis, candidate.template().toolSyntax());
        results.add(result);
        writeReport(
            configuration,
            artifactSha256,
            suite,
            createdAt,
            environment,
            diagnostics,
            selected.size(),
            inapplicable,
            results,
            null,
            false,
            mapper);
        System.out.printf(
            "%-24s %s %6d ms selection=%s schema=%s arguments=%d/%d%n",
            item.id(),
            result.passed() ? "PASS" : "FAIL",
            elapsedMillis,
            result.selectionExact(),
            result.schemaValid() && result.declaredArgumentsOnly(),
            result.expectedArgumentMatches(),
            result.expectedArguments());
      }
      followUp =
          runFollowUp(candidate, backend, generation, suite, mapper, configuration.maxTokens());
      if (followUp.applicable()) {
        System.out.printf(
            "%-24s %s %6d ms conversational=%s grounded=%s repeated-call=%s%n",
            followUp.id(),
            followUp.passed() ? "PASS" : "FAIL",
            followUp.endToEndMillis(),
            followUp.conversational(),
            followUp.grounded(),
            followUp.repeatedToolCall());
      }
    }

    Needle2ToolQualification.Summary summary = Needle2ToolQualification.summarize(results);
    writeReport(
        configuration,
        artifactSha256,
        suite,
        createdAt,
        environment,
        diagnostics,
        selected.size(),
        inapplicable,
        results,
        followUp,
        true,
        mapper);
    boolean qualified = summary.qualified() && (!followUp.applicable() || followUp.passed());
    System.out.printf(
        "%s selection=%.3f schema=%.3f arguments=%.3f refusal=%.3f follow-up=%s report=%s%n",
        qualified ? "PASS" : "FAIL",
        summary.toolSelectionExactRate(),
        summary.schemaValidityRate(),
        summary.expectedArgumentAccuracy(),
        summary.refusalAccuracy(),
        followUp.applicable() ? followUp.passed() : "host-rendered",
        configuration.report().toAbsolutePath());
    return qualified ? PASS : FAIL;
  }

  private static FollowUpResult runFollowUp(
      ToolCallingCandidate candidate,
      PureJavaBackend backend,
      GenerationLoop generation,
      Needle2ToolQualification.Suite suite,
      ObjectMapper mapper,
      int maxTokens) {
    if (candidate == ToolCallingCandidate.NEEDLE2) {
      return new FollowUpResult(
          "tool-result-follow-up",
          false,
          0,
          "",
          false,
          false,
          false,
          true,
          List.of(
              "Needle 2 is a selector; its host-side typed result renderer completes the turn"));
    }
    Needle2ToolQualification.Case weather =
        suite.cases().stream()
            .filter(item -> item.id().equals("spring-weather-zipcode"))
            .findFirst()
            .orElseThrow();
    List<ToolSpec> tools = weather.toolSpecs(mapper);
    String result =
        "{\"zipcode\":\"88252\",\"conditions\":\"Raining cats and dogs\","
            + "\"temperatureInFahrenheit\":78}";
    ModelPrompt prompt =
        candidate
            .template()
            .render(
                List.of(
                    ChatMessage.user(weather.query()),
                    ChatMessage.assistantToolCalls(
                        "",
                        List.of(
                            ToolCall.of(0, "get-weather-for-zipcode", "{\"zipcode\":\"88252\"}"))),
                    ChatMessage.tool("get-weather-for-zipcode", result)),
                tools);
    SamplingOptions options =
        SamplingOptions.builder().temperature(0.0f).maxTokens(Math.min(maxTokens, 96)).build();
    long start = System.nanoTime();
    String output = generation.generate(prompt, options);
    long elapsedMillis = (System.nanoTime() - start) / 1_000_000L;
    return evaluateFollowUp(
        mapper, candidate.template().toolSyntax(), tools, output, elapsedMillis);
  }

  static FollowUpResult evaluateFollowUp(
      ObjectMapper mapper,
      ToolSyntax syntax,
      List<ToolSpec> tools,
      String output,
      long elapsedMillis) {
    String generated = output == null ? "" : output.strip();
    List<String> diagnostics = new ArrayList<>();
    boolean repeatedCall = ToolCallScanner.scan(generated, syntax, tools).hasCalls();
    if (repeatedCall) {
      diagnostics.add("model requested another tool instead of answering from the result");
    }
    boolean rawJson = false;
    if (generated.startsWith("{") && generated.endsWith("}")) {
      try {
        rawJson = mapper.readTree(generated).isObject();
      } catch (IOException ignored) {
        // Malformed JSON is assessed as ordinary non-conversational output below.
      }
    }
    boolean conversational =
        !generated.isBlank()
            && !rawJson
            && !generated.contains("<tool_call>")
            && !generated.contains("<function name=");
    if (!conversational) {
      diagnostics.add("follow-up was empty, raw JSON, or tool protocol instead of prose");
    }
    String normalized = generated.toLowerCase(java.util.Locale.ROOT);
    boolean grounded = normalized.contains("78") && normalized.contains("rain");
    if (!grounded) {
      diagnostics.add("follow-up did not preserve the temperature and conditions");
    }
    boolean passed = conversational && grounded && !repeatedCall;
    return new FollowUpResult(
        "tool-result-follow-up",
        true,
        elapsedMillis,
        generated,
        conversational,
        grounded,
        repeatedCall,
        passed,
        diagnostics);
  }

  private static String generate(
      ToolCallingCandidate candidate,
      PureJavaBackend backend,
      GenerationLoop generation,
      ModelPrompt prompt,
      SamplingOptions options,
      List<ToolSpec> tools) {
    if (candidate == ToolCallingCandidate.NEEDLE2) {
      return generation.generate(
          prompt,
          options,
          ToolCallTokenConstraints.compile(
                  backend.tokenizer(),
                  candidate.template().toolSyntax(),
                  tools,
                  ignored -> List.of())
              .orElseThrow());
    }
    return generation.generate(prompt, options);
  }

  private static List<ToolSpec> selectedTools(
      ToolCallingCandidate candidate,
      PureJavaBackend backend,
      String query,
      List<ToolSpec> declaredTools) {
    if (candidate != ToolCallingCandidate.NEEDLE2 || declaredTools.size() <= 5) {
      return declaredTools;
    }
    ToolSpecRetriever retriever =
        new ToolSpecRetriever(new InModelContrastiveEmbeddingBackend(backend), declaredTools);
    return retriever.select(query, 5).stream().map(ToolSpecRetriever.Match::tool).toList();
  }

  private static List<Needle2ToolQualification.Case> selectCases(
      Needle2ToolQualification.Suite suite, ToolCallingCandidate candidate, String selectedCase) {
    List<Needle2ToolQualification.Case> applicable =
        suite.cases().stream()
            .filter(
                item ->
                    candidate.template().toolSyntax().parallelCalls()
                        || item.expectedCalls().size() <= 1)
            .toList();
    if (selectedCase == null) {
      return applicable;
    }
    List<Needle2ToolQualification.Case> selected =
        applicable.stream().filter(item -> item.id().equals(selectedCase)).toList();
    if (selected.isEmpty()) {
      throw new IllegalArgumentException(
          "unknown or protocol-inapplicable tool qualification case: " + selectedCase);
    }
    return selected;
  }

  private static List<String> inapplicableCases(
      Needle2ToolQualification.Suite suite, ToolCallingCandidate candidate) {
    if (candidate.template().toolSyntax().parallelCalls()) {
      return List.of();
    }
    return suite.cases().stream()
        .filter(item -> item.expectedCalls().size() > 1)
        .map(Needle2ToolQualification.Case::id)
        .toList();
  }

  private static void writeReport(
      Configuration configuration,
      String artifactSha256,
      Needle2ToolQualification.Suite suite,
      String createdAt,
      BenchmarkEnvironment environment,
      BackendDiagnostics diagnostics,
      int plannedAttempts,
      List<String> inapplicable,
      List<Needle2ToolQualification.CaseResult> results,
      FollowUpResult followUp,
      boolean complete,
      ObjectMapper mapper)
      throws IOException {
    ToolCallingCandidate candidate = configuration.candidate();
    Needle2ToolQualification.Summary summary = Needle2ToolQualification.summarize(results);
    boolean qualified =
        complete
            && summary.qualified()
            && (followUp == null || !followUp.applicable() || followUp.passed());
    Report report =
        new Report(
            1,
            createdAt,
            POLICY_VERSION,
            configuration.modelsRevision(),
            candidate.modelId(),
            candidate.modelName(),
            "pure-java",
            Needle2ToolQualificationCli.implementationVersion(configuration.modelsRevision()),
            artifactSha256,
            Files.size(configuration.model()),
            suiteIdentity(suite),
            new GenerationControls(0.0, configuration.maxTokens(), candidate.template().id()),
            environment,
            diagnostics,
            plannedAttempts,
            inapplicable,
            complete,
            summary,
            followUp,
            qualified,
            qualified ? "PASS" : "FAIL",
            results);
    write(configuration.report(), mapper, report);
  }

  private static SuiteIdentity suiteIdentity(Needle2ToolQualification.Suite suite)
      throws IOException {
    String suiteText;
    try (InputStream input =
        ToolCallingQualificationCli.class
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

  private static void write(Path output, ObjectMapper mapper, Report report) throws IOException {
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.writeString(temporary, mapper.writeValueAsString(report) + System.lineSeparator());
    try {
      Files.move(
          temporary, output, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    } catch (java.nio.file.AtomicMoveNotSupportedException unsupported) {
      Files.move(temporary, output, StandardCopyOption.REPLACE_EXISTING);
    }
  }
}
