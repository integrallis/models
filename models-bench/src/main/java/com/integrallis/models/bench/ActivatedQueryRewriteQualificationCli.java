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
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.TokenConstraint;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Small, reproducible smoke for an activated query-rewrite specialist.
 *
 * <p>The conversation is the public IBM model-card example. It deliberately uses the exact Granite
 * control tokens and appends the adapter-provided invocation as a control segment, so tokenization
 * cannot silently turn the activation marker into ordinary prose.
 */
final class ActivatedQueryRewriteQualificationCli {
  private static final Set<String> OPTIONS =
      Set.of("model", "adapter", "models-revision", "suite", "report");
  private static final ObjectMapper JSON = new ObjectMapper();

  private ActivatedQueryRewriteQualificationCli() {}

  record Configuration(Path model, Path adapter, String modelsRevision, Path suite, Path report) {}

  record Suite(int schemaVersion, String suiteId, JsonNode source, List<Case> cases) {}

  record Case(String id, List<Message> messages, String expectedRewrite) {}

  record Message(String role, String text) {}

  record CaseResult(
      String id,
      boolean physicallyShared,
      int sharedPrefixTokens,
      boolean structured,
      String expectedRewrite,
      String actualRewrite,
      String output) {}

  record SuiteReport(
      int schemaVersion,
      String createdAt,
      String modelsRevision,
      String suiteId,
      JsonNode source,
      List<CaseResult> cases,
      boolean executionPassed) {}

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    Path model = Path.of(required(values, "model"));
    Path adapter = Path.of(required(values, "adapter"));
    String revision = required(values, "models-revision");
    if (!revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact 40-character Git SHA");
    }
    Path suite = optionalPath(values, "suite");
    Path report = optionalPath(values, "report");
    if ((suite == null) != (report == null)) {
      throw new IllegalArgumentException("--suite and --report must be supplied together");
    }
    return new Configuration(model, adapter, revision, suite, report);
  }

  static ModelPrompt modelCardPrompt(String invocation) {
    return ModelPrompt.builder()
        .control("<|start_of_role|>system<|end_of_role|><|end_of_text|>\n")
        .control("<|start_of_role|>user<|end_of_role|>Who is the CEO of Apple?<|end_of_text|>\n")
        .control(
            "<|start_of_role|>assistant<|end_of_role|>Tim Cook is the CEO of Apple.<|end_of_text|>\n")
        .control("<|start_of_role|>user<|end_of_role|>and for Microsoft?<|end_of_text|>\n")
        .control(invocation)
        .build();
  }

  static int run(String[] args) throws Exception {
    Configuration configuration = parse(args);
    if (configuration.suite() != null) {
      return runSuite(configuration);
    }
    try (PureJavaBackend backend =
            PureJavaBackend.loadActivatedAdapter(configuration.model(), configuration.adapter());
        ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn turn =
            model.openToolTurn(
                modelCardPrompt(model.adapter().invocationText()),
                ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
      String output =
          turn.generateToolCall(
              SamplingOptions.builder().temperature(0).maxTokens(64).build(),
              TokenConstraint.unrestricted());
      JsonNode parsed = JSON.readTree(output);
      String rewrite = parsed.path("rewritten_question").asText();
      boolean qualified =
          turn.physicallySharesPrefix()
              && rewrite.toLowerCase(java.util.Locale.ROOT).contains("microsoft")
              && rewrite.toLowerCase(java.util.Locale.ROOT).contains("ceo");
      System.out.printf(
          "%s shared=%s prefix=%d rewrite=%s%n",
          qualified ? "PASS" : "FAIL",
          turn.physicallySharesPrefix(),
          turn.sharedPrefixTokens(),
          output);
      return qualified ? 0 : 1;
    }
  }

  private static int runSuite(Configuration configuration) throws IOException {
    Suite suite = JSON.readValue(configuration.suite().toFile(), Suite.class);
    validateSuite(suite);
    List<CaseResult> results = new ArrayList<>();
    try (PureJavaBackend backend =
            PureJavaBackend.loadActivatedAdapter(configuration.model(), configuration.adapter());
        ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1)) {
      SamplingOptions options = SamplingOptions.builder().temperature(0).maxTokens(80).build();
      for (Case item : suite.cases()) {
        try (ActivatedToolTurn turn =
            model.openToolTurn(
                prompt(item.messages(), model.adapter().invocationText()),
                ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
          String output = turn.generateToolCall(options, TokenConstraint.unrestricted());
          JsonNode parsed;
          try {
            parsed = JSON.readTree(output);
          } catch (IOException invalidJson) {
            parsed = null;
          }
          String rewrite = parsed == null ? "" : parsed.path("rewritten_question").asText("");
          boolean structured = parsed != null && parsed.isObject() && !rewrite.isBlank();
          results.add(
              new CaseResult(
                  item.id(),
                  turn.physicallySharesPrefix(),
                  turn.sharedPrefixTokens(),
                  structured,
                  item.expectedRewrite(),
                  rewrite,
                  output));
          System.out.printf(
              "%s structured=%s shared=%s rewrite=%s%n",
              item.id(), structured, turn.physicallySharesPrefix(), rewrite);
        }
      }
    }
    boolean passed =
        results.stream().allMatch(result -> result.structured() && result.physicallyShared());
    SuiteReport report =
        new SuiteReport(
            1,
            Instant.now().toString(),
            configuration.modelsRevision(),
            suite.suiteId(),
            suite.source(),
            List.copyOf(results),
            passed);
    JSON.writerWithDefaultPrettyPrinter().writeValue(configuration.report().toFile(), report);
    return passed ? 0 : 1;
  }

  static ModelPrompt prompt(List<Message> messages, String invocation) {
    ModelPrompt.Builder prompt =
        ModelPrompt.builder().control("<|start_of_role|>system<|end_of_role|><|end_of_text|>\n");
    for (Message message : messages) {
      prompt.control(
          "<|start_of_role|>"
              + message.role()
              + "<|end_of_role|>"
              + message.text()
              + "<|end_of_text|>\n");
    }
    return prompt.control(invocation).build();
  }

  private static void validateSuite(Suite suite) {
    if (suite.schemaVersion() != 1
        || suite.suiteId() == null
        || suite.source() == null
        || suite.cases() == null
        || suite.cases().isEmpty()) {
      throw new IllegalArgumentException("invalid query-rewrite suite");
    }
    for (Case item : suite.cases()) {
      if (item.id() == null
          || item.id().isBlank()
          || item.expectedRewrite() == null
          || item.expectedRewrite().isBlank()
          || item.messages() == null
          || item.messages().isEmpty()
          || !"user".equals(item.messages().getLast().role())) {
        throw new IllegalArgumentException("invalid query-rewrite case: " + item.id());
      }
      for (Message message : item.messages()) {
        if (!("user".equals(message.role()) || "assistant".equals(message.role()))
            || message.text() == null
            || message.text().isBlank()) {
          throw new IllegalArgumentException("invalid message in case: " + item.id());
        }
      }
    }
  }

  private static Path optionalPath(Map<String, String> values, String name) {
    String value = values.get(name);
    return value == null ? null : Path.of(value);
  }

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }
}
