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
import com.fasterxml.jackson.databind.SerializationFeature;
import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.TokenConstraint;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.ToolCallScanner;
import com.integrallis.models.runtime.chat.ToolSyntax;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Verifies semantic and token-exact retention across a physical 4096-token shared KV prefix. */
final class ActivatedLongContextQualificationCli {
  static final String POLICY_VERSION = "activated-long-context-retention-v1";
  static final String SUITE_RESOURCE = "tool-qualification/activated-long-context-v1.json";
  private static final double MINIMUM_NATIVE_CORRECT_RATE = 0.75;
  private static final String ZIPCODE = "88252";
  private static final int TEMPERATURE = 78;
  private static final Set<String> OPTIONS =
      Set.of("model", "adapter", "report", "models-revision", "max-tokens");
  private static final ToolSpec WEATHER =
      new ToolSpec(
          "get-weather-for-zipcode",
          "Gets weather for a given zipcode",
          "{\"type\":\"object\",\"properties\":{\"zipcode\":{\"type\":\"string\"}},"
              + "\"required\":[\"zipcode\"]}");

  private ActivatedLongContextQualificationCli() {}

  record Configuration(
      Path model, Path adapter, Path report, String modelsRevision, int maxTokens) {}

  record Case(String id, String route, String archiveCode) {}

  record Suite(int schemaVersion, String workload, int targetPrefixTokens, List<Case> cases) {
    Suite {
      cases = List.copyOf(cases);
    }
  }

  record SuiteIdentity(String resource, String sha256) {}

  record CaseResult(
      String id,
      int prefixTokens,
      boolean nativeCorrect,
      boolean retainedCorrect,
      boolean exactOutputMatch,
      boolean toolCallCorrect,
      boolean physicallyShared,
      String nativeOutput,
      String sharedOutput,
      String toolOutput,
      long nativeMillis,
      long sharedMillis,
      long toolMillis) {}

  record Summary(
      int attempts,
      int nativeCorrect,
      int retainedNativeCorrect,
      int exactOutputMatches,
      int correctToolCalls,
      int physicallyShared,
      int exactPrefixTier,
      boolean qualified) {}

  record Report(
      int schemaVersion,
      String createdAt,
      String policyVersion,
      String modelsRevision,
      String modelSha256,
      long modelSizeBytes,
      ActivatedAdapterMetadata adapter,
      long adapterDirectoryBytes,
      BenchmarkEnvironment environment,
      SuiteIdentity suite,
      int targetPrefixTokens,
      double minimumNativeCorrectRate,
      int maxTokens,
      List<CaseResult> cases,
      Summary summary,
      boolean qualified) {}

  private record PreparedPrompt(ModelPrompt selection, ModelPrompt result) {}

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    Path model = requiredFile(values, "model");
    Path adapter = requiredDirectory(values, "adapter");
    String revision = values.get("models-revision");
    if (revision == null || !revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact 40-character Git SHA");
    }
    int maxTokens = BenchmarkCliArguments.integer(values, "max-tokens", 32);
    if (maxTokens < 1 || maxTokens > 96) {
      throw new IllegalArgumentException("--max-tokens must be between 1 and 96");
    }
    Path report =
        Path.of(
            values.getOrDefault(
                "report", "build/reports/activated-long-context/qualification.json"));
    return new Configuration(model, adapter, report, revision, maxTokens);
  }

  static Suite loadSuite() throws IOException {
    ObjectMapper mapper = new ObjectMapper();
    try (InputStream input =
        ActivatedLongContextQualificationCli.class
            .getClassLoader()
            .getResourceAsStream(SUITE_RESOURCE)) {
      if (input == null) {
        throw new IOException("missing long-context qualification suite: " + SUITE_RESOURCE);
      }
      Suite suite = mapper.readValue(input, Suite.class);
      validateSuite(suite);
      return suite;
    }
  }

  static int run(String[] args) throws IOException {
    Configuration configuration = parse(args);
    Suite suite = loadSuite();
    SuiteIdentity suiteIdentity = suiteIdentity();
    List<CaseResult> results = new ArrayList<>();
    SamplingOptions responseOptions =
        SamplingOptions.builder().temperature(0).maxTokens(configuration.maxTokens()).build();
    SamplingOptions toolOptions = SamplingOptions.builder().temperature(0).maxTokens(48).build();
    ObjectMapper mapper = new ObjectMapper();

    try (PureJavaBackend backend =
            PureJavaBackend.loadActivatedAdapter(configuration.model(), configuration.adapter());
        ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend)) {
      for (Case item : suite.cases()) {
        PreparedPrompt prompt = exactPrefixPrompt(model.tokenizer(), model.adapter(), suite, item);
        long nativeStarted = System.nanoTime();
        String nativeOutput = model.generate(prompt.result(), responseOptions);
        long nativeMillis = (System.nanoTime() - nativeStarted) / 1_000_000L;

        String toolOutput;
        String sharedOutput;
        long toolMillis;
        long sharedMillis;
        boolean physicallyShared;
        int sharedPrefixTokens;
        try (ActivatedToolTurn turn = model.openToolTurn(prompt.selection())) {
          physicallyShared = turn.physicallySharesPrefix();
          sharedPrefixTokens = turn.sharedPrefixTokens();
          long toolStarted = System.nanoTime();
          toolOutput = turn.generateToolCall(toolOptions, TokenConstraint.unrestricted());
          toolMillis = (System.nanoTime() - toolStarted) / 1_000_000L;
          long sharedStarted = System.nanoTime();
          sharedOutput = turn.generateBaseResponse(prompt.result(), responseOptions);
          sharedMillis = (System.nanoTime() - sharedStarted) / 1_000_000L;
        }

        boolean nativeCorrect = correctAnswer(nativeOutput, item.archiveCode());
        boolean retainedCorrect = correctAnswer(sharedOutput, item.archiveCode());
        CaseResult result =
            new CaseResult(
                item.id(),
                sharedPrefixTokens,
                nativeCorrect,
                retainedCorrect,
                nativeOutput.equals(sharedOutput),
                correctToolCall(mapper, toolOutput),
                physicallyShared,
                nativeOutput,
                sharedOutput,
                toolOutput,
                nativeMillis,
                sharedMillis,
                toolMillis);
        results.add(result);
        System.out.printf(
            "%-12s native=%s retained=%s exact=%s tool=%s shared=%s%n",
            item.id(),
            result.nativeCorrect(),
            result.retainedCorrect(),
            result.exactOutputMatch(),
            result.toolCallCorrect(),
            result.physicallyShared());
      }

      Summary summary = summarize(results);
      Report report =
          new Report(
              1,
              Instant.now().toString(),
              POLICY_VERSION,
              configuration.modelsRevision(),
              Hashing.sha256(configuration.model()),
              Files.size(configuration.model()),
              model.adapter(),
              directoryBytes(configuration.adapter()),
              BenchmarkEnvironment.capture(),
              suiteIdentity,
              suite.targetPrefixTokens(),
              MINIMUM_NATIVE_CORRECT_RATE,
              configuration.maxTokens(),
              List.copyOf(results),
              summary,
              summary.qualified());
      write(configuration.report(), report);
      System.out.printf(
          "%s native=%d/%d retained=%d/%d exact=%d/%d tool=%d/%d report=%s%n",
          summary.qualified() ? "PASS" : "FAIL",
          summary.nativeCorrect(),
          summary.attempts(),
          summary.retainedNativeCorrect(),
          summary.nativeCorrect(),
          summary.exactOutputMatches(),
          summary.attempts(),
          summary.correctToolCalls(),
          summary.attempts(),
          configuration.report().toAbsolutePath());
      return summary.qualified() ? 0 : 1;
    }
  }

  static Summary summarize(List<CaseResult> results) {
    int attempts = results.size();
    int nativeCorrect = (int) results.stream().filter(CaseResult::nativeCorrect).count();
    int retainedNativeCorrect =
        (int)
            results.stream()
                .filter(CaseResult::nativeCorrect)
                .filter(CaseResult::retainedCorrect)
                .count();
    int exact = (int) results.stream().filter(CaseResult::exactOutputMatch).count();
    int tool = (int) results.stream().filter(CaseResult::toolCallCorrect).count();
    int shared = (int) results.stream().filter(CaseResult::physicallyShared).count();
    int exactPrefix =
        (int) results.stream().filter(result -> result.prefixTokens() == 4_096).count();
    int minimumNativeCorrect = (int) Math.ceil(attempts * MINIMUM_NATIVE_CORRECT_RATE);
    boolean qualified =
        attempts > 0
            && nativeCorrect >= minimumNativeCorrect
            && retainedNativeCorrect == nativeCorrect
            && exact == attempts
            && tool == attempts
            && shared == attempts
            && exactPrefix == attempts;
    return new Summary(
        attempts,
        nativeCorrect,
        retainedNativeCorrect,
        exact,
        tool,
        shared,
        exactPrefix,
        qualified);
  }

  private static PreparedPrompt exactPrefixPrompt(
      Tokenizer tokenizer, ActivatedAdapterMetadata adapter, Suite suite, Case item) {
    int low = 0;
    int high = suite.targetPrefixTokens() * 2;
    while (low <= high) {
      int repetitions = (low + high) >>> 1;
      String user = userText(item, " transit".repeat(repetitions));
      ModelPrompt selection =
          ChatTemplate.CHATML_NO_THINK.render(List.of(ChatMessage.user(user)), List.of(WEATHER));
      int prefix = activationBoundary(tokenizer.encode(selection), adapter);
      if (prefix == suite.targetPrefixTokens()) {
        ToolCall call = ToolCall.of(0, WEATHER.name(), "{\"zipcode\":\"" + ZIPCODE + "\"}");
        String result =
            "{\"zipcode\":\""
                + ZIPCODE
                + "\",\"conditions\":\"Raining cats and dogs\","
                + "\"temperatureInFahrenheit\":"
                + TEMPERATURE
                + "}";
        ModelPrompt completed =
            ChatTemplate.CHATML_NO_THINK.render(
                List.of(
                    ChatMessage.user(user),
                    ChatMessage.assistantToolCalls("", List.of(call)),
                    ChatMessage.tool(WEATHER.name(), result)),
                List.of(WEATHER));
        return new PreparedPrompt(selection, completed);
      }
      if (prefix < suite.targetPrefixTokens()) {
        low = repetitions + 1;
      } else {
        high = repetitions - 1;
      }
    }
    throw new IllegalStateException(
        "could not construct an exact " + suite.targetPrefixTokens() + "-token prefix");
  }

  private static String userText(Case item, String filler) {
    return "Transit archive fact: route "
        + item.route()
        + " has archive code "
        + item.archiveCode()
        + ".\nBackground records:"
        + filler
        + "\nCall get-weather-for-zipcode for "
        + ZIPCODE
        + ". After the tool result, reply with exactly: "
        + item.archiveCode()
        + " | "
        + TEMPERATURE
        + "F";
  }

  private static int activationBoundary(int[] promptTokens, ActivatedAdapterMetadata adapter) {
    int[] invocation = adapter.invocationTokens().stream().mapToInt(Integer::intValue).toArray();
    for (int start = promptTokens.length - invocation.length; start >= 0; start--) {
      boolean match = true;
      for (int index = 0; index < invocation.length; index++) {
        if (promptTokens[start + index] != invocation[index]) {
          match = false;
          break;
        }
      }
      if (match) {
        return start;
      }
    }
    throw new IllegalStateException("rendered prompt omitted the adapter invocation sequence");
  }

  private static boolean correctAnswer(String output, String code) {
    String normalized = output == null ? "" : output.toUpperCase(java.util.Locale.ROOT);
    return normalized.contains(code) && normalized.contains(Integer.toString(TEMPERATURE));
  }

  private static boolean correctToolCall(ObjectMapper mapper, String output) {
    ToolCallScanner.Result scan = ToolCallScanner.scan(output, ToolSyntax.QWEN, List.of(WEATHER));
    if (scan.toolCalls().size() != 1) {
      return false;
    }
    ToolCall call = scan.toolCalls().getFirst();
    if (!WEATHER.name().equals(call.name())) {
      return false;
    }
    try {
      JsonNode arguments = mapper.readTree(call.argumentsJson());
      return arguments.isObject()
          && arguments.size() == 1
          && ZIPCODE.equals(arguments.path("zipcode").asText());
    } catch (IOException malformed) {
      return false;
    }
  }

  private static void validateSuite(Suite suite) {
    if (suite.schemaVersion() != 1
        || suite.workload() == null
        || suite.workload().isBlank()
        || suite.targetPrefixTokens() != 4_096
        || suite.cases().size() != 8) {
      throw new IllegalArgumentException("invalid activated long-context qualification suite");
    }
    Set<String> ids = new HashSet<>();
    Set<String> codes = new HashSet<>();
    for (Case item : suite.cases()) {
      if (item.id() == null
          || item.id().isBlank()
          || item.route() == null
          || item.route().isBlank()
          || item.archiveCode() == null
          || item.archiveCode().isBlank()
          || !ids.add(item.id())
          || !codes.add(item.archiveCode())) {
        throw new IllegalArgumentException("long-context suite cases must be complete and unique");
      }
    }
  }

  private static SuiteIdentity suiteIdentity() throws IOException {
    try (InputStream input =
        ActivatedLongContextQualificationCli.class
            .getClassLoader()
            .getResourceAsStream(SUITE_RESOURCE)) {
      if (input == null) {
        throw new IOException("missing long-context qualification suite: " + SUITE_RESOURCE);
      }
      return new SuiteIdentity(
          SUITE_RESOURCE, Hashing.sha256(new String(input.readAllBytes(), StandardCharsets.UTF_8)));
    }
  }

  private static Path requiredFile(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank() || !Files.isRegularFile(Path.of(value))) {
      throw new IllegalArgumentException("--" + name + " must name an existing file");
    }
    return Path.of(value);
  }

  private static Path requiredDirectory(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank() || !Files.isDirectory(Path.of(value))) {
      throw new IllegalArgumentException("--" + name + " must name an existing directory");
    }
    return Path.of(value);
  }

  private static long directoryBytes(Path directory) throws IOException {
    try (var files = Files.walk(directory)) {
      return files
          .filter(Files::isRegularFile)
          .mapToLong(ActivatedLongContextQualificationCli::size)
          .sum();
    }
  }

  private static long size(Path path) {
    try {
      return Files.size(path);
    } catch (IOException failure) {
      throw new IllegalStateException("could not measure " + path, failure);
    }
  }

  private static void write(Path output, Report report) throws IOException {
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.writeString(temporary, mapper.writeValueAsString(report) + System.lineSeparator());
    try {
      Files.move(
          temporary, output, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    } catch (AtomicMoveNotSupportedException unsupported) {
      Files.move(temporary, output, StandardCopyOption.REPLACE_EXISTING);
    }
  }
}
