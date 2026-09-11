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
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.GenerationMetrics;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.ToolCallTokenConstraints;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatRole;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.VirtualChatModel;
import com.integrallis.vectors.core.VectorRuntimeCapabilities;
import com.integrallis.vectors.core.VectorUtil;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** Runs one fresh-process arm for a Qwen control or Needle 2 plus SmolLM2 virtual model. */
final class NeedleSmolHybridQualificationCli {
  static final String POLICY_VERSION = "needle-smollm-virtual-chat-tool-v1";
  static final String CONTROL_SHA256 =
      "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a";
  static final String SMOLLM_SHA256 =
      "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201";
  static final String QWEN3_0_6B_SHA256 =
      "da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4";
  static final String NEEDLE_SHA256 =
      "b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba";

  private static final int PASS = 0;
  private static final int FAIL = 1;
  private static final ObjectMapper JSON =
      new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
  private static final Set<String> OPTIONS =
      Set.of(
          "arm",
          "control-model",
          "chat-model",
          "chat-profile",
          "tool-model",
          "tool-profile",
          "models-revision",
          "run-id",
          "max-tokens",
          "report");
  private static final String SYSTEM =
      "Answer directly and briefly. Use a declared tool whenever the user asks for one. After a "
          + "tool result, confirm the result in natural language and include every returned id.";
  private static final List<ToolSpec> TOOLS =
      List.of(
          new ToolSpec(
              "remember",
              "Store one durable user fact.",
              "{\"type\":\"object\",\"properties\":{\"fact\":{\"type\":\"string\"}},"
                  + "\"required\":[\"fact\"]}"),
          new ToolSpec(
              "recall",
              "Search durable user facts.",
              "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},"
                  + "\"required\":[\"query\"]}"));
  private static final List<Step> STEPS =
      List.of(
          Step.prose("cold-prose", ChatMessage.user("Say hello in one short sentence."), "hello"),
          Step.prose(
              "warm-prose",
              ChatMessage.user(
                  "In one short sentence, explain what long-term memory gives an assistant."),
              "memory"),
          Step.tool(
              "first-tool-switch",
              ChatMessage.user("Call remember with fact exactly \"I prefer aisle seats.\""),
              "remember",
              "{\"fact\":\"I prefer aisle seats.\"}"),
          Step.prose(
              "tool-result",
              "chat",
              ChatMessage.tool("remember", "{\"stored\":true,\"memoryId\":\"mem-2048\"}"),
              "mem-2048"),
          Step.prose(
              "prose-switch-back",
              ChatMessage.user("What memory record id was assigned? Reply with only the id."),
              "mem-2048"),
          Step.tool(
              "tool-switch-back",
              ChatMessage.user("Call recall with query exactly \"seat preference\"."),
              "recall",
              "{\"query\":\"seat preference\"}"));
  private static final String PROTOCOL_SHA256 = Hashing.sha256(protocolIdentity());

  private NeedleSmolHybridQualificationCli() {}

  enum Arm {
    CONTROL,
    HYBRID;

    static Arm parse(String value) {
      if (value == null || value.isBlank()) {
        throw new IllegalArgumentException("--arm is required");
      }
      try {
        return valueOf(value.trim().toUpperCase(Locale.ROOT));
      } catch (IllegalArgumentException failure) {
        throw new IllegalArgumentException("--arm must be control or hybrid: " + value, failure);
      }
    }
  }

  enum ToolProfile {
    NEEDLE2("needle2", NEEDLE_SHA256, "needle-2", ChatTemplate.NEEDLE2, "needle2"),
    QWEN3_1_7B(
        "qwen3-1.7b", CONTROL_SHA256, "qwen-1.7b-tools", ChatTemplate.CHATML_NO_THINK, "qwen3");

    private final String id;
    private final String sha256;
    private final String memberId;
    private final ChatTemplate template;
    private final String architecture;

    ToolProfile(
        String id, String sha256, String memberId, ChatTemplate template, String architecture) {
      this.id = id;
      this.sha256 = sha256;
      this.memberId = memberId;
      this.template = template;
      this.architecture = architecture;
    }

    static ToolProfile parse(String value) {
      for (ToolProfile profile : values()) {
        if (profile.id.equals(value)) {
          return profile;
        }
      }
      throw new IllegalArgumentException("--tool-profile must be needle2 or qwen3-1.7b: " + value);
    }

    String memberId() {
      return memberId;
    }
  }

  enum ChatProfile {
    SMOLLM2_360M("smollm2-360m", SMOLLM_SHA256, "smollm2-360m", "llama"),
    QWEN3_0_6B("qwen3-0.6b", QWEN3_0_6B_SHA256, "qwen-0.6b-chat", "qwen3");

    private final String id;
    private final String sha256;
    private final String memberId;
    private final String architecture;

    ChatProfile(String id, String sha256, String memberId, String architecture) {
      this.id = id;
      this.sha256 = sha256;
      this.memberId = memberId;
      this.architecture = architecture;
    }

    static ChatProfile parse(String value) {
      for (ChatProfile profile : values()) {
        if (profile.id.equals(value)) {
          return profile;
        }
      }
      throw new IllegalArgumentException(
          "--chat-profile must be smollm2-360m or qwen3-0.6b: " + value);
    }

    String memberId() {
      return memberId;
    }
  }

  record Configuration(
      Arm arm,
      Path controlModel,
      Path chatModel,
      ChatProfile chatProfile,
      Path toolModel,
      ToolProfile toolProfile,
      String modelsRevision,
      String runId,
      int maxTokens,
      Path report) {}

  record Assessment(boolean passed, List<String> diagnostics) {
    Assessment {
      diagnostics = List.copyOf(diagnostics);
    }
  }

  record ArtifactEvidence(
      String role, String sha256, long sizeBytes, boolean loaded, String architecture) {}

  record TurnMetrics(
      long tokenizationMillis,
      long promptPreparationMillis,
      long prefillMillis,
      long timeToFirstTokenMillis,
      long decodeMillis,
      long totalMillis,
      int promptTokens,
      int completionTokens,
      int cacheInputTokens,
      int cacheReadInputTokens,
      int cacheWriteInputTokens,
      double decodeTokensPerSecond) {}

  record TurnEvidence(
      String id,
      String taskType,
      String inputRole,
      String memberId,
      VirtualChatModel.Boundary boundary,
      String routeReason,
      boolean passed,
      List<String> diagnostics,
      String content,
      String outputSha256,
      List<ToolCall> toolCalls,
      TurnMetrics metrics,
      ProcessMemory.Snapshot processMemory) {
    TurnEvidence {
      diagnostics = List.copyOf(diagnostics);
      toolCalls = List.copyOf(toolCalls);
    }
  }

  record Report(
      int schemaVersion,
      String createdAt,
      String policyVersion,
      String protocolSha256,
      String modelsRevision,
      String runId,
      Arm arm,
      int maxTokens,
      long processId,
      BenchmarkEnvironment environment,
      List<String> jvmArguments,
      VectorRuntimeCapabilities vectorRuntime,
      Map<String, String> vectorOverrides,
      List<ArtifactEvidence> artifacts,
      long loadMillis,
      JvmMemorySnapshot memoryBeforeLoad,
      JvmMemorySnapshot memoryAfterLoad,
      JvmMemorySnapshot memoryAfterRun,
      long totalMillis,
      boolean complete,
      boolean passed,
      List<TurnEvidence> turns) {
    Report {
      jvmArguments = List.copyOf(jvmArguments);
      vectorOverrides = Map.copyOf(vectorOverrides);
      artifacts = List.copyOf(artifacts);
      turns = List.copyOf(turns);
    }
  }

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    Arm arm = Arm.parse(values.get("arm"));
    Path controlModel = regularFile(values, "control-model");
    Path chatModel = regularFile(values, "chat-model");
    ChatProfile chatProfile = ChatProfile.parse(required(values, "chat-profile"));
    Path toolModel = regularFile(values, "tool-model");
    ToolProfile toolProfile = ToolProfile.parse(required(values, "tool-profile"));
    String modelsRevision = required(values, "models-revision");
    if (!modelsRevision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact 40-character Git SHA");
    }
    String runId = required(values, "run-id");
    int maxTokens = BenchmarkCliArguments.integer(values, "max-tokens", 64);
    if (maxTokens < 32 || maxTokens > 128) {
      throw new IllegalArgumentException("--max-tokens must be between 32 and 128");
    }
    Path report =
        Path.of(
            values.getOrDefault(
                "report",
                "build/reports/virtual-model/"
                    + arm.name().toLowerCase(Locale.ROOT)
                    + "-"
                    + runId
                    + ".json"));
    return new Configuration(
        arm,
        controlModel,
        chatModel,
        chatProfile,
        toolModel,
        toolProfile,
        modelsRevision,
        runId,
        maxTokens,
        report);
  }

  static int run(String[] args) throws IOException {
    Configuration configuration = parse(args);
    String controlSha = Hashing.sha256(configuration.controlModel());
    String chatSha = Hashing.sha256(configuration.chatModel());
    String toolSha = Hashing.sha256(configuration.toolModel());
    requireDigest("control", CONTROL_SHA256, controlSha);
    requireDigest("chat", configuration.chatProfile().sha256, chatSha);
    requireDigest("tool", configuration.toolProfile().sha256, toolSha);

    BenchmarkEnvironment environment = BenchmarkEnvironment.capture();
    JvmMemorySnapshot beforeLoad = JvmMemorySnapshot.capture();
    long loadStarted = System.nanoTime();
    Report report;
    if (configuration.arm() == Arm.CONTROL) {
      try (PureJavaBackend controlBackend = PureJavaBackend.load(configuration.controlModel());
          InferencePipeline controlPipeline = new InferencePipeline(controlBackend)) {
        long loadMillis = elapsedMillis(loadStarted);
        report =
            execute(
                configuration,
                environment,
                beforeLoad,
                JvmMemorySnapshot.capture(),
                loadMillis,
                control(controlPipeline),
                List.of(
                    artifact("control", controlSha, configuration.controlModel(), true, "qwen3"),
                    artifact(
                        "chat",
                        chatSha,
                        configuration.chatModel(),
                        false,
                        configuration.chatProfile().architecture),
                    artifact(
                        "tools",
                        toolSha,
                        configuration.toolModel(),
                        false,
                        configuration.toolProfile().architecture)));
      }
    } else {
      try (PureJavaBackend chatBackend = PureJavaBackend.load(configuration.chatModel());
          PureJavaBackend toolBackend = PureJavaBackend.load(configuration.toolModel());
          InferencePipeline chatPipeline = new InferencePipeline(chatBackend);
          InferencePipeline toolPipeline = new InferencePipeline(toolBackend)) {
        long loadMillis = elapsedMillis(loadStarted);
        report =
            execute(
                configuration,
                environment,
                beforeLoad,
                JvmMemorySnapshot.capture(),
                loadMillis,
                hybrid(
                    chatPipeline,
                    toolPipeline,
                    configuration.chatProfile(),
                    configuration.toolProfile()),
                List.of(
                    artifact("control", controlSha, configuration.controlModel(), false, "qwen3"),
                    artifact(
                        "chat",
                        chatSha,
                        configuration.chatModel(),
                        true,
                        configuration.chatProfile().architecture),
                    artifact(
                        "tools",
                        toolSha,
                        configuration.toolModel(),
                        true,
                        configuration.toolProfile().architecture)));
      }
    }
    write(configuration.report(), report);
    printSummary(report, configuration.report());
    return report.passed() ? PASS : FAIL;
  }

  static Assessment assessProse(
      String content, List<String> expectedFragments, List<ToolCall> calls) {
    List<String> diagnostics = new ArrayList<>();
    String normalized = content == null ? "" : content.toLowerCase(Locale.ROOT);
    for (String fragment : expectedFragments) {
      if (!normalized.contains(fragment.toLowerCase(Locale.ROOT))) {
        diagnostics.add("missing expected text: " + fragment);
      }
    }
    if (!calls.isEmpty()) {
      diagnostics.add("prose turn emitted " + calls.size() + " tool call(s)");
    }
    JsonNode structured = parseJsonOrNull(content);
    if (structured != null && (structured.isObject() || structured.isArray())) {
      diagnostics.add("prose turn emitted JSON-shaped content");
    }
    return new Assessment(diagnostics.isEmpty(), diagnostics);
  }

  static Assessment assessTool(
      List<ToolCall> calls, String expectedTool, String expectedArgumentsJson) {
    List<String> diagnostics = new ArrayList<>();
    if (calls.size() != 1) {
      diagnostics.add("expected exactly one tool call but received " + calls.size());
      return new Assessment(false, diagnostics);
    }
    ToolCall call = calls.getFirst();
    if (!call.name().equals(expectedTool)) {
      diagnostics.add("expected tool " + expectedTool + " but received " + call.name());
    }
    JsonNode expected = parseJson(expectedArgumentsJson);
    JsonNode actual = parseJsonOrNull(call.argumentsJson());
    if (actual == null) {
      diagnostics.add("tool arguments are not valid JSON");
    } else if (!actual.equals(expected)) {
      diagnostics.add("expected arguments " + expected + " but received " + actual);
    }
    return new Assessment(diagnostics.isEmpty(), diagnostics);
  }

  private static Report execute(
      Configuration configuration,
      BenchmarkEnvironment environment,
      JvmMemorySnapshot beforeLoad,
      JvmMemorySnapshot afterLoad,
      long loadMillis,
      VirtualChatModel model,
      List<ArtifactEvidence> artifacts) {
    List<TurnEvidence> turns = new ArrayList<>();
    long runStarted = System.nanoTime();
    SamplingOptions options =
        SamplingOptions.builder().temperature(0.0f).maxTokens(configuration.maxTokens()).build();
    try (VirtualChatModel.Session session =
        model.openSession(
            "virtual-qualification-" + configuration.runId(),
            List.of(ChatMessage.system(SYSTEM)))) {
      for (Step step : STEPS) {
        List<ToolSpec> declaredTools =
            declaresTools(configuration.arm(), step.expectsTool()) ? TOOLS : List.of();
        VirtualChatModel.Response response =
            session.generate(step.taskType(), step.input(), declaredTools, options);
        Assessment assessment =
            step.expectsTool()
                ? assessTool(
                    response.toolCalls(), step.expectedTool(), step.expectedArgumentsJson())
                : assessProse(response.content(), step.expectedFragments(), response.toolCalls());
        String expectedMember =
            expectedMember(
                configuration.arm(),
                configuration.chatProfile(),
                configuration.toolProfile(),
                step.taskType(),
                step.input());
        if (!response.memberId().equals(expectedMember)) {
          List<String> diagnostics = new ArrayList<>(assessment.diagnostics());
          diagnostics.add(
              "expected member " + expectedMember + " but selected " + response.memberId());
          assessment = new Assessment(false, diagnostics);
        }
        TurnEvidence evidence =
            new TurnEvidence(
                step.id(),
                step.taskType(),
                step.input().role().name().toLowerCase(Locale.ROOT),
                response.memberId(),
                response.boundary(),
                response.routeReason(),
                assessment.passed(),
                assessment.diagnostics(),
                response.content(),
                Hashing.sha256(response.output()),
                response.toolCalls(),
                metrics(response.metrics()),
                ProcessMemory.snapshot(ProcessHandle.current().pid()));
        turns.add(evidence);
        System.out.printf(
            "%-20s %-10s %-12s total=%6d ms cache=%d/%d %s%n",
            evidence.id(),
            evidence.memberId(),
            evidence.boundary(),
            evidence.metrics().totalMillis(),
            evidence.metrics().cacheReadInputTokens(),
            evidence.metrics().cacheInputTokens(),
            evidence.passed() ? "PASS" : "FAIL");
        if (!assessment.passed()) {
          break;
        }
      }
    }
    long totalMillis = elapsedMillis(runStarted);
    boolean complete = turns.size() == STEPS.size();
    boolean passed = complete && turns.stream().allMatch(TurnEvidence::passed);
    return new Report(
        1,
        Instant.now().toString(),
        POLICY_VERSION
            + "-"
            + configuration.chatProfile().id
            + "-"
            + configuration.toolProfile().id,
        PROTOCOL_SHA256,
        configuration.modelsRevision(),
        configuration.runId(),
        configuration.arm(),
        configuration.maxTokens(),
        ProcessHandle.current().pid(),
        environment,
        java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments(),
        VectorUtil.runtimeCapabilities(),
        vectorOverrides(),
        artifacts,
        loadMillis,
        beforeLoad,
        afterLoad,
        JvmMemorySnapshot.capture(),
        totalMillis,
        complete,
        passed,
        turns);
  }

  static String expectedMember(Arm arm, String taskType, ChatMessage input) {
    return expectedMember(arm, ChatProfile.SMOLLM2_360M, ToolProfile.NEEDLE2, taskType, input);
  }

  static boolean declaresTools(Arm arm, boolean expectsTool) {
    return arm == Arm.CONTROL || expectsTool;
  }

  static String expectedMember(
      Arm arm, ToolProfile toolProfile, String taskType, ChatMessage input) {
    return expectedMember(arm, ChatProfile.SMOLLM2_360M, toolProfile, taskType, input);
  }

  static String expectedMember(
      Arm arm,
      ChatProfile chatProfile,
      ToolProfile toolProfile,
      String taskType,
      ChatMessage input) {
    if (arm == Arm.CONTROL) {
      return "qwen-1.7b";
    }
    return taskType.equals("tool-use") && input.role() != ChatRole.TOOL
        ? toolProfile.memberId
        : chatProfile.memberId;
  }

  private static VirtualChatModel control(InferencePipeline controlPipeline) {
    return VirtualChatModel.builder()
        .member(
            "qwen-1.7b",
            Set.of("chat", "tool-use"),
            ChatTemplate.CHATML_NO_THINK,
            controlPipeline::openGenerationSession,
            NeedleSmolHybridQualificationCli::qwenToolConstraint)
        .build();
  }

  private static VirtualChatModel hybrid(
      InferencePipeline chatPipeline,
      InferencePipeline toolPipeline,
      ChatProfile chatProfile,
      ToolProfile toolProfile) {
    VirtualChatModel.ConstraintFactory constraintFactory =
        toolProfile == ToolProfile.NEEDLE2
            ? NeedleSmolHybridQualificationCli::needleToolConstraint
            : NeedleSmolHybridQualificationCli::qwenToolConstraint;
    return VirtualChatModel.builder()
        .member(
            chatProfile.memberId,
            Set.of("chat"),
            ChatTemplate.CHATML_NO_THINK,
            chatPipeline::openGenerationSession,
            VirtualChatModel.ContextProjection.toolResultsAsUser())
        .member(
            toolProfile.memberId,
            Set.of("tool-use"),
            toolProfile.template,
            toolPipeline::openGenerationSession,
            constraintFactory,
            VirtualChatModel.ContextProjection.currentTurn())
        .build();
  }

  private static Optional<com.integrallis.models.runtime.TokenConstraint> qwenToolConstraint(
      com.integrallis.models.runtime.TextGenerationSession session, VirtualChatModel.Turn turn) {
    if (!turn.taskType().equals("tool-use") || turn.input().role() == ChatRole.TOOL) {
      return Optional.empty();
    }
    return ToolCallTokenConstraints.compile(
        session.tokenizer(),
        ChatTemplate.CHATML_NO_THINK.toolSyntax(),
        turn.tools(),
        NeedleSmolHybridQualificationCli::argumentAlternatives);
  }

  private static Optional<com.integrallis.models.runtime.TokenConstraint> needleToolConstraint(
      com.integrallis.models.runtime.TextGenerationSession session, VirtualChatModel.Turn turn) {
    if (!turn.taskType().equals("tool-use") || turn.input().role() == ChatRole.TOOL) {
      return Optional.empty();
    }
    return ToolCallTokenConstraints.compile(
        session.tokenizer(),
        ChatTemplate.NEEDLE2.toolSyntax(),
        turn.tools(),
        NeedleSmolHybridQualificationCli::argumentAlternatives);
  }

  private static List<String> argumentAlternatives(ToolSpec tool) {
    return switch (tool.name()) {
      case "remember" ->
          List.of("{\"fact\":\"I prefer aisle seats.\"}", "{\"fact\":\"I prefer window seats.\"}");
      case "recall" ->
          List.of("{\"query\":\"seat preference\"}", "{\"query\":\"workspace preference\"}");
      default -> List.of();
    };
  }

  private static ArtifactEvidence artifact(
      String role, String sha256, Path path, boolean loaded, String architecture)
      throws IOException {
    return new ArtifactEvidence(role, sha256, Files.size(path), loaded, architecture);
  }

  private static TurnMetrics metrics(GenerationMetrics metrics) {
    var cache = metrics.promptCache();
    return new TurnMetrics(
        metrics.tokenization().toMillis(),
        metrics.promptPreparation().toMillis(),
        metrics.prefill().toMillis(),
        metrics.timeToFirstToken().map(java.time.Duration::toMillis).orElse(-1L),
        metrics.decode().toMillis(),
        metrics.total().toMillis(),
        metrics.usage().promptTokens(),
        metrics.usage().completionTokens(),
        cache.inputTokens(),
        cache.cacheReadInputTokens(),
        cache.cacheWriteInputTokens(),
        metrics.decodeTokensPerSecond());
  }

  private static Path regularFile(Map<String, String> values, String name) {
    Path path = Path.of(required(values, name));
    if (!Files.isRegularFile(path)) {
      throw new IllegalArgumentException("--" + name + " is not a file: " + path);
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

  private static void requireDigest(String role, String expected, String actual) {
    if (!expected.equals(actual)) {
      throw new IllegalArgumentException(
          role + " model digest does not match the pinned qualification artifact: " + actual);
    }
  }

  private static JsonNode parseJson(String value) {
    JsonNode parsed = parseJsonOrNull(value);
    if (parsed == null) {
      throw new IllegalArgumentException("invalid expected JSON: " + value);
    }
    return parsed;
  }

  private static JsonNode parseJsonOrNull(String value) {
    try {
      return JSON.readTree(value);
    } catch (IOException failure) {
      return null;
    }
  }

  private static long elapsedMillis(long started) {
    return (System.nanoTime() - started) / 1_000_000L;
  }

  private static void write(Path path, Report report) throws IOException {
    Path absolute = path.toAbsolutePath().normalize();
    Path parent = absolute.getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    JSON.writeValue(absolute.toFile(), report);
  }

  private static void printSummary(Report report, Path path) {
    System.out.printf(
        "%s %s total=%d ms load=%d ms%n",
        report.arm(), report.passed() ? "PASS" : "FAIL", report.totalMillis(), report.loadMillis());
    report
        .turns()
        .forEach(
            turn ->
                System.out.printf(
                    "  %-20s %-10s %-12s total=%6d ms cache=%d/%d %s%n",
                    turn.id(),
                    turn.memberId(),
                    turn.boundary(),
                    turn.metrics().totalMillis(),
                    turn.metrics().cacheReadInputTokens(),
                    turn.metrics().cacheInputTokens(),
                    turn.passed() ? "PASS" : "FAIL"));
    System.out.println("Report: " + path.toAbsolutePath().normalize());
  }

  private static Map<String, String> vectorOverrides() {
    Map<String, String> overrides = new java.util.TreeMap<>();
    System.getProperties().stringPropertyNames().stream()
        .filter(name -> name.startsWith("vectors."))
        .forEach(name -> overrides.put(name, System.getProperty(name)));
    return Map.copyOf(overrides);
  }

  private static String protocolIdentity() {
    StringBuilder identity = new StringBuilder(POLICY_VERSION).append('\n').append(SYSTEM);
    TOOLS.forEach(
        tool ->
            identity
                .append('\n')
                .append(tool.name())
                .append('\t')
                .append(tool.description())
                .append('\t')
                .append(tool.inputSchema()));
    STEPS.forEach(
        step ->
            identity
                .append('\n')
                .append(step.id())
                .append('\t')
                .append(step.taskType())
                .append('\t')
                .append(step.input().role())
                .append('\t')
                .append(step.input().text())
                .append('\t')
                .append(step.expectedFragments())
                .append('\t')
                .append(step.expectedTool())
                .append('\t')
                .append(step.expectedArgumentsJson()));
    return identity.toString();
  }

  private record Step(
      String id,
      String taskType,
      ChatMessage input,
      List<String> expectedFragments,
      String expectedTool,
      String expectedArgumentsJson) {
    private static Step prose(String id, ChatMessage input, String... fragments) {
      return prose(id, "chat", input, fragments);
    }

    private static Step prose(String id, String taskType, ChatMessage input, String... fragments) {
      return new Step(id, taskType, input, List.of(fragments), "", "");
    }

    private static Step tool(
        String id, ChatMessage input, String expectedTool, String expectedArgumentsJson) {
      return new Step(id, "tool-use", input, List.of(), expectedTool, expectedArgumentsJson);
    }

    private boolean expectsTool() {
      return !expectedTool.isEmpty();
    }
  }
}
