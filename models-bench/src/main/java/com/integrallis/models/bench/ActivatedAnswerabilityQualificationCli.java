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
import com.integrallis.models.runtime.chat.GraniteDocumentsPrompt;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Gate 4 runner for an activated answerability specialist over a frozen qualification window.
 *
 * <p>Both arms render the documents prompt exactly as the published Granite 4.1 chat template
 * renders a documents-only request: the documents system message, the conversation turns, and the
 * assistant marker. The specialist arm activates the adapter at that marker over a physically
 * shared base prefix. The base arm prepends the fixed one-word instruction to the system message
 * and generates with the unactivated base pipeline. Every case records the raw output, so the
 * aggregate can be recomputed independently; nothing in this runner interprets a result beyond the
 * exact two-label contract in the adapter's published {@code io.yaml}.
 */
final class ActivatedAnswerabilityQualificationCli {
  static final String INVOCATION = GraniteDocumentsPrompt.ASSISTANT_MARKER;
  static final String BASE_INSTRUCTION = "Answer with exactly one word: answerable or unanswerable";

  static final String DOCUMENTS_OPEN = GraniteDocumentsPrompt.DOCUMENTS_OPEN;
  static final String DOCUMENTS_CLOSE = GraniteDocumentsPrompt.DOCUMENTS_CLOSE;
  static final String DOCUMENTS_INTRO = GraniteDocumentsPrompt.INTRO;
  static final String DOCUMENTS_TAGS_SUFFIX = GraniteDocumentsPrompt.TAGS_SUFFIX;
  static final String DOCUMENTS_OUTRO = GraniteDocumentsPrompt.OUTRO;
  static final Set<String> LABELS = Set.of("answerable", "unanswerable");
  private static final int MAX_COMPLETION_TOKENS = 6;
  private static final Set<String> OPTIONS =
      Set.of(
          "model",
          "adapter",
          "models-revision",
          "window",
          "suite",
          "arm",
          "report",
          "limit",
          "dump-first-prompt",
          "dump-prompts",
          "backend");
  private static final ObjectMapper JSON = new ObjectMapper();

  private ActivatedAnswerabilityQualificationCli() {}

  enum Arm {
    SPECIALIST,
    BASE
  }

  record Configuration(
      Path model,
      Path adapter,
      String modelsRevision,
      Path window,
      String suite,
      Arm arm,
      Path report,
      int limit,
      Path dumpFirstPrompt,
      Path dumpPrompts,
      String backend) {}

  record Message(String role, String text) {}

  record Document(int docId, String text) {}

  record Case(String id, String label, List<Message> messages, List<Document> documents) {}

  record CaseResult(
      String id,
      String label,
      String prediction,
      boolean structured,
      boolean correct,
      boolean physicallyShared,
      int sharedPrefixTokens,
      long millis,
      long prefillMillis,
      long decodeMillis,
      String output) {}

  record Summary(
      int cases,
      int structured,
      double structuredRate,
      int answerableCases,
      int answerableCorrect,
      int unanswerableCases,
      int unanswerableCorrect,
      double balancedAccuracy,
      int physicallyShared,
      boolean executionPassed) {}

  record Report(
      int schemaVersion,
      String createdAt,
      String modelsRevision,
      String backend,
      String kernelPlan,
      String arm,
      String suite,
      JsonNode source,
      String windowSha256,
      String windowFileSha256,
      int limit,
      boolean complete,
      Map<String, String> environment,
      Summary summary,
      List<CaseResult> cases) {}

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    String revision = required(values, "models-revision");
    if (!revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact 40-character Git SHA");
    }
    Arm arm;
    try {
      arm = Arm.valueOf(required(values, "arm").toUpperCase(Locale.ROOT));
    } catch (IllegalArgumentException invalid) {
      throw new IllegalArgumentException("--arm must be specialist or base");
    }
    int limit = BenchmarkCliArguments.integer(values, "limit", 0);
    if (limit < 0) {
      throw new IllegalArgumentException("--limit must be >= 0");
    }
    String backend = values.getOrDefault("backend", "pure-java");
    if (!Set.of("pure-java", "rust-ffm").contains(backend)) {
      throw new IllegalArgumentException("--backend must be pure-java or rust-ffm");
    }
    return new Configuration(
        Path.of(required(values, "model")),
        Path.of(required(values, "adapter")),
        revision,
        Path.of(required(values, "window")),
        required(values, "suite"),
        arm,
        Path.of(required(values, "report")),
        limit,
        values.get("dump-first-prompt") == null ? null : Path.of(values.get("dump-first-prompt")),
        values.get("dump-prompts") == null ? null : Path.of(values.get("dump-prompts")),
        backend);
  }

  /** Renders one document exactly as Transformers' {@code tojson} filter renders the mapping. */
  static String documentJson(Document document) {
    return GraniteDocumentsPrompt.documentJson(document.docId(), document.text());
  }

  /** The document block rendered between the two {@code <documents>} markers. */
  static String documentsBlock(List<Document> documents) {
    return GraniteDocumentsPrompt.documentsBlock(documents.stream().map(Document::text).toList());
  }

  /**
   * Renders the prompt for one arm through the shared Granite documents renderer: only the role
   * delimiters, the two documents markers, and the assistant marker are control.
   */
  static ModelPrompt prompt(Case item, Arm arm) {
    ModelPrompt.Builder prompt =
        GraniteDocumentsPrompt.appendSystem(
            ModelPrompt.builder(),
            item.documents().stream().map(Document::text).toList(),
            arm == Arm.BASE ? BASE_INSTRUCTION : null);
    for (Message message : item.messages()) {
      GraniteDocumentsPrompt.appendTurn(prompt, message.role(), message.text());
    }
    return GraniteDocumentsPrompt.finish(prompt);
  }

  /**
   * Applies each arm's own output contract. The specialist follows the adapter's published {@code
   * io.yaml}: the completion is a JSON string whose value is one of the two labels. The base arm is
   * instructed to answer with exactly one word, so its contract is the bare label. Surrounding
   * whitespace is trimmed on both; anything else is unstructured.
   */
  static String prediction(String output, Arm arm) {
    String trimmed = output == null ? "" : output.strip();
    for (String label : LABELS) {
      String expected;
      if (arm == Arm.BASE) {
        expected = label;
      } else {
        try {
          expected = JSON.writeValueAsString(label);
        } catch (IOException impossible) {
          throw new IllegalStateException(impossible);
        }
      }
      if (expected.equals(trimmed)) {
        return label;
      }
    }
    return "";
  }

  /** Specialist contract, kept for callers that only see adapter completions. */
  static String prediction(String output) {
    return prediction(output, Arm.SPECIALIST);
  }

  static int run(String[] args) throws Exception {
    Configuration configuration = parse(args);
    JsonNode window = JSON.readTree(configuration.window().toFile());
    JsonNode suite = suite(window, configuration.suite());
    List<Case> cases = cases(suite);
    boolean complete = configuration.limit() == 0 || configuration.limit() >= cases.size();
    if (!complete) {
      cases = cases.subList(0, configuration.limit());
    }
    SamplingOptions options =
        SamplingOptions.builder().temperature(0).maxTokens(MAX_COMPLETION_TOKENS).build();
    List<CaseResult> results = new ArrayList<>();
    try (PureJavaBackend backend = loadBackend(configuration);
        ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1)) {
      if (!INVOCATION.equals(model.adapter().invocationText())) {
        throw new IllegalStateException(
            "adapter marker differs from the Granite assistant marker: "
                + model.adapter().invocationText());
      }
      if (configuration.dumpPrompts() != null) {
        Files.createDirectories(configuration.dumpPrompts());
        int index = 0;
        for (Case item : cases) {
          ModelPrompt prompt = prompt(item, configuration.arm());
          dumpPrompt(
              configuration.dumpPrompts().resolve(index++ + ".json"),
              item,
              prompt,
              model.tokenizer().encode(prompt));
        }
        System.out.printf(
            "DUMPED %d %s prompts for %s to %s%n",
            cases.size(),
            configuration.arm().name().toLowerCase(Locale.ROOT),
            configuration.suite(),
            configuration.dumpPrompts());
        return 0;
      }
      for (Case item : cases) {
        ModelPrompt prompt = prompt(item, configuration.arm());
        if (configuration.dumpFirstPrompt() != null && results.isEmpty()) {
          dumpPrompt(
              configuration.dumpFirstPrompt(), item, prompt, model.tokenizer().encode(prompt));
        }
        long started = System.nanoTime();
        String output;
        boolean shared = false;
        int sharedTokens = 0;
        long prefillMillis = -1;
        long decodeMillis = -1;
        if (configuration.arm() == Arm.SPECIALIST) {
          try (ActivatedToolTurn turn =
              model.openToolTurn(prompt, ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
            output = turn.generateToolCall(options, TokenConstraint.unrestricted());
            shared = turn.physicallySharesPrefix();
            sharedTokens = turn.sharedPrefixTokens();
            var metrics = turn.toolMetrics();
            if (metrics.available()) {
              prefillMillis = metrics.prefill().toMillis();
              decodeMillis = metrics.decode().toMillis();
            }
          }
        } else {
          output = model.generate(prompt, options);
        }
        long millis = (System.nanoTime() - started) / 1_000_000L;
        String predicted = prediction(output, configuration.arm());
        boolean structured = !predicted.isEmpty();
        boolean correct = structured && predicted.equals(item.label());
        results.add(
            new CaseResult(
                item.id(),
                item.label(),
                predicted,
                structured,
                correct,
                shared,
                sharedTokens,
                millis,
                prefillMillis,
                decodeMillis,
                output));
        System.out.printf(
            "%s label=%s prediction=%s structured=%s shared=%s millis=%d prefill=%d decode=%d%n",
            item.id(),
            item.label(),
            predicted.isEmpty() ? "-" : predicted,
            structured,
            shared,
            millis,
            prefillMillis,
            decodeMillis);
      }
    }
    Summary summary = summarize(results, configuration.arm());
    Report report =
        new Report(
            1,
            Instant.now().toString(),
            configuration.modelsRevision(),
            configuration.backend(),
            kernelPlan(configuration.backend()),
            configuration.arm().name().toLowerCase(Locale.ROOT),
            configuration.suite(),
            suite.get("source"),
            window.path("windowSha256").asText(),
            sha256(configuration.window()),
            configuration.limit(),
            complete,
            environment(),
            summary,
            List.copyOf(results));
    JSON.writerWithDefaultPrettyPrinter().writeValue(configuration.report().toFile(), report);
    System.out.printf(
        "%s backend=%s arm=%s suite=%s cases=%d structured=%.4f balancedAccuracy=%.4f shared=%d"
            + " complete=%s%n",
        summary.executionPassed() ? "EXECUTED" : "EXECUTION-FAILED",
        report.backend(),
        report.arm(),
        report.suite(),
        summary.cases(),
        summary.structuredRate(),
        summary.balancedAccuracy(),
        summary.physicallyShared(),
        complete);
    return summary.executionPassed() ? 0 : 1;
  }

  private static final String RUST_BACKEND_CLASS =
      "com.integrallis.models.backend.nativekernel.RustFfmBackend";

  /**
   * Loads the activated backend for the requested kernel runtime. The Rust arm is the same Java
   * transformer, adapter, and shared-prefix code with only the base matrix products native.
   */
  static PureJavaBackend loadBackend(Configuration configuration) {
    if ("pure-java".equals(configuration.backend())) {
      return PureJavaBackend.loadActivatedAdapter(configuration.model(), configuration.adapter());
    }
    try {
      Class<?> backendClass = Class.forName(RUST_BACKEND_CLASS);
      return (PureJavaBackend)
          backendClass
              .getMethod("loadActivatedAdapter", Path.class, Path.class)
              .invoke(null, configuration.model(), configuration.adapter());
    } catch (ClassNotFoundException failure) {
      throw new IllegalStateException(
          "rust-ffm qualification requires the optional backend-native runtime; "
              + "rerun with -PmodelsBenchNative=true",
          failure);
    } catch (ReflectiveOperationException failure) {
      throw new IllegalStateException("Could not load the Models-owned Rust/FFM backend", failure);
    }
  }

  static String kernelPlan(String backend) {
    if ("pure-java".equals(backend)) {
      return "pure-java";
    }
    try {
      return (String) Class.forName(RUST_BACKEND_CLASS).getField("PLAN_VERSION").get(null);
    } catch (ReflectiveOperationException failure) {
      throw new IllegalStateException("Could not read the Rust/FFM kernel plan version", failure);
    }
  }

  static Summary summarize(List<CaseResult> results, Arm arm) {
    int structured = 0;
    int answerable = 0;
    int answerableCorrect = 0;
    int unanswerable = 0;
    int unanswerableCorrect = 0;
    int shared = 0;
    for (CaseResult result : results) {
      if (result.structured()) structured++;
      if (result.physicallyShared()) shared++;
      if ("answerable".equals(result.label())) {
        answerable++;
        if (result.correct()) answerableCorrect++;
      } else {
        unanswerable++;
        if (result.correct()) unanswerableCorrect++;
      }
    }
    double answerableRecall = answerable == 0 ? 0 : (double) answerableCorrect / answerable;
    double unanswerableRecall = unanswerable == 0 ? 0 : (double) unanswerableCorrect / unanswerable;
    boolean executionPassed =
        !results.isEmpty()
            && structured == results.size()
            && (arm == Arm.BASE || shared == results.size());
    return new Summary(
        results.size(),
        structured,
        results.isEmpty() ? 0 : (double) structured / results.size(),
        answerable,
        answerableCorrect,
        unanswerable,
        unanswerableCorrect,
        (answerableRecall + unanswerableRecall) / 2,
        shared,
        executionPassed);
  }

  static JsonNode suite(JsonNode window, String name) {
    if (window.path("schemaVersion").asInt() != 1 || !window.path("suites").isArray()) {
      throw new IllegalArgumentException("invalid qualification window");
    }
    for (JsonNode suite : window.get("suites")) {
      if (name.equals(suite.path("name").asText())) {
        return suite;
      }
    }
    throw new IllegalArgumentException("window has no suite named " + name);
  }

  static List<Case> cases(JsonNode suite) {
    List<Case> cases = new ArrayList<>();
    for (JsonNode node : suite.path("cases")) {
      String id = node.path("id").asText("");
      String label = node.path("label").asText("");
      if (id.isBlank() || !LABELS.contains(label)) {
        throw new IllegalArgumentException("invalid case: " + id);
      }
      List<Message> messages = new ArrayList<>();
      for (JsonNode message : node.path("messages")) {
        String role = message.path("role").asText("");
        String text = message.path("text").asText("");
        if (!("user".equals(role) || "assistant".equals(role)) || text.isBlank()) {
          throw new IllegalArgumentException("invalid message in case: " + id);
        }
        messages.add(new Message(role, text));
      }
      if (messages.isEmpty() || !"user".equals(messages.getLast().role())) {
        throw new IllegalArgumentException("conversation must end with a user turn: " + id);
      }
      List<Document> documents = new ArrayList<>();
      for (JsonNode document : node.path("documents")) {
        int docId = document.path("doc_id").asInt(0);
        String text = document.path("text").asText("");
        if (docId <= 0 || text.isBlank()) {
          throw new IllegalArgumentException("invalid document in case: " + id);
        }
        documents.add(new Document(docId, text));
      }
      if (documents.isEmpty()) {
        throw new IllegalArgumentException("case carries no documents: " + id);
      }
      cases.add(new Case(id, label, List.copyOf(messages), List.copyOf(documents)));
    }
    if (cases.isEmpty()) {
      throw new IllegalArgumentException("suite has no cases");
    }
    return cases;
  }

  /** Records the exact rendered bytes and token IDs so an external template oracle can compare. */
  static void dumpPrompt(Path target, Case item, ModelPrompt prompt, int[] tokens)
      throws IOException {
    StringBuilder text = new StringBuilder();
    for (ModelPrompt.Segment segment : prompt.segments()) {
      text.append(segment.text());
    }
    Map<String, Object> dump = new java.util.LinkedHashMap<>();
    dump.put("caseId", item.id());
    dump.put("text", text.toString());
    dump.put(
        "textSha256", sha256(text.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8)));
    dump.put("tokens", tokens);
    dump.put("tokenCount", tokens.length);
    JSON.writerWithDefaultPrettyPrinter().writeValue(target.toFile(), dump);
  }

  private static String sha256(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (NoSuchAlgorithmException failure) {
      throw new IllegalStateException(failure);
    }
  }

  private static Map<String, String> environment() {
    return Map.of(
        "osName", System.getProperty("os.name", ""),
        "osArch", System.getProperty("os.arch", ""),
        "javaVersion", System.getProperty("java.version", ""),
        "javaVendor", System.getProperty("java.vendor", ""),
        "processors", Integer.toString(Runtime.getRuntime().availableProcessors()));
  }

  private static String sha256(Path path) throws IOException {
    return sha256(Files.readAllBytes(path));
  }

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }
}
