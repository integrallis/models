///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS com.integrallis:backend-java:0.3.42
//DEPS com.integrallis:models-runtime:0.3.42
//JAVA_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED

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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.TokenConstraint;
import com.integrallis.models.runtime.chat.GraniteDocumentsPrompt;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Clean-host Java 25 run of the Integrallis Granite 4.1 3B answerability activated LoRA, using only
 * the Maven Central Models 0.3.42 artifacts (backend-java, models-runtime and their transitive
 * dependencies).
 *
 * <p>Downloads the pinned base GGUF, the pinned adapter bundle (ModelJars GitHub release) and the
 * frozen qualification window, each hash-checked and failing closed. Runs the first three cases of
 * squad-v2-dev and msmarco-v2.1-validation through the pure-Java backend with a physically shared
 * prefix, rendering the prompt exactly as {@code ActivatedAnswerabilityQualificationCli} (models-bench,
 * v0.3.42) renders the specialist arm, and compares every output byte-for-byte, and its shared-prefix
 * token count, with the committed qualification evidence.
 *
 * <p>Usage: {@code jbang AnswerabilityCleanHost.java [--store <dir>]} (default store: {@code ./store}).
 * Exit 0 only if all six cases are structured, physically shared, byte-identical to the
 * qualification output and report the same shared-prefix token count; 1 on any mismatch; 2 when a
 * download or hash check fails. The only network access is the pinned downloads below.
 */
public final class AnswerabilityCleanHost {
  static final String MODELS_VERSION = "0.3.42";

  static final String BASE_URL =
      "https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF/resolve/"
          + "ab4701481089b58a082ef63cc1cee738887293ff/granite-4.1-3b-Q4_K_M.gguf";
  static final String BASE_SHA256 =
      "662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29";

  static final String ADAPTER_RELEASE =
      "https://github.com/ModelJars/activated-adapters/releases/download/"
          + "granite-4.1-3b-answerability-alora-integrallis-v1/";
  static final Map<String, String> ADAPTER_FILES = new LinkedHashMap<>();

  static {
    ADAPTER_FILES.put(
        "adapter_model.safetensors",
        "67533dff14cd0cfa9ae0d00e83a9955226426ff1f98f1fd28790b2fd8757eea6");
    ADAPTER_FILES.put("LICENSE", "f3b8149a65f7ae0e2fe55de40d52b98dd8965e6b213b7c441a9596ba670c2e64");
    ADAPTER_FILES.put(
        "models-activated-lora.json",
        "27de22cf55ca941832bb3d5d6e0ce46de570edfc85f00ea33a11c30b2d33ccd3");
    ADAPTER_FILES.put("NOTICE", "4d3b1f70e77411b1d51ba33215dfe2bd7d1f4090d384725b7f1820df7a8937b1");
  }

  // integrallis/models commit cb2f42624d6cb75e8caa9fbe1715233ea7148eb2 froze window v2.
  static final String WINDOW_URL =
      "https://raw.githubusercontent.com/integrallis/models/cb2f42624d6cb75e8caa9fbe1715233ea7148eb2/"
          + "benchmark-results/2026-09-15-granite-4.1-3b-alora-hybrid/qualification-window-v2.json";
  static final String WINDOW_SHA256 =
      "dfb8cd7203418c79bae27306ffc4857f0ab02be7f9f8ddcae793af556e9cab37";

  /** Same as the qualification runner: greedy, six completion tokens, unconstrained. */
  static final int MAX_COMPLETION_TOKENS = 6;

  static final int CASES_PER_SUITE = 3;

  record Expected(String suite, String id, String output, int sharedPrefixTokens) {}

  /**
   * The first three cases of each suite from the committed pilot-2 qualification evidence
   * (release-pilot2/evidence/pj1/window-squad-v2-dev-specialist-pure-java.json and
   * release-pilot2/evidence/pj3/window-msmarco-v2.1-validation-specialist-pure-java.json).
   */
  static final List<Expected> EXPECTED =
      List.of(
          new Expected("squad-v2-dev", "5ad247b0d7d075001a428b45", "\"unanswerable\"", 269),
          new Expected("squad-v2-dev", "57267640f1498d1400e8e074", "\"answerable\"", 386),
          new Expected("squad-v2-dev", "5737432bc3c5551400e51e9b", "\"answerable\"", 357),
          new Expected("msmarco-v2.1-validation", "95542", "\"unanswerable\"", 776),
          new Expected("msmarco-v2.1-validation", "851555", "\"answerable\"", 1082),
          new Expected("msmarco-v2.1-validation", "1067349", "\"answerable\"", 853));

  static final class DownloadFailure extends RuntimeException {
    DownloadFailure(String message) {
      super(message);
    }
  }

  public static void main(String[] args) throws Exception {
    Path store = Path.of("store");
    for (int i = 0; i < args.length; i++) {
      if ("--store".equals(args[i]) && i + 1 < args.length) {
        store = Path.of(args[++i]);
      } else {
        System.err.println("usage: AnswerabilityCleanHost [--store <dir>]");
        System.exit(64);
      }
    }
    int exit;
    try {
      exit = run(store.toAbsolutePath().normalize());
    } catch (DownloadFailure failure) {
      System.out.println("FAIL download: " + failure.getMessage());
      exit = 2;
    }
    System.out.flush();
    System.exit(exit);
  }

  static int run(Path store) throws Exception {
    System.out.printf(
        "clean-host modelsVersion=%s java=%s vendor=%s os=%s/%s processors=%d%n",
        MODELS_VERSION,
        System.getProperty("java.version"),
        System.getProperty("java.vendor"),
        System.getProperty("os.name"),
        System.getProperty("os.arch"),
        Runtime.getRuntime().availableProcessors());
    Path modelDir = Files.createDirectories(store.resolve("models"));
    Path adapterDir = Files.createDirectories(store.resolve("adapter"));
    HttpClient http =
        HttpClient.newBuilder()
            .followRedirects(HttpClient.Redirect.NORMAL)
            .connectTimeout(Duration.ofSeconds(30))
            .build();

    Path base = fetch(http, BASE_URL, modelDir.resolve("granite-4.1-3b-Q4_K_M.gguf"), BASE_SHA256);
    for (Map.Entry<String, String> file : ADAPTER_FILES.entrySet()) {
      fetch(http, ADAPTER_RELEASE + file.getKey(), adapterDir.resolve(file.getKey()), file.getValue());
    }
    try (var listing = Files.list(adapterDir)) {
      List<String> extra =
          listing
              .map(p -> p.getFileName().toString())
              .filter(name -> !ADAPTER_FILES.containsKey(name))
              .sorted()
              .toList();
      if (!extra.isEmpty()) {
        throw new DownloadFailure("adapter directory holds files outside the release: " + extra);
      }
    }
    Path windowFile = fetch(http, WINDOW_URL, store.resolve("qualification-window-v2.json"), WINDOW_SHA256);

    Object window = new Json(Files.readString(windowFile, StandardCharsets.UTF_8)).parse();
    List<Case> cases = new ArrayList<>();
    for (String suite : List.of("squad-v2-dev", "msmarco-v2.1-validation")) {
      cases.addAll(firstCases(window, suite, CASES_PER_SUITE));
    }
    if (cases.size() != EXPECTED.size()) {
      throw new IllegalStateException("expected " + EXPECTED.size() + " cases, got " + cases.size());
    }

    SamplingOptions options =
        SamplingOptions.builder().temperature(0).maxTokens(MAX_COMPLETION_TOKENS).build();
    int passed = 0;
    long loadStarted = System.nanoTime();
    try (PureJavaBackend backend = PureJavaBackend.loadActivatedAdapter(base, adapterDir);
        ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1)) {
      System.out.printf(
          "loaded backend=pure-java adapterSha256=%s invocation=%s millis=%d%n",
          model.adapter().adapterSha256(),
          jsonString(model.adapter().invocationText()),
          (System.nanoTime() - loadStarted) / 1_000_000L);
      if (!GraniteDocumentsPrompt.ASSISTANT_MARKER.equals(model.adapter().invocationText())) {
        System.out.println("FAIL adapter marker differs from the Granite assistant marker");
        return 1;
      }
      for (int index = 0; index < cases.size(); index++) {
        Case item = cases.get(index);
        Expected expected = EXPECTED.get(index);
        if (!expected.suite().equals(item.suite()) || !expected.id().equals(item.id())) {
          System.out.printf(
              "FAIL window case %d is %s/%s, expected %s/%s%n",
              index, item.suite(), item.id(), expected.suite(), expected.id());
          return 1;
        }
        ModelPrompt prompt = prompt(item);
        StringBuilder text = new StringBuilder();
        for (ModelPrompt.Segment segment : prompt.segments()) {
          text.append(segment.text());
        }
        long started = System.nanoTime();
        String output;
        boolean shared;
        int sharedTokens;
        try (ActivatedToolTurn turn =
            model.openToolTurn(prompt, ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
          output = turn.generateToolCall(options, TokenConstraint.unrestricted());
          shared = turn.physicallySharesPrefix();
          sharedTokens = turn.sharedPrefixTokens();
        }
        long millis = (System.nanoTime() - started) / 1_000_000L;
        boolean structured = structured(output);
        boolean identical = expected.output().equals(output);
        boolean sameShared = expected.sharedPrefixTokens() == sharedTokens;
        boolean ok = structured && shared && identical && sameShared;
        if (ok) passed++;
        System.out.printf(
            "CASE %s suite=%s id=%s promptSha256=%s output=%s structured=%s physicallySharesPrefix=%s"
                + " sharedPrefixTokens=%d expectedOutput=%s expectedSharedPrefixTokens=%d"
                + " byteIdentical=%s millis=%d%n",
            ok ? "PASS" : "FAIL",
            item.suite(),
            item.id(),
            sha256(text.toString().getBytes(StandardCharsets.UTF_8)),
            jsonString(output),
            structured,
            shared,
            sharedTokens,
            jsonString(expected.output()),
            expected.sharedPrefixTokens(),
            identical,
            millis);
      }
    }
    boolean pass = passed == EXPECTED.size();
    System.out.printf("%s cases=%d passed=%d%n", pass ? "PASS" : "FAIL", EXPECTED.size(), passed);
    return pass ? 0 : 1;
  }

  /** The adapter's io.yaml contract, as in the qualification runner: a JSON string of one label. */
  static boolean structured(String output) {
    String trimmed = output == null ? "" : output.strip();
    return "\"answerable\"".equals(trimmed) || "\"unanswerable\"".equals(trimmed);
  }

  record Message(String role, String text) {}

  record Case(String suite, String id, List<Message> messages, List<String> documents) {}

  /** Specialist-arm rendering, identical to ActivatedAnswerabilityQualificationCli.prompt. */
  static ModelPrompt prompt(Case item) {
    ModelPrompt.Builder prompt =
        GraniteDocumentsPrompt.appendSystem(ModelPrompt.builder(), item.documents(), null);
    for (Message message : item.messages()) {
      GraniteDocumentsPrompt.appendTurn(prompt, message.role(), message.text());
    }
    return GraniteDocumentsPrompt.finish(prompt);
  }

  @SuppressWarnings("unchecked")
  static List<Case> firstCases(Object window, String suiteName, int limit) {
    Map<String, Object> root = (Map<String, Object>) window;
    if (!Long.valueOf(1).equals(root.get("schemaVersion"))) {
      throw new IllegalArgumentException("invalid qualification window");
    }
    for (Object suiteNode : (List<Object>) root.get("suites")) {
      Map<String, Object> suite = (Map<String, Object>) suiteNode;
      if (!suiteName.equals(suite.get("name"))) continue;
      List<Case> cases = new ArrayList<>();
      for (Object caseNode : (List<Object>) suite.get("cases")) {
        if (cases.size() == limit) break;
        Map<String, Object> node = (Map<String, Object>) caseNode;
        String id = (String) node.get("id");
        List<Message> messages = new ArrayList<>();
        for (Object m : (List<Object>) node.get("messages")) {
          Map<String, Object> message = (Map<String, Object>) m;
          String role = (String) message.get("role");
          String text = (String) message.get("text");
          if (!("user".equals(role) || "assistant".equals(role)) || text == null || text.isBlank()) {
            throw new IllegalArgumentException("invalid message in case: " + id);
          }
          messages.add(new Message(role, text));
        }
        if (messages.isEmpty() || !"user".equals(messages.getLast().role())) {
          throw new IllegalArgumentException("conversation must end with a user turn: " + id);
        }
        List<String> documents = new ArrayList<>();
        for (Object d : (List<Object>) node.get("documents")) {
          Map<String, Object> document = (Map<String, Object>) d;
          Object docId = document.get("doc_id");
          String text = (String) document.get("text");
          if (!(docId instanceof Long l && l > 0) || text == null || text.isBlank()) {
            throw new IllegalArgumentException("invalid document in case: " + id);
          }
          documents.add(text);
        }
        if (documents.isEmpty()) {
          throw new IllegalArgumentException("case carries no documents: " + id);
        }
        cases.add(new Case(suiteName, id, List.copyOf(messages), List.copyOf(documents)));
      }
      return cases;
    }
    throw new IllegalArgumentException("window has no suite named " + suiteName);
  }

  /** Downloads to a temporary file while hashing; the target exists only if the hash matched. */
  static Path fetch(HttpClient http, String url, Path target, String sha256) throws Exception {
    if (Files.isRegularFile(target)) {
      String present = sha256(target);
      if (present.equals(sha256)) {
        System.out.printf("present %s sha256=%s bytes=%d%n", target.getFileName(), sha256, Files.size(target));
        return target;
      }
      throw new DownloadFailure(
          "existing " + target + " has sha256 " + present + ", expected " + sha256 + "; remove it");
    }
    Path part = target.resolveSibling(target.getFileName() + ".part");
    String failure = null;
    for (int attempt = 1; attempt <= 4; attempt++) {
      Files.deleteIfExists(part);
      try {
        HttpResponse<InputStream> response =
            http.send(HttpRequest.newBuilder(URI.create(url)).GET().build(),
                HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
          response.body().close();
          failure = "HTTP " + response.statusCode() + " for " + url;
        } else {
          MessageDigest digest = MessageDigest.getInstance("SHA-256");
          try (InputStream in = response.body(); OutputStream out = Files.newOutputStream(part)) {
            byte[] buffer = new byte[1 << 20];
            for (int n; (n = in.read(buffer)) >= 0; ) {
              digest.update(buffer, 0, n);
              out.write(buffer, 0, n);
            }
          }
          String actual = HexFormat.of().formatHex(digest.digest());
          if (!actual.equals(sha256)) {
            Files.deleteIfExists(part);
            // A wrong hash is not transient: fail closed at once.
            throw new DownloadFailure(
                "HASH MISMATCH " + url + " expected " + sha256 + " got " + actual);
          }
          Files.move(part, target, StandardCopyOption.ATOMIC_MOVE);
          System.out.printf("fetched %s sha256=%s bytes=%d%n", target.getFileName(), sha256, Files.size(target));
          return target;
        }
      } catch (IOException io) {
        failure = io.getClass().getSimpleName() + " " + io.getMessage() + " for " + url;
      }
      System.out.printf("fetch attempt %d failed: %s%n", attempt, failure);
      Thread.sleep(attempt * 15_000L);
    }
    Files.deleteIfExists(part);
    throw new DownloadFailure("giving up: " + failure);
  }

  static String sha256(Path path) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (InputStream in = Files.newInputStream(path)) {
      byte[] buffer = new byte[1 << 20];
      for (int n; (n = in.read(buffer)) >= 0; ) digest.update(buffer, 0, n);
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  static String sha256(byte[] bytes) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
  }

  /** ASCII-safe JSON string literal for log lines. */
  static String jsonString(String value) {
    if (value == null) return "null";
    StringBuilder out = new StringBuilder("\"");
    for (char c : value.toCharArray()) {
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        default -> {
          if (c < 0x20 || c > 0x7e) out.append(String.format("\\u%04x", (int) c));
          else out.append(c);
        }
      }
    }
    return out.append('"').toString();
  }

  /**
   * Minimal strict JSON reader (objects, arrays, strings, integers, booleans, null) so the script
   * needs no dependency beyond the Models artifacts. Strings decode exactly as any conforming parser
   * does, surrogate-pair escapes included.
   */
  static final class Json {
    private final String s;
    private int i;

    Json(String s) {
      this.s = s;
    }

    Object parse() {
      Object value = value();
      ws();
      if (i != s.length()) throw error("trailing content");
      return value;
    }

    private Object value() {
      ws();
      if (i >= s.length()) throw error("unexpected end");
      char c = s.charAt(i);
      return switch (c) {
        case '{' -> object();
        case '[' -> array();
        case '"' -> string();
        case 't' -> literal("true", Boolean.TRUE);
        case 'f' -> literal("false", Boolean.FALSE);
        case 'n' -> literal("null", null);
        default -> number();
      };
    }

    private Map<String, Object> object() {
      Map<String, Object> map = new LinkedHashMap<>();
      i++;
      ws();
      if (peek() == '}') {
        i++;
        return map;
      }
      while (true) {
        ws();
        if (peek() != '"') throw error("expected key");
        String key = string();
        ws();
        expect(':');
        if (map.put(key, value()) != null) throw error("duplicate key " + key);
        ws();
        char c = next();
        if (c == '}') return map;
        if (c != ',') throw error("expected , or }");
      }
    }

    private List<Object> array() {
      List<Object> list = new ArrayList<>();
      i++;
      ws();
      if (peek() == ']') {
        i++;
        return list;
      }
      while (true) {
        list.add(value());
        ws();
        char c = next();
        if (c == ']') return list;
        if (c != ',') throw error("expected , or ]");
      }
    }

    private String string() {
      expect('"');
      StringBuilder out = new StringBuilder();
      while (true) {
        char c = next();
        if (c == '"') return out.toString();
        if (c < 0x20) throw error("control character in string");
        if (c != '\\') {
          out.append(c);
          continue;
        }
        char e = next();
        switch (e) {
          case '"' -> out.append('"');
          case '\\' -> out.append('\\');
          case '/' -> out.append('/');
          case 'b' -> out.append('\b');
          case 'f' -> out.append('\f');
          case 'n' -> out.append('\n');
          case 'r' -> out.append('\r');
          case 't' -> out.append('\t');
          case 'u' -> {
            if (i + 4 > s.length()) throw error("short unicode escape");
            out.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
            i += 4;
          }
          default -> throw error("bad escape");
        }
      }
    }

    private Object number() {
      int start = i;
      if (peek() == '-') i++;
      while (i < s.length() && "0123456789.eE+-".indexOf(s.charAt(i)) >= 0) i++;
      String token = s.substring(start, i);
      if (token.isEmpty() || token.equals("-")) throw error("bad value");
      if (token.chars().allMatch(ch -> ch == '-' || Character.isDigit(ch))) {
        return Long.parseLong(token);
      }
      return Double.parseDouble(token);
    }

    private Object literal(String word, Object value) {
      if (!s.startsWith(word, i)) throw error("bad literal");
      i += word.length();
      return value;
    }

    private void ws() {
      while (i < s.length() && " \t\r\n".indexOf(s.charAt(i)) >= 0) i++;
    }

    private char peek() {
      return i < s.length() ? s.charAt(i) : '\0';
    }

    private char next() {
      if (i >= s.length()) throw error("unexpected end");
      return s.charAt(i++);
    }

    private void expect(char c) {
      if (next() != c) throw error("expected " + c);
    }

    private IllegalArgumentException error(String message) {
      return new IllegalArgumentException("window JSON: " + message + " at offset " + i);
    }
  }
}
