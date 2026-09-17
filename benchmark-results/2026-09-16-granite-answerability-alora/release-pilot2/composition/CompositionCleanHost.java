///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS org.modeljars.composite:granite-answerability:0.1.47
//DEPS org.modeljars:modeljars:0.1.47
//DEPS org.modeljars.huggingface:ibm-granite.granite-4.1-3b-gguf.q4_k_m:4.1.0-q4_k_m.2
//DEPS org.modeljars.github:modeljars.activated-adapters.granite-4.1-3b-answerability-alora-integrallis.f32:1.0.0-f32.1
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
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URL;
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
import java.util.Collections;
import java.util.Enumeration;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarActivatedRuntime;
import org.modeljars.composite.granite.GraniteAnswerability;
import org.modeljars.composite.granite.GraniteAnswerability.Answerability;
import org.modeljars.composite.granite.GraniteAnswerability.Prefix;
import org.modeljars.composite.granite.GraniteAnswerability.Request;
import org.modeljars.composite.granite.GraniteAnswerability.Turn;
import org.modeljars.composite.granite.GraniteAnswerability.Verdict;

/**
 * Clean-host Java 25 COMPOSITION run of the Granite 4.1 3B answerability hybrid.
 *
 * <p>Both members are resolved from Maven Central as ordinary ModelJars markers and opened through
 * the public API ({@link GraniteAnswerability#open()}, which calls {@code
 * ModelJars.openActivatedToolRuntime}). This program downloads no model bytes itself: the only
 * thing it fetches is the frozen qualification window, pinned by SHA-256. Every weight and adapter
 * file is installed by the ModelJars runtime from the verified marker descriptors.
 *
 * <p>Two arms run over the same frozen-window prompts on one loaded hybrid:
 *
 * <ul>
 *   <li><b>control</b> - {@link Prefix#RECOMPUTED}: the base prefix is evaluated independently for
 *       every specialist call, with no physical KV sharing.
 *   <li><b>composite</b> - {@link Prefix#SHARED}: both branches fork from one physical KV prefix.
 * </ul>
 *
 * <p>Every case in both arms must be byte-identical to the committed pilot-2 qualification output
 * and report the same shared-prefix token count; the composite arm must physically share on every
 * case and the control arm must physically share on none. Anything else exits non-zero.
 *
 * <p>Usage: {@code jbang CompositionCleanHost.java [--store <dir>] [--report <file>]
 * [--cases-per-suite <n>] [--arm-order composite-first|control-first]}.
 *
 * <p>Exit 0 only when every case in both arms passed; 1 on any mismatch; 2 when the window download
 * or a hash check fails; 3 when the hybrid could not be opened through the public API.
 */
public final class CompositionCleanHost {
  static final String MODELJARS_VERSION = "0.1.47";
  static final String MODELS_VERSION = "0.3.42";

  static final String BASE_COORDINATE =
      "org.modeljars.huggingface:ibm-granite.granite-4.1-3b-gguf.q4_k_m:4.1.0-q4_k_m.2";
  static final String SPECIALIST_COORDINATE =
      "org.modeljars.github:modeljars.activated-adapters."
          + "granite-4.1-3b-answerability-alora-integrallis.f32:1.0.0-f32.1";

  /** Catalog ids the composition entry names as its members. */
  static final String BASE_MODEL_ID = "ibm_granite_granite_4_1_3b_gguf_q4_k_m";

  static final String SPECIALIST_MODEL_ID =
      "modeljars_granite_4_1_3b_answerability_alora_integrallis_f32";

  // integrallis/models commit cb2f42624d6cb75e8caa9fbe1715233ea7148eb2 froze window v2.
  static final String WINDOW_URL =
      "https://raw.githubusercontent.com/integrallis/models/cb2f42624d6cb75e8caa9fbe1715233ea7148eb2/"
          + "benchmark-results/2026-09-15-granite-4.1-3b-alora-hybrid/qualification-window-v2.json";
  static final String WINDOW_SHA256 = GraniteAnswerability.WINDOW_SHA256;

  /** Default frozen-window cases per suite; every embedded expectation may be used. */
  static final int DEFAULT_CASES_PER_SUITE = 3;

  static final List<String> SUITES = List.of("squad-v2-dev", "msmarco-v2.1-validation");

  /** One committed pilot-2 expectation for a frozen-window case. */
  record Expected(String suite, String id, String output, int sharedPrefixTokens) {}

  /**
   * The first six cases of each suite from the committed pilot-2 qualification evidence
   * (release-pilot2/evidence/pj1/window-squad-v2-dev-specialist-pure-java.json and
   * release-pilot2/evidence/pj3/window-msmarco-v2.1-validation-specialist-pure-java.json).
   */
  static final List<Expected> EXPECTED =
      List.of(
          new Expected("squad-v2-dev", "5ad247b0d7d075001a428b45", "\"unanswerable\"", 269),
          new Expected("squad-v2-dev", "57267640f1498d1400e8e074", "\"answerable\"", 386),
          new Expected("squad-v2-dev", "5737432bc3c5551400e51e9b", "\"answerable\"", 357),
          new Expected("squad-v2-dev", "5705f09e75f01819005e77a4", "\"unanswerable\"", 330),
          new Expected("squad-v2-dev", "57265746dd62a815002e821c", "\"answerable\"", 254),
          new Expected("squad-v2-dev", "5a2c1397bfd06b001a5ae9c9", "\"unanswerable\"", 273),
          new Expected("msmarco-v2.1-validation", "95542", "\"unanswerable\"", 776),
          new Expected("msmarco-v2.1-validation", "851555", "\"answerable\"", 1082),
          new Expected("msmarco-v2.1-validation", "1067349", "\"answerable\"", 853),
          new Expected("msmarco-v2.1-validation", "480961", "\"unanswerable\"", 895),
          new Expected("msmarco-v2.1-validation", "1030759", "\"answerable\"", 1073),
          new Expected("msmarco-v2.1-validation", "806703", "\"unanswerable\"", 1306));

  static final class DownloadFailure extends RuntimeException {
    DownloadFailure(String message) {
      super(message);
    }
  }

  static final class OpenFailure extends RuntimeException {
    OpenFailure(String message, Throwable cause) {
      super(message, cause);
    }
  }

  public static void main(String[] args) throws Exception {
    Path store = Path.of("store");
    Path report = null;
    int casesPerSuite = DEFAULT_CASES_PER_SUITE;
    boolean compositeFirst = true;
    for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
        case "--store" -> store = Path.of(requireValue(args, ++i, "--store"));
        case "--report" -> report = Path.of(requireValue(args, ++i, "--report"));
        case "--cases-per-suite" ->
            casesPerSuite = Integer.parseInt(requireValue(args, ++i, "--cases-per-suite"));
        case "--arm-order" -> {
          String order = requireValue(args, ++i, "--arm-order");
          switch (order) {
            case "composite-first" -> compositeFirst = true;
            case "control-first" -> compositeFirst = false;
            default -> usage("unknown --arm-order " + order);
          }
        }
        default -> usage("unknown argument " + args[i]);
      }
    }
    int maxPerSuite = expectedPerSuite();
    if (casesPerSuite < 1 || casesPerSuite > maxPerSuite) {
      usage("--cases-per-suite must be between 1 and " + maxPerSuite);
    }
    int exit;
    try {
      exit =
          run(
              store.toAbsolutePath().normalize(),
              report == null ? null : report.toAbsolutePath().normalize(),
              casesPerSuite,
              compositeFirst);
    } catch (DownloadFailure failure) {
      System.out.println("FAIL download: " + failure.getMessage());
      exit = 2;
    } catch (OpenFailure failure) {
      System.out.println("FAIL open: " + failure.getMessage());
      Throwable cause = failure.getCause();
      if (cause != null) {
        System.out.println("FAIL cause: " + cause.getClass().getName() + ": " + cause.getMessage());
      }
      exit = 3;
    }
    System.out.flush();
    System.exit(exit);
  }

  static String requireValue(String[] args, int index, String flag) {
    if (index >= args.length) {
      usage("missing value for " + flag);
    }
    return args[index];
  }

  static void usage(String message) {
    System.err.println(message);
    System.err.println(
        "usage: CompositionCleanHost [--store <dir>] [--report <file>] [--cases-per-suite <n>]"
            + " [--arm-order composite-first|control-first]");
    System.exit(64);
  }

  static int expectedPerSuite() {
    int minimum = Integer.MAX_VALUE;
    for (String suite : SUITES) {
      int count = 0;
      for (Expected expected : EXPECTED) {
        if (expected.suite().equals(suite)) {
          count++;
        }
      }
      minimum = Math.min(minimum, count);
    }
    return minimum;
  }

  static int run(Path store, Path reportPath, int casesPerSuite, boolean compositeFirst)
      throws Exception {
    System.out.printf(
        "composition-clean-host modeljarsVersion=%s modelsVersion=%s java=%s vendor=%s os=%s/%s"
            + " processors=%d casesPerSuite=%d armOrder=%s%n",
        MODELJARS_VERSION,
        MODELS_VERSION,
        System.getProperty("java.version"),
        System.getProperty("java.vendor"),
        System.getProperty("os.name"),
        System.getProperty("os.arch"),
        Runtime.getRuntime().availableProcessors(),
        casesPerSuite,
        compositeFirst ? "composite-first" : "control-first");

    Files.createDirectories(store);
    HttpClient http =
        HttpClient.newBuilder()
            .followRedirects(HttpClient.Redirect.NORMAL)
            .connectTimeout(Duration.ofSeconds(30))
            .build();
    Path windowFile =
        fetch(http, WINDOW_URL, store.resolve("qualification-window-v2.json"), WINDOW_SHA256);

    Object window = new Json(Files.readString(windowFile, StandardCharsets.UTF_8)).parse();
    List<Case> cases = new ArrayList<>();
    List<Expected> expectations = new ArrayList<>();
    for (String suite : SUITES) {
      cases.addAll(firstCases(window, suite, casesPerSuite));
      int taken = 0;
      for (Expected expected : EXPECTED) {
        if (expected.suite().equals(suite) && taken < casesPerSuite) {
          expectations.add(expected);
          taken++;
        }
      }
    }
    if (cases.size() != expectations.size()) {
      throw new IllegalStateException(
          "expected " + expectations.size() + " cases, got " + cases.size());
    }

    // Where each member marker was resolved from, taken from the classpath, not asserted.
    Map<String, MarkerSource> markerSources =
        Map.of(
            BASE_MODEL_ID,
            markerSource(BASE_MODEL_ID, BASE_COORDINATE),
            SPECIALIST_MODEL_ID,
            markerSource(SPECIALIST_MODEL_ID, SPECIALIST_COORDINATE));
    for (MarkerSource source : List.copyOf(markerSources.values())) {
      System.out.printf(
          "marker modelId=%s coordinate=%s url=%s mavenLayout=%s%n",
          source.modelId(), source.coordinate(), source.url(), source.mavenLayout());
    }

    List<Measurement> measurements = new ArrayList<>();
    int passed = 0;
    long loadStarted = System.nanoTime();
    ModelJarActivatedRuntime runtime;
    try {
      runtime = GraniteAnswerability.open();
    } catch (RuntimeException failure) {
      throw new OpenFailure(
          "ModelJars could not open the hybrid through its public API"
              + " (ModelJars.openActivatedToolRuntime). Both markers resolved from Maven Central;"
              + " the failure is a runtime or catalog precondition, not a download.",
          failure);
    }
    try (ModelJarActivatedRuntime hybrid = runtime) {
      ModelJarDescriptor base = hybrid.baseDescriptor();
      ModelJarDescriptor specialist = hybrid.adapterDescriptor();
      System.out.printf(
          "loaded backend=pure-java baseCoordinate=%s baseSha256=%s specialistCoordinate=%s"
              + " specialistSha256=%s minimumSharedPrefixTokens=%d chatTemplate=%s millis=%d%n",
          base.markerCoordinate(),
          base.sha256().orElse("absent"),
          specialist.markerCoordinate(),
          specialist.sha256().orElse("absent"),
          hybrid.model().minimumSharedPrefixTokens(),
          hybrid.chatTemplate(),
          (System.nanoTime() - loadStarted) / 1_000_000L);

      List<Prefix> order =
          compositeFirst
              ? List.of(Prefix.SHARED, Prefix.RECOMPUTED)
              : List.of(Prefix.RECOMPUTED, Prefix.SHARED);
      for (int index = 0; index < cases.size(); index++) {
        Case item = cases.get(index);
        Expected expected = expectations.get(index);
        if (!expected.suite().equals(item.suite()) || !expected.id().equals(item.id())) {
          System.out.printf(
              "FAIL window case %d is %s/%s, expected %s/%s%n",
              index, item.suite(), item.id(), expected.suite(), expected.id());
          return 1;
        }
        Request request = new Request(item.documents(), item.conversation());
        String promptSha256 = sha256(text(GraniteAnswerability.render(request)));
        for (Prefix prefix : order) {
          Verdict verdict = GraniteAnswerability.classify(hybrid, request, prefix);
          boolean shouldShare = prefix == Prefix.SHARED;
          boolean structured = verdict.structured();
          boolean identical = expected.output().equals(verdict.output());
          boolean sameShared = expected.sharedPrefixTokens() == verdict.sharedPrefixTokens();
          boolean sharingCorrect = verdict.physicallySharesPrefix() == shouldShare;
          boolean ok = structured && identical && sameShared && sharingCorrect;
          if (ok) {
            passed++;
          }
          measurements.add(
              new Measurement(
                  arm(prefix),
                  item.suite(),
                  item.id(),
                  promptSha256,
                  verdict.output(),
                  verdict.answerability(),
                  structured,
                  verdict.physicallySharesPrefix(),
                  verdict.sharedPrefixTokens(),
                  verdict.millis(),
                  ok));
          System.out.printf(
              "CASE %s arm=%s suite=%s id=%s promptSha256=%s output=%s label=%s structured=%s"
                  + " physicallySharesPrefix=%s expectedPhysicalSharing=%s sharedPrefixTokens=%d"
                  + " expectedOutput=%s expectedSharedPrefixTokens=%d byteIdentical=%s millis=%d%n",
              ok ? "PASS" : "FAIL",
              arm(prefix),
              item.suite(),
              item.id(),
              promptSha256,
              jsonString(verdict.output()),
              verdict.answerability(),
              structured,
              verdict.physicallySharesPrefix(),
              shouldShare,
              verdict.sharedPrefixTokens(),
              jsonString(expected.output()),
              expected.sharedPrefixTokens(),
              identical,
              verdict.millis());
        }
      }

      double controlMedian = median(millis(measurements, "control"));
      double compositeMedian = median(millis(measurements, "composite"));
      double improvement = (controlMedian - compositeMedian) / controlMedian;
      System.out.printf(
          Locale.ROOT,
          "MEDIANS controlMedianMillis=%.6f compositeMedianMillis=%.6f latencyImprovement=%.16f%n",
          controlMedian,
          compositeMedian,
          improvement);

      int expectedMeasurements = cases.size() * 2;
      boolean pass = passed == expectedMeasurements;
      if (reportPath != null) {
        writeReport(
            reportPath,
            hybrid,
            markerSources,
            measurements,
            casesPerSuite,
            compositeFirst,
            controlMedian,
            compositeMedian,
            improvement,
            pass);
        System.out.printf("wrote %s%n", reportPath);
      }
      System.out.printf(
          "%s measurements=%d passed=%d%n",
          pass ? "PASS" : "FAIL", expectedMeasurements, passed);
      return pass ? 0 : 1;
    }
  }

  static String arm(Prefix prefix) {
    return prefix == Prefix.SHARED ? "composite" : "control";
  }

  static String text(ModelPrompt prompt) {
    StringBuilder out = new StringBuilder();
    for (ModelPrompt.Segment segment : prompt.segments()) {
      out.append(segment.text());
    }
    return out.toString();
  }

  record Measurement(
      String arm,
      String suite,
      String id,
      String promptSha256,
      String output,
      Answerability label,
      boolean structured,
      boolean physicallySharesPrefix,
      int sharedPrefixTokens,
      long millis,
      boolean pass) {}

  record MarkerSource(String modelId, String coordinate, String url, boolean mavenLayout) {}

  static List<Long> millis(List<Measurement> measurements, String arm) {
    List<Long> values = new ArrayList<>();
    for (Measurement measurement : measurements) {
      if (measurement.arm().equals(arm)) {
        values.add(measurement.millis());
      }
    }
    return values;
  }

  /** Median with the usual even-count convention: the mean of the two central values. */
  static double median(List<Long> values) {
    if (values.isEmpty()) {
      throw new IllegalArgumentException("no measurements to take a median of");
    }
    List<Long> sorted = new ArrayList<>(values);
    Collections.sort(sorted);
    int size = sorted.size();
    if (size % 2 == 1) {
      return sorted.get(size / 2);
    }
    return (sorted.get(size / 2 - 1) + sorted.get(size / 2)) / 2.0;
  }

  /**
   * Finds the classpath marker JAR that declares a model id and checks it sits at the Maven
   * repository layout its coordinate implies. Nothing here trusts the coordinate string: the
   * descriptor was loaded from this resource, so this is where it actually came from.
   */
  static MarkerSource markerSource(String modelId, String coordinate) throws IOException {
    ClassLoader loader = CompositionCleanHost.class.getClassLoader();
    Enumeration<URL> resources = loader.getResources("META-INF/modeljars/registry.properties");
    while (resources.hasMoreElements()) {
      URL resource = resources.nextElement();
      Properties properties = new Properties();
      try (InputStream input = resource.openStream()) {
        properties.load(input);
      }
      String declared = properties.getProperty("model." + modelId + ".markerCoordinate");
      if (declared == null) {
        continue;
      }
      if (!declared.equals(coordinate)) {
        throw new IllegalStateException(
            "marker for " + modelId + " declares " + declared + ", expected " + coordinate);
      }
      return new MarkerSource(
          modelId, coordinate, resource.toString(), atMavenLayout(resource.toString(), coordinate));
    }
    throw new IllegalStateException("no classpath marker declares " + modelId);
  }

  /** True when the resolved JAR URL ends at {@code <group as path>/<artifact>/<version>/}. */
  static boolean atMavenLayout(String url, String coordinate) {
    String[] parts = coordinate.split(":");
    if (parts.length != 3) {
      return false;
    }
    String expected =
        "/" + parts[0].replace('.', '/') + "/" + parts[1] + "/" + parts[2] + "/" + parts[1] + "-"
            + parts[2] + ".jar";
    return url.contains(expected);
  }

  static void writeReport(
      Path path,
      ModelJarActivatedRuntime hybrid,
      Map<String, MarkerSource> markerSources,
      List<Measurement> measurements,
      int casesPerSuite,
      boolean compositeFirst,
      double controlMedian,
      double compositeMedian,
      double improvement,
      boolean pass)
      throws IOException {
    ModelJarDescriptor base = hybrid.baseDescriptor();
    ModelJarDescriptor specialist = hybrid.adapterDescriptor();
    StringBuilder out = new StringBuilder();
    out.append("{\n");
    out.append("  \"schemaVersion\": 1,\n");
    out.append("  \"pass\": ").append(pass).append(",\n");
    out.append("  \"modeljarsVersion\": ").append(jsonString(MODELJARS_VERSION)).append(",\n");
    out.append("  \"modelsVersion\": ").append(jsonString(MODELS_VERSION)).append(",\n");
    out.append("  \"windowSha256\": ").append(jsonString(WINDOW_SHA256)).append(",\n");
    out.append("  \"casesPerSuite\": ").append(casesPerSuite).append(",\n");
    out.append("  \"armOrder\": ")
        .append(jsonString(compositeFirst ? "composite-first" : "control-first"))
        .append(",\n");
    out.append("  \"minimumSharedPrefixTokens\": ")
        .append(hybrid.model().minimumSharedPrefixTokens())
        .append(",\n");
    out.append("  \"chatTemplate\": ")
        .append(jsonString(String.valueOf(hybrid.chatTemplate())))
        .append(",\n");
    out.append("  \"controlMedianMillis\": ").append(number(controlMedian)).append(",\n");
    out.append("  \"compositeMedianMillis\": ").append(number(compositeMedian)).append(",\n");
    out.append("  \"latencyImprovement\": ").append(number(improvement)).append(",\n");
    out.append("  \"publishedArtifacts\": [\n");
    appendArtifact(out, BASE_MODEL_ID, base, markerSources.get(BASE_MODEL_ID), pass);
    out.append(",\n");
    appendArtifact(out, SPECIALIST_MODEL_ID, specialist, markerSources.get(SPECIALIST_MODEL_ID), pass);
    out.append("\n  ],\n");
    out.append("  \"measurements\": [\n");
    for (int i = 0; i < measurements.size(); i++) {
      Measurement measurement = measurements.get(i);
      out.append("    {")
          .append("\"arm\": ").append(jsonString(measurement.arm()))
          .append(", \"suite\": ").append(jsonString(measurement.suite()))
          .append(", \"id\": ").append(jsonString(measurement.id()))
          .append(", \"promptSha256\": ").append(jsonString(measurement.promptSha256()))
          .append(", \"output\": ").append(jsonString(measurement.output()))
          .append(", \"label\": ").append(jsonString(String.valueOf(measurement.label())))
          .append(", \"structured\": ").append(measurement.structured())
          .append(", \"physicallySharesPrefix\": ").append(measurement.physicallySharesPrefix())
          .append(", \"sharedPrefixTokens\": ").append(measurement.sharedPrefixTokens())
          .append(", \"millis\": ").append(measurement.millis())
          .append(", \"pass\": ").append(measurement.pass())
          .append("}");
      out.append(i + 1 < measurements.size() ? ",\n" : "\n");
    }
    out.append("  ]\n");
    out.append("}\n");
    Files.writeString(path, out.toString(), StandardCharsets.UTF_8);
  }

  static void appendArtifact(
      StringBuilder out,
      String modelId,
      ModelJarDescriptor descriptor,
      MarkerSource source,
      boolean pass) {
    out.append("    {")
        .append("\"modelId\": ").append(jsonString(modelId))
        .append(", \"coordinate\": ")
        .append(jsonString(String.valueOf(descriptor.markerCoordinate())))
        .append(", \"sha256\": ").append(jsonString(descriptor.sha256().orElse("")))
        .append(", \"resolvedFromCentral\": ").append(source != null && source.mavenLayout())
        .append(", \"runViaPublicApi\": ").append(pass)
        .append(", \"markerResourceUrl\": ")
        .append(jsonString(source == null ? "" : source.url()))
        .append("}");
  }

  /** Emits a JSON number that round-trips: integral doubles keep a fractional part. */
  static String number(double value) {
    if (!Double.isFinite(value)) {
      throw new IllegalArgumentException("not a finite measurement: " + value);
    }
    String text = Double.toString(value);
    return text.endsWith(".0") ? text.substring(0, text.length() - 2) : text;
  }

  record Case(String suite, String id, List<Turn> conversation, List<String> documents) {}

  @SuppressWarnings("unchecked")
  static List<Case> firstCases(Object window, String suiteName, int limit) {
    Map<String, Object> root = (Map<String, Object>) window;
    if (!Long.valueOf(1).equals(root.get("schemaVersion"))) {
      throw new IllegalArgumentException("invalid qualification window");
    }
    for (Object suiteNode : (List<Object>) root.get("suites")) {
      Map<String, Object> suite = (Map<String, Object>) suiteNode;
      if (!suiteName.equals(suite.get("name"))) {
        continue;
      }
      List<Case> cases = new ArrayList<>();
      for (Object caseNode : (List<Object>) suite.get("cases")) {
        if (cases.size() == limit) {
          break;
        }
        Map<String, Object> node = (Map<String, Object>) caseNode;
        String id = (String) node.get("id");
        List<Turn> conversation = new ArrayList<>();
        for (Object m : (List<Object>) node.get("messages")) {
          Map<String, Object> message = (Map<String, Object>) m;
          conversation.add(new Turn((String) message.get("role"), (String) message.get("text")));
        }
        List<String> documents = new ArrayList<>();
        for (Object d : (List<Object>) node.get("documents")) {
          Map<String, Object> document = (Map<String, Object>) d;
          Object docId = document.get("doc_id");
          String docText = (String) document.get("text");
          if (!(docId instanceof Long l && l > 0) || docText == null || docText.isBlank()) {
            throw new IllegalArgumentException("invalid document in case: " + id);
          }
          documents.add(docText);
        }
        cases.add(new Case(suiteName, id, List.copyOf(conversation), List.copyOf(documents)));
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
        System.out.printf(
            "present %s sha256=%s bytes=%d%n", target.getFileName(), sha256, Files.size(target));
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
            http.send(
                HttpRequest.newBuilder(URI.create(url)).GET().build(),
                HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
          response.body().close();
          failure = "HTTP " + response.statusCode() + " for " + url;
        } else {
          MessageDigest digest = MessageDigest.getInstance("SHA-256");
          try (InputStream in = response.body();
              OutputStream out = Files.newOutputStream(part)) {
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
          System.out.printf(
              "fetched %s sha256=%s bytes=%d%n", target.getFileName(), sha256, Files.size(target));
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
      for (int n; (n = in.read(buffer)) >= 0; ) {
        digest.update(buffer, 0, n);
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  static String sha256(String value) throws Exception {
    return HexFormat.of()
        .formatHex(
            MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8)));
  }

  /** ASCII-safe JSON string literal for log lines and the report. */
  static String jsonString(String value) {
    if (value == null) {
      return "null";
    }
    StringBuilder out = new StringBuilder("\"");
    for (char c : value.toCharArray()) {
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        default -> {
          if (c < 0x20 || c > 0x7e) {
            out.append(String.format("\\u%04x", (int) c));
          } else {
            out.append(c);
          }
        }
      }
    }
    return out.append('"').toString();
  }

  /**
   * Minimal strict JSON reader (objects, arrays, strings, numbers, booleans, null) so the script
   * needs no dependency beyond the published ModelJars artifacts.
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
      if (i != s.length()) {
        throw error("trailing content");
      }
      return value;
    }

    private Object value() {
      ws();
      if (i >= s.length()) {
        throw error("unexpected end");
      }
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
        if (peek() != '"') {
          throw error("expected key");
        }
        String key = string();
        ws();
        expect(':');
        if (map.put(key, value()) != null) {
          throw error("duplicate key " + key);
        }
        ws();
        char c = next();
        if (c == '}') {
          return map;
        }
        if (c != ',') {
          throw error("expected , or }");
        }
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
        if (c == ']') {
          return list;
        }
        if (c != ',') {
          throw error("expected , or ]");
        }
      }
    }

    private String string() {
      expect('"');
      StringBuilder out = new StringBuilder();
      while (true) {
        char c = next();
        if (c == '"') {
          return out.toString();
        }
        if (c < 0x20) {
          throw error("control character in string");
        }
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
            if (i + 4 > s.length()) {
              throw error("short unicode escape");
            }
            out.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
            i += 4;
          }
          default -> throw error("bad escape");
        }
      }
    }

    private Object number() {
      int start = i;
      if (peek() == '-') {
        i++;
      }
      while (i < s.length() && "0123456789.eE+-".indexOf(s.charAt(i)) >= 0) {
        i++;
      }
      String token = s.substring(start, i);
      if (token.isEmpty() || token.equals("-")) {
        throw error("bad value");
      }
      if (token.chars().allMatch(ch -> ch == '-' || Character.isDigit(ch))) {
        return Long.parseLong(token);
      }
      return Double.parseDouble(token);
    }

    private Object literal(String word, Object value) {
      if (!s.startsWith(word, i)) {
        throw error("bad literal");
      }
      i += word.length();
      return value;
    }

    private void ws() {
      while (i < s.length() && " \t\r\n".indexOf(s.charAt(i)) >= 0) {
        i++;
      }
    }

    private char peek() {
      return i < s.length() ? s.charAt(i) : '\0';
    }

    private char next() {
      if (i >= s.length()) {
        throw error("unexpected end");
      }
      return s.charAt(i++);
    }

    private void expect(char c) {
      if (next() != c) {
        throw error("expected " + c);
      }
    }

    private IllegalArgumentException error(String message) {
      return new IllegalArgumentException("window JSON: " + message + " at offset " + i);
    }
  }
}
