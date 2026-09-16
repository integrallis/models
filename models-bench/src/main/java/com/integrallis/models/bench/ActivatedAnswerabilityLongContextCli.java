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
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.TokenConstraint;
import com.integrallis.models.runtime.chat.GraniteDocumentsPrompt;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Gate 5 for an answerability specialist: semantic and token-exact retention across a physical
 * 4,096-token shared documents prefix.
 *
 * <p>Each case plants an archive fact as the first document, pads a second document to an exact
 * 4,096-token activation prefix, and asks for one route's archive code. The specialist answers the
 * answerability label from the shared prefix; the base branch then answers the same question from
 * the same physical blocks and must reproduce a native base run byte for byte. Answerable cases ask
 * about the planted route; unanswerable cases ask about a route that no document mentions.
 */
final class ActivatedAnswerabilityLongContextCli {
  static final String POLICY_VERSION = "activated-answerability-long-context-retention-v1";
  static final String SUITE_RESOURCE =
      "answerability-qualification/activated-answerability-long-context-v1.json";
  static final double MINIMUM_NATIVE_CORRECT_RATE = 0.75;
  static final double MINIMUM_SPECIALIST_CORRECT_RATE = 0.75;
  private static final Set<String> OPTIONS =
      Set.of("model", "adapter", "report", "models-revision", "max-tokens");
  private static final List<String> REFUSAL_MARKERS =
      List.of(
          "cannot", "can't", "not available", "does not", "doesn't", "no information", "unable");

  private ActivatedAnswerabilityLongContextCli() {}

  record Configuration(
      Path model, Path adapter, Path report, String modelsRevision, int maxTokens) {}

  record Case(String id, String route, String archiveCode, String askedRoute, String label) {}

  record Suite(int schemaVersion, String workload, int targetPrefixTokens, List<Case> cases) {
    Suite {
      cases = List.copyOf(cases);
    }
  }

  record SuiteIdentity(String resource, String sha256) {}

  record CaseResult(
      String id,
      String label,
      int prefixTokens,
      boolean specialistCorrect,
      boolean nativeCorrect,
      boolean retainedCorrect,
      boolean exactOutputMatch,
      boolean physicallyShared,
      String specialistOutput,
      String nativeOutput,
      String sharedOutput,
      long nativeMillis,
      long specialistMillis,
      long sharedMillis) {}

  record Summary(
      int attempts,
      int specialistCorrect,
      int nativeCorrect,
      int retainedNativeCorrect,
      int exactOutputMatches,
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
      double minimumSpecialistCorrectRate,
      int maxTokens,
      List<CaseResult> cases,
      Summary summary,
      boolean qualified) {}

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    Path model = Path.of(required(values, "model"));
    Path adapter = Path.of(required(values, "adapter"));
    String revision = required(values, "models-revision");
    if (!revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact 40-character Git SHA");
    }
    int maxTokens = BenchmarkCliArguments.integer(values, "max-tokens", 48);
    if (maxTokens < 8 || maxTokens > 256) {
      throw new IllegalArgumentException("--max-tokens must be between 8 and 256");
    }
    return new Configuration(
        model, adapter, Path.of(required(values, "report")), revision, maxTokens);
  }

  static Suite loadSuite() throws IOException {
    try (InputStream input = resource()) {
      Suite suite = new ObjectMapper().readValue(input, Suite.class);
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
    SamplingOptions specialistOptions =
        SamplingOptions.builder().temperature(0).maxTokens(6).build();

    try (PureJavaBackend backend =
            PureJavaBackend.loadActivatedAdapter(configuration.model(), configuration.adapter());
        ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1)) {
      Set<String> allCodes = new HashSet<>();
      for (Case item : suite.cases()) {
        allCodes.add(item.archiveCode());
      }
      for (Case item : suite.cases()) {
        ModelPrompt prompt =
            exactPrefixPrompt(model.tokenizer(), model.adapter(), suite.targetPrefixTokens(), item);
        int prefixTokens = activationBoundary(model.tokenizer().encode(prompt), model.adapter());

        long nativeStarted = System.nanoTime();
        String nativeOutput = model.generate(prompt, responseOptions);
        long nativeMillis = (System.nanoTime() - nativeStarted) / 1_000_000L;

        String specialistOutput;
        String sharedOutput;
        long specialistMillis;
        long sharedMillis;
        boolean physicallyShared;
        try (ActivatedToolTurn turn =
            model.openToolTurn(prompt, ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
          physicallyShared = turn.physicallySharesPrefix();
          long specialistStarted = System.nanoTime();
          specialistOutput =
              turn.generateToolCall(specialistOptions, TokenConstraint.unrestricted());
          specialistMillis = (System.nanoTime() - specialistStarted) / 1_000_000L;
          long sharedStarted = System.nanoTime();
          sharedOutput = turn.generateBaseResponse(prompt, responseOptions);
          sharedMillis = (System.nanoTime() - sharedStarted) / 1_000_000L;
        }

        boolean nativeCorrect = correctAnswer(nativeOutput, item, allCodes);
        CaseResult result =
            new CaseResult(
                item.id(),
                item.label(),
                prefixTokens,
                item.label()
                    .equals(ActivatedAnswerabilityQualificationCli.prediction(specialistOutput)),
                nativeCorrect,
                correctAnswer(sharedOutput, item, allCodes),
                nativeOutput.equals(sharedOutput),
                physicallyShared,
                specialistOutput,
                nativeOutput,
                sharedOutput,
                nativeMillis,
                specialistMillis,
                sharedMillis);
        results.add(result);
        System.out.printf(
            "%-6s %-12s prefix=%d specialist=%s native=%s retained=%s exact=%s shared=%s%n",
            item.id(),
            item.label(),
            prefixTokens,
            result.specialistCorrect(),
            result.nativeCorrect(),
            result.retainedCorrect(),
            result.exactOutputMatch(),
            result.physicallyShared());
      }

      Summary summary = summarize(results, suite.targetPrefixTokens());
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
              MINIMUM_SPECIALIST_CORRECT_RATE,
              configuration.maxTokens(),
              List.copyOf(results),
              summary,
              summary.qualified());
      Files.createDirectories(configuration.report().toAbsolutePath().getParent());
      new ObjectMapper()
          .writerWithDefaultPrettyPrinter()
          .writeValue(configuration.report().toFile(), report);
      System.out.printf(
          "%s specialist=%d/%d native=%d/%d retained=%d/%d exact=%d/%d shared=%d/%d report=%s%n",
          summary.qualified() ? "PASS" : "FAIL",
          summary.specialistCorrect(),
          summary.attempts(),
          summary.nativeCorrect(),
          summary.attempts(),
          summary.retainedNativeCorrect(),
          summary.nativeCorrect(),
          summary.exactOutputMatches(),
          summary.attempts(),
          summary.physicallyShared(),
          summary.attempts(),
          configuration.report().toAbsolutePath());
      return summary.qualified() ? 0 : 1;
    }
  }

  static Summary summarize(List<CaseResult> results, int targetPrefixTokens) {
    int attempts = results.size();
    int specialist = (int) results.stream().filter(CaseResult::specialistCorrect).count();
    int nativeCorrect = (int) results.stream().filter(CaseResult::nativeCorrect).count();
    int retained =
        (int)
            results.stream()
                .filter(CaseResult::nativeCorrect)
                .filter(CaseResult::retainedCorrect)
                .count();
    int exact = (int) results.stream().filter(CaseResult::exactOutputMatch).count();
    int shared = (int) results.stream().filter(CaseResult::physicallyShared).count();
    int exactPrefix =
        (int)
            results.stream().filter(result -> result.prefixTokens() == targetPrefixTokens).count();
    boolean qualified =
        attempts > 0
            && nativeCorrect >= (int) Math.ceil(attempts * MINIMUM_NATIVE_CORRECT_RATE)
            && specialist >= (int) Math.ceil(attempts * MINIMUM_SPECIALIST_CORRECT_RATE)
            && retained == nativeCorrect
            && exact == attempts
            && shared == attempts
            && exactPrefix == attempts;
    return new Summary(
        attempts, specialist, nativeCorrect, retained, exact, shared, exactPrefix, qualified);
  }

  /**
   * An answerable case is correct when the output names the planted code. An unanswerable case is
   * correct when the output names no code from the suite and says the answer is unavailable.
   */
  static boolean correctAnswer(String output, Case item, Set<String> allCodes) {
    String upper = output == null ? "" : output.toUpperCase(Locale.ROOT);
    if ("answerable".equals(item.label())) {
      return upper.contains(item.archiveCode().toUpperCase(Locale.ROOT));
    }
    for (String code : allCodes) {
      if (upper.contains(code.toUpperCase(Locale.ROOT))) {
        return false;
      }
    }
    String lower = output == null ? "" : output.toLowerCase(Locale.ROOT);
    return REFUSAL_MARKERS.stream().anyMatch(lower::contains);
  }

  static String question(Case item) {
    return "What is the archive code for route " + item.askedRoute() + "?";
  }

  static String factDocument(Case item) {
    return "Transit archive fact: route "
        + item.route()
        + " has archive code "
        + item.archiveCode()
        + ".";
  }

  static ModelPrompt prompt(Case item, String filler) {
    ModelPrompt.Builder builder =
        GraniteDocumentsPrompt.appendSystem(
            ModelPrompt.builder(),
            List.of(factDocument(item), "Background records:" + filler),
            null);
    GraniteDocumentsPrompt.appendTurn(builder, "user", question(item));
    return GraniteDocumentsPrompt.finish(builder);
  }

  static ModelPrompt exactPrefixPrompt(
      Tokenizer tokenizer, ActivatedAdapterMetadata adapter, int targetPrefixTokens, Case item) {
    String unit =
        List.of(" transit", " context", " data", " x").stream()
            .filter(candidate -> tokenizer.encode(candidate).length == 1)
            .findFirst()
            .orElseThrow(() -> new IllegalStateException("no stable one-token filler was found"));
    int low = 0;
    int high = targetPrefixTokens * 2;
    while (low <= high) {
      int repetitions = (low + high) >>> 1;
      ModelPrompt candidate = prompt(item, unit.repeat(repetitions));
      int prefix = activationBoundary(tokenizer.encode(candidate), adapter);
      if (prefix == targetPrefixTokens) {
        return candidate;
      }
      if (prefix < targetPrefixTokens) {
        low = repetitions + 1;
      } else {
        high = repetitions - 1;
      }
    }
    throw new IllegalStateException(
        "could not construct an exact " + targetPrefixTokens + "-token prefix for " + item.id());
  }

  static int activationBoundary(int[] promptTokens, ActivatedAdapterMetadata adapter) {
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
    throw new IllegalStateException("prompt omitted the adapter invocation");
  }

  static void validateSuite(Suite suite) {
    if (suite.schemaVersion() != 1
        || suite.workload() == null
        || suite.workload().isBlank()
        || suite.targetPrefixTokens() != 4_096
        || suite.cases().size() != 8) {
      throw new IllegalArgumentException("invalid answerability long-context suite");
    }
    Set<String> ids = new HashSet<>();
    Set<String> codes = new HashSet<>();
    Set<String> routes = new HashSet<>();
    for (Case item : suite.cases()) {
      boolean complete =
          item.id() != null
              && !item.id().isBlank()
              && item.route() != null
              && !item.route().isBlank()
              && item.archiveCode() != null
              && !item.archiveCode().isBlank()
              && item.askedRoute() != null
              && !item.askedRoute().isBlank()
              && ActivatedAnswerabilityQualificationCli.LABELS.contains(item.label());
      if (!complete || !ids.add(item.id()) || !codes.add(item.archiveCode())) {
        throw new IllegalArgumentException("long-context suite cases must be complete and unique");
      }
      routes.add(item.route());
      boolean asksPlanted = item.route().equals(item.askedRoute());
      if (asksPlanted != "answerable".equals(item.label())) {
        throw new IllegalArgumentException(
            "case " + item.id() + " must ask about the planted route exactly when answerable");
      }
    }
    for (Case item : suite.cases()) {
      if ("unanswerable".equals(item.label()) && routes.contains(item.askedRoute())) {
        throw new IllegalArgumentException(
            "unanswerable case " + item.id() + " asks about a route another case plants");
      }
    }
  }

  private static SuiteIdentity suiteIdentity() throws IOException {
    try (InputStream input = resource()) {
      return new SuiteIdentity(
          SUITE_RESOURCE, Hashing.sha256(new String(input.readAllBytes(), StandardCharsets.UTF_8)));
    }
  }

  private static InputStream resource() throws IOException {
    InputStream input =
        ActivatedAnswerabilityLongContextCli.class
            .getClassLoader()
            .getResourceAsStream(SUITE_RESOURCE);
    if (input == null) {
      throw new IOException("missing answerability long-context suite: " + SUITE_RESOURCE);
    }
    return input;
  }

  private static long directoryBytes(Path directory) throws IOException {
    try (var files = Files.walk(directory)) {
      return files
          .filter(Files::isRegularFile)
          .mapToLong(ActivatedAnswerabilityLongContextCli::size)
          .sum();
    }
  }

  private static long size(Path path) {
    try {
      return Files.size(path);
    } catch (IOException failure) {
      throw new IllegalStateException(failure);
    }
  }

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }
}
