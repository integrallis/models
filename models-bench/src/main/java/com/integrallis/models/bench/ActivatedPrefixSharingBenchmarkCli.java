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
import com.fasterxml.jackson.datatype.jdk8.Jdk8Module;
import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.GenerationMetrics;
import com.integrallis.models.runtime.TokenConstraint;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** Measures an activated adapter against independent prefix recomputation on the same model. */
final class ActivatedPrefixSharingBenchmarkCli {
  static final String POLICY_VERSION = "activated-prefix-sharing-crossover-v3";
  private static final List<Integer> PREFIX_TIERS = List.of(256, 1_024, 4_096);
  private static final int EXACT_OUTPUT_TOKENS = 8;
  private static final double MINIMUM_FOUR_K_IMPROVEMENT = 0.20;
  private static final Set<String> OPTIONS =
      Set.of("model", "adapter", "report", "models-revision", "warmups", "trials", "template");
  private static final Set<String> TEMPLATES = Set.of("chatml", "granite-documents");

  private ActivatedPrefixSharingBenchmarkCli() {}

  record Configuration(
      Path model,
      Path adapter,
      Path report,
      String modelsRevision,
      int warmups,
      int trials,
      String template) {}

  record Measurement(
      int prefixTokens,
      String strategy,
      int trial,
      long openNanos,
      long generationTimeToFirstTokenNanos,
      long handoffTimeToFirstTokenNanos,
      long totalNanos,
      int sharedPrefixTokens,
      long sharedPrefixBytes,
      long baseInferenceStateBytes,
      long toolInferenceStateBytes,
      long uniqueInferenceStateBytes,
      boolean physicallyShared,
      String output,
      List<Integer> outputTokenIds,
      ProcessMemory.Snapshot processBefore,
      ProcessMemory.Snapshot processAfter,
      JvmMemorySnapshot jvmBefore,
      JvmMemorySnapshot jvmAfter) {}

  record TierSummary(
      int prefixTokens,
      double sharedMedianHandoffMillis,
      double recomputedMedianHandoffMillis,
      double improvement,
      long sharedMedianUniqueStateBytes,
      long recomputedMedianUniqueStateBytes,
      boolean sharedPhysicalContractPassed,
      boolean recomputedIndependencePassed,
      boolean tokenExactAcrossStrategies) {}

  record Verdict(
      Optional<Integer> crossoverPrefixTokens, double fourKImprovement, boolean passed) {}

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
      int warmups,
      int trials,
      List<Integer> prefixTiers,
      List<Measurement> measurements,
      List<TierSummary> summaries,
      Verdict verdict,
      long peakRssBytes,
      boolean memoryAccountingComplete,
      boolean qualified) {}

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    Path model = requiredFile(values, "model");
    Path adapter = requiredDirectory(values, "adapter");
    String modelsRevision = values.get("models-revision");
    if (modelsRevision == null || !modelsRevision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact 40-character Git SHA");
    }
    int warmups = BenchmarkCliArguments.integer(values, "warmups", 1);
    int trials = BenchmarkCliArguments.integer(values, "trials", 3);
    if (warmups < 0 || warmups > 5) {
      throw new IllegalArgumentException("--warmups must be between 0 and 5");
    }
    if (trials < 1 || trials > 10) {
      throw new IllegalArgumentException("--trials must be between 1 and 10");
    }
    Path report =
        Path.of(
            values.getOrDefault(
                "report", "build/reports/activated-prefix-sharing/comparison.json"));
    String template = values.getOrDefault("template", "chatml");
    if (!TEMPLATES.contains(template)) {
      throw new IllegalArgumentException("--template must be chatml or granite-documents");
    }
    return new Configuration(model, adapter, report, modelsRevision, warmups, trials, template);
  }

  static int run(String[] args) throws IOException {
    Configuration configuration = parse(args);
    List<Measurement> measurements = new ArrayList<>();
    List<TierSummary> summaries = new ArrayList<>();
    BenchmarkEnvironment environment = BenchmarkEnvironment.capture();
    long peakRssBytes = 0;
    ActivatedAdapterMetadata adapter;
    List<ModelPrompt> prompts = new ArrayList<>();

    try (PureJavaBackend backend =
            PureJavaBackend.loadActivatedAdapter(configuration.model(), configuration.adapter());
        ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1)) {
      adapter = model.adapter();
      for (int prefixTokens : PREFIX_TIERS) {
        prompts.add(
            promptWithExactPrefix(
                model.tokenizer(), adapter, prefixTokens, configuration.template()));
      }
      SamplingOptions options =
          SamplingOptions.builder().temperature(0).maxTokens(EXACT_OUTPUT_TOKENS).build();
      TokenConstraint nonTerminal = nonTerminal(model.tokenizer());

      for (int index = 0; index < PREFIX_TIERS.size(); index++) {
        int prefixTokens = PREFIX_TIERS.get(index);
        ModelPrompt prompt = prompts.get(index);
        for (int warmup = 0; warmup < configuration.warmups(); warmup++) {
          measure(model, prompt, prefixTokens, PrefixMode.SHARED, -1, options, nonTerminal);
          measure(model, prompt, prefixTokens, PrefixMode.RECOMPUTED, -1, options, nonTerminal);
        }
        for (int trial = 0; trial < configuration.trials(); trial++) {
          if ((trial & 1) == 0) {
            measurements.add(
                measure(
                    model,
                    prompt,
                    prefixTokens,
                    PrefixMode.RECOMPUTED,
                    trial,
                    options,
                    nonTerminal));
            measurements.add(
                measure(
                    model, prompt, prefixTokens, PrefixMode.SHARED, trial, options, nonTerminal));
          } else {
            measurements.add(
                measure(
                    model, prompt, prefixTokens, PrefixMode.SHARED, trial, options, nonTerminal));
            measurements.add(
                measure(
                    model,
                    prompt,
                    prefixTokens,
                    PrefixMode.RECOMPUTED,
                    trial,
                    options,
                    nonTerminal));
          }
        }
        summaries.add(summarize(prefixTokens, measurements));
      }
    }

    for (Measurement measurement : measurements) {
      peakRssBytes = Math.max(peakRssBytes, measurement.processAfter().highWaterBytes());
    }
    boolean memoryComplete =
        peakRssBytes > 0
            && measurements.stream()
                .allMatch(
                    measurement ->
                        measurement.baseInferenceStateBytes() > 0
                            && measurement.toolInferenceStateBytes() > 0
                            && measurement.uniqueInferenceStateBytes() > 0
                            && measurement.jvmAfter().nativeMemory().available());
    Verdict verdict = verdict(summaries);
    boolean qualified = verdict.passed() && memoryComplete;
    Report report =
        new Report(
            3,
            Instant.now().toString(),
            POLICY_VERSION,
            configuration.modelsRevision(),
            Hashing.sha256(configuration.model()),
            Files.size(configuration.model()),
            adapter,
            directoryBytes(configuration.adapter()),
            environment,
            configuration.warmups(),
            configuration.trials(),
            PREFIX_TIERS,
            List.copyOf(measurements),
            List.copyOf(summaries),
            verdict,
            peakRssBytes,
            memoryComplete,
            qualified);
    write(configuration.report(), report);
    System.out.printf(
        "%s crossover=%s 4k-improvement=%.1f%% peak-rss=%d report=%s%n",
        qualified ? "PASS" : "FAIL",
        verdict.crossoverPrefixTokens().map(Object::toString).orElse("none"),
        verdict.fourKImprovement() * 100,
        peakRssBytes,
        configuration.report().toAbsolutePath());
    return qualified ? 0 : 1;
  }

  static Verdict verdict(List<TierSummary> summaries) {
    Optional<Integer> crossover =
        summaries.stream()
            .filter(
                summary ->
                    summary.sharedMedianHandoffMillis() < summary.recomputedMedianHandoffMillis())
            .map(TierSummary::prefixTokens)
            .min(Integer::compareTo);
    TierSummary fourK =
        summaries.stream()
            .filter(summary -> summary.prefixTokens() == 4_096)
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException("a 4096-token tier is required"));
    boolean contracts =
        summaries.stream()
            .allMatch(
                summary ->
                    summary.sharedPhysicalContractPassed()
                        && summary.recomputedIndependencePassed()
                        && summary.tokenExactAcrossStrategies()
                        && summary.sharedMedianUniqueStateBytes()
                            < summary.recomputedMedianUniqueStateBytes());
    return new Verdict(
        crossover,
        fourK.improvement(),
        crossover.isPresent() && fourK.improvement() >= MINIMUM_FOUR_K_IMPROVEMENT && contracts);
  }

  private static TierSummary summarize(int prefixTokens, List<Measurement> all) {
    List<Measurement> shared =
        all.stream()
            .filter(measurement -> measurement.prefixTokens() == prefixTokens)
            .filter(measurement -> measurement.strategy().equals(PrefixMode.SHARED.label))
            .toList();
    List<Measurement> recomputed =
        all.stream()
            .filter(measurement -> measurement.prefixTokens() == prefixTokens)
            .filter(measurement -> measurement.strategy().equals(PrefixMode.RECOMPUTED.label))
            .toList();
    double sharedMillis =
        median(shared.stream().map(Measurement::handoffTimeToFirstTokenNanos).toList())
            / 1_000_000.0;
    double recomputedMillis =
        median(recomputed.stream().map(Measurement::handoffTimeToFirstTokenNanos).toList())
            / 1_000_000.0;
    return new TierSummary(
        prefixTokens,
        sharedMillis,
        recomputedMillis,
        1.0 - sharedMillis / recomputedMillis,
        median(shared.stream().map(Measurement::uniqueInferenceStateBytes).toList()),
        median(recomputed.stream().map(Measurement::uniqueInferenceStateBytes).toList()),
        shared.stream().allMatch(Measurement::physicallyShared),
        recomputed.stream().noneMatch(Measurement::physicallyShared),
        tokenSequencesExact(
            java.util.stream.Stream.concat(shared.stream(), recomputed.stream())
                .map(Measurement::outputTokenIds)
                .toList()));
  }

  static boolean tokenSequencesExact(List<List<Integer>> sequences) {
    return !sequences.isEmpty() && sequences.stream().distinct().count() == 1;
  }

  private static Measurement measure(
      ActivatedToolCallingModel model,
      ModelPrompt prompt,
      int prefixTokens,
      PrefixMode mode,
      int trial,
      SamplingOptions options,
      TokenConstraint constraint) {
    ProcessMemory.Snapshot processBefore = ProcessMemory.snapshot(ProcessHandle.current().pid());
    JvmMemorySnapshot jvmBefore = JvmMemorySnapshot.capture();
    long started = System.nanoTime();
    try (ActivatedToolTurn turn = model.openToolTurn(prompt, mode.strategy)) {
      long opened = System.nanoTime();
      RecordingConstraint recording = new RecordingConstraint(constraint);
      String output = turn.generateToolCall(options, recording);
      long completed = System.nanoTime();
      GenerationMetrics metrics = turn.toolMetrics();
      long generatedTtft =
          metrics
              .timeToFirstToken()
              .orElseThrow(() -> new IllegalStateException("tool branch produced no first token"))
              .toNanos();
      long baseBytes = turn.baseInferenceStateBytes().orElse(0);
      long toolBytes = turn.toolInferenceStateBytes().orElse(0);
      long uniqueBytes = turn.uniqueInferenceStateBytes().orElse(0);
      Measurement measurement =
          new Measurement(
              prefixTokens,
              mode.label,
              trial,
              opened - started,
              generatedTtft,
              Math.addExact(opened - started, generatedTtft),
              completed - started,
              turn.sharedPrefixTokens(),
              turn.sharedPrefixBytes(),
              baseBytes,
              toolBytes,
              uniqueBytes,
              turn.physicallySharesPrefix(),
              output,
              recording.acceptedTokens(),
              processBefore,
              ProcessMemory.snapshot(ProcessHandle.current().pid()),
              jvmBefore,
              JvmMemorySnapshot.capture());
      if (trial >= 0) {
        System.out.printf(
            "%4d %-10s trial=%d handoff-ttft=%.1f ms unique-state=%d shared=%s%n",
            prefixTokens,
            mode.label,
            trial + 1,
            measurement.handoffTimeToFirstTokenNanos() / 1_000_000.0,
            measurement.uniqueInferenceStateBytes(),
            measurement.physicallyShared());
      }
      return measurement;
    }
  }

  private static ModelPrompt promptWithExactPrefix(
      Tokenizer tokenizer,
      ActivatedAdapterMetadata adapter,
      int targetPrefixTokens,
      String template) {
    if ("granite-documents".equals(template)) {
      return graniteDocumentsPromptWithExactPrefix(tokenizer, adapter, targetPrefixTokens);
    }
    String leading = "<|im_start|>system\nYou are a tool selector.<|im_end|>\n<|im_start|>user\n";
    String trailing = "<|im_end|>\n<|im_start|>assistant\n";
    String invocationText = adapter.invocationText();
    String unit =
        List.of(" context", " transit", " data", " x").stream()
            .filter(candidate -> tokenizer.encode(candidate).length == 1)
            .findFirst()
            .orElseThrow(() -> new IllegalStateException("no stable one-token filler was found"));
    int low = 0;
    int high = targetPrefixTokens * 2;
    while (low <= high) {
      int count = (low + high) >>> 1;
      ModelPrompt.Builder prompt =
          ModelPrompt.builder().control(leading).text(unit.repeat(count)).control(trailing);
      if (!invocationText.isEmpty()) {
        prompt.control(invocationText);
      }
      int prefix = activationBoundary(tokenizer.encode(prompt.build()), adapter);
      if (prefix == targetPrefixTokens) {
        return prompt.build();
      }
      if (prefix < targetPrefixTokens) {
        low = count + 1;
      } else {
        high = count - 1;
      }
    }
    throw new IllegalStateException(
        "could not construct an exact " + targetPrefixTokens + "-token activation prefix");
  }

  /**
   * The Granite documents shape: the filler is a document body, the question is a user turn, and
   * the adapter's own marker closes the prompt. Rendering goes through the shared renderer so the
   * measured prefix is the same prompt shape gates 4 and 5 use.
   */
  private static ModelPrompt graniteDocumentsPromptWithExactPrefix(
      Tokenizer tokenizer, ActivatedAdapterMetadata adapter, int targetPrefixTokens) {
    if (!GraniteDocumentsPrompt.ASSISTANT_MARKER.equals(adapter.invocationText())) {
      throw new IllegalArgumentException(
          "granite-documents template requires the assistant marker as the adapter invocation");
    }
    String unit =
        List.of(" transit", " context", " data", " x").stream()
            .filter(candidate -> tokenizer.encode(candidate).length == 1)
            .findFirst()
            .orElseThrow(() -> new IllegalStateException("no stable one-token filler was found"));
    int low = 0;
    int high = targetPrefixTokens * 2;
    while (low <= high) {
      int count = (low + high) >>> 1;
      ModelPrompt.Builder builder =
          GraniteDocumentsPrompt.appendSystem(
              ModelPrompt.builder(), List.of("Background records:" + unit.repeat(count)), null);
      GraniteDocumentsPrompt.appendTurn(
          builder, "user", "Is the archive code for route Blue Line 17 in the documents?");
      ModelPrompt candidate = GraniteDocumentsPrompt.finish(builder);
      int prefix = activationBoundary(tokenizer.encode(candidate), adapter);
      if (prefix == targetPrefixTokens) {
        return candidate;
      }
      if (prefix < targetPrefixTokens) {
        low = count + 1;
      } else {
        high = count - 1;
      }
    }
    throw new IllegalStateException(
        "could not construct an exact " + targetPrefixTokens + "-token documents prefix");
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
    throw new IllegalStateException("generated benchmark prompt omitted the adapter invocation");
  }

  private static TokenConstraint nonTerminal(Tokenizer tokenizer) {
    return new TokenConstraint() {
      @Override
      public boolean allows(int token) {
        return !tokenizer.isEndOfGeneration(token);
      }

      @Override
      public void accept(int token) {}
    };
  }

  private static final class RecordingConstraint implements TokenConstraint {
    private final TokenConstraint delegate;
    private final List<Integer> acceptedTokens = new ArrayList<>();

    private RecordingConstraint(TokenConstraint delegate) {
      this.delegate = delegate;
    }

    @Override
    public boolean allows(int token) {
      return delegate.allows(token);
    }

    @Override
    public void accept(int token) {
      delegate.accept(token);
      acceptedTokens.add(token);
    }

    @Override
    public boolean isComplete() {
      return delegate.isComplete();
    }

    private List<Integer> acceptedTokens() {
      return List.copyOf(acceptedTokens);
    }
  }

  private static long median(List<Long> values) {
    if (values.isEmpty()) {
      throw new IllegalArgumentException("cannot summarize an empty measurement set");
    }
    List<Long> sorted = values.stream().sorted(Comparator.naturalOrder()).toList();
    int middle = sorted.size() / 2;
    if ((sorted.size() & 1) == 1) {
      return sorted.get(middle);
    }
    return (sorted.get(middle - 1) + sorted.get(middle)) / 2;
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
    try (var paths = Files.walk(directory)) {
      return paths
          .filter(Files::isRegularFile)
          .mapToLong(ActivatedPrefixSharingBenchmarkCli::size)
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
    ObjectMapper mapper = reportMapper();
    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.writeString(temporary, mapper.writeValueAsString(report) + System.lineSeparator());
    try {
      Files.move(
          temporary, output, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    } catch (AtomicMoveNotSupportedException unsupported) {
      Files.move(temporary, output, StandardCopyOption.REPLACE_EXISTING);
    }
  }

  static ObjectMapper reportMapper() {
    return new ObjectMapper()
        .registerModule(new Jdk8Module())
        .enable(SerializationFeature.INDENT_OUTPUT);
  }

  private enum PrefixMode {
    SHARED("shared", ActivatedToolCallingModel.PrefixStrategy.SHARED),
    RECOMPUTED("recomputed", ActivatedToolCallingModel.PrefixStrategy.RECOMPUTED);

    private final String label;
    private final ActivatedToolCallingModel.PrefixStrategy strategy;

    PrefixMode(String label, ActivatedToolCallingModel.PrefixStrategy strategy) {
      this.label = label;
      this.strategy = strategy;
    }
  }
}
