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
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.tornado.TornadoBackend;
import com.integrallis.models.backend.tornado.TornadoBackendOptions;
import com.integrallis.models.backend.tornado.TornadoBackendRuntime;
import com.integrallis.models.backend.tornado.TornadoBackendStatus;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Runs one prefill and one greedy decode of a GGUF model through the TornadoVM backend and reports
 * the evidence a hardware gate needs.
 *
 * <p>The report is deliberately shaped like the other gates: it records the selected device and the
 * fallback reason if any, the readiness cost the eager plan compilation paid, prefill and decode
 * throughput, and — the point of this command — the number of projections that actually reached the
 * device <b>per GGUF weight format</b>. A Q4_K_M model whose {@code Q6_K} count is zero routed only
 * its Q4_K tensors, which looks like a working accelerator on the throughput line alone.
 *
 * <p>Off a qualified GPU this command still runs: the Tornado backend falls back to the Vector API
 * and the report says so, with {@code accelerated=false} and the reason. Pass {@code --require
 * true} to make an unavailable accelerator a hard failure instead.
 */
final class AcceleratorProfileCli {

  private static final Set<String> OPTIONS =
      Set.of(
          "model",
          "prompt",
          "prompt-file",
          "tokens",
          "warmups",
          "context",
          "batch",
          "decode",
          "eager",
          "require",
          "output");
  private static final String DEFAULT_PROMPT =
      "Explain why a GPU projection kernel must be checked against the CPU kernel it replaces "
          + "before any throughput number from it is worth reporting.";

  private AcceleratorProfileCli() {}

  static void run(String[] args) throws IOException {
    Configuration configuration = parse(args);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY,
        Integer.toString(configuration.contextLength()));
    TornadoBackendOptions options =
        new TornadoBackendOptions(
            configuration.decode(),
            configuration.eager(),
            configuration.require(),
            configuration.batch());
    try (TornadoBackendRuntime runtime = TornadoBackend.open(configuration.model(), options)) {
      Report report = profile(runtime, configuration);
      write(report, configuration.output());
      print(report, configuration.output());
    }
  }

  static Configuration parse(String[] args) throws IOException {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    String model = values.get("model");
    if (model == null || model.isBlank()) {
      throw new IllegalArgumentException("--model must be a GGUF file path");
    }
    Path modelPath = Path.of(model).toAbsolutePath().normalize();
    if (!Files.isRegularFile(modelPath)) {
      throw new IllegalArgumentException("--model must be a regular file: " + modelPath);
    }
    String prompt =
        values.containsKey("prompt-file")
            ? Files.readString(Path.of(values.get("prompt-file")))
            : values.getOrDefault("prompt", DEFAULT_PROMPT);
    return new Configuration(
        modelPath,
        prompt,
        BenchmarkCliArguments.integer(values, "tokens", 64),
        BenchmarkCliArguments.integer(values, "warmups", 1),
        BenchmarkCliArguments.integer(values, "context", 2_048),
        BenchmarkCliArguments.integer(values, "batch", 32),
        flag(values, "decode", true),
        flag(values, "eager", true),
        flag(values, "require", false),
        Path.of(values.getOrDefault("output", "build/reports/inference/accelerator-profile.json")));
  }

  static Report profile(TornadoBackendRuntime runtime, Configuration configuration) {
    PureJavaBackend backend = runtime.backend();
    Tokenizer tokenizer = backend.tokenizer();
    int[] promptTokens = tokenizer.encode(configuration.prompt());
    if (promptTokens.length == 0) {
      throw new IllegalArgumentException("prompt produced no tokens");
    }
    if (promptTokens.length + configuration.tokens() > configuration.contextLength()) {
      throw new IllegalArgumentException(
          "context length "
              + configuration.contextLength()
              + " cannot hold "
              + promptTokens.length
              + " prompt tokens plus "
              + configuration.tokens()
              + " generated tokens");
    }

    for (int warmup = 0; warmup < configuration.warmups(); warmup++) {
      backend.reset();
      float[] logits = backend.prefill(promptTokens, 0);
      generate(backend, logits, promptTokens.length, Math.min(4, configuration.tokens()));
    }

    backend.reset();
    long prefillStart = System.nanoTime();
    float[] logits = backend.prefill(promptTokens, 0);
    long prefillNanos = System.nanoTime() - prefillStart;

    long decodeStart = System.nanoTime();
    int generated = generate(backend, logits, promptTokens.length, configuration.tokens());
    long decodeNanos = System.nanoTime() - decodeStart;

    TornadoBackendStatus status = runtime.status();
    return new Report(
        Instant.now().toString(),
        configuration.model().toString(),
        status.accelerated(),
        status.device(),
        status.reason(),
        status.requiredDeviceBytes(),
        status.readinessTime().toMillis(),
        runtime.projectionPlanCount(),
        new LinkedHashMap<>(runtime.routedProjectionsByFormat()),
        promptTokens.length,
        generated,
        perSecond(promptTokens.length, prefillNanos),
        perSecond(generated, decodeNanos),
        prefillNanos / 1_000_000.0,
        decodeNanos / 1_000_000.0);
  }

  private static int generate(PureJavaBackend backend, float[] logits, int position, int tokens) {
    float[] current = logits;
    int generated = 0;
    for (int index = 0; index < tokens; index++) {
      int next = argmax(current);
      generated++;
      if (index + 1 < tokens) {
        current = backend.forward(next, position + index);
      }
    }
    return generated;
  }

  private static int argmax(float[] logits) {
    int best = 0;
    for (int index = 1; index < logits.length; index++) {
      if (logits[index] > logits[best]) {
        best = index;
      }
    }
    return best;
  }

  private static double perSecond(int count, long nanos) {
    return nanos <= 0 ? 0.0 : count * 1_000_000_000.0 / nanos;
  }

  private static void write(Report report, Path output) throws IOException {
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    new ObjectMapper()
        .enable(SerializationFeature.INDENT_OUTPUT)
        .writeValue(output.toFile(), report);
  }

  private static void print(Report report, Path output) {
    System.out.printf(
        "accelerator profile: accelerated=%s device=%s reason=%s readiness=%d ms plans=%d%n"
            + "  prompt=%d tokens prefill=%.2f tok/s (%.1f ms) "
            + "decode=%d tokens %.2f tok/s (%.1f ms)%n"
            + "  routed projections by format: %s%nreport: %s%n",
        report.accelerated(),
        report.device(),
        report.reason(),
        report.readinessMillis(),
        report.projectionPlans(),
        report.promptTokens(),
        report.prefillTokensPerSecond(),
        report.prefillMillis(),
        report.generatedTokens(),
        report.decodeTokensPerSecond(),
        report.decodeMillis(),
        report.routedProjectionsByFormat().isEmpty()
            ? "none (nothing reached the device)"
            : report.routedProjectionsByFormat(),
        output.toAbsolutePath());
  }

  private static boolean flag(Map<String, String> values, String name, boolean defaultValue) {
    String value = values.get(name);
    if (value == null) {
      return defaultValue;
    }
    if ("true".equalsIgnoreCase(value) || "false".equalsIgnoreCase(value)) {
      return Boolean.parseBoolean(value);
    }
    throw new IllegalArgumentException("--" + name + " must be true or false: " + value);
  }

  record Configuration(
      Path model,
      String prompt,
      int tokens,
      int warmups,
      int contextLength,
      int batch,
      boolean decode,
      boolean eager,
      boolean require,
      Path output) {}

  record Report(
      String timestamp,
      String model,
      boolean accelerated,
      String device,
      String reason,
      long requiredBytes,
      long readinessMillis,
      int projectionPlans,
      Map<String, Long> routedProjectionsByFormat,
      int promptTokens,
      int generatedTokens,
      double prefillTokensPerSecond,
      double decodeTokensPerSecond,
      double prefillMillis,
      double decodeMillis) {}
}
