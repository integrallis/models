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
package com.integrallis.models.accelerator;

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.GenerationLoop;
import com.integrallis.models.runtime.GenerationMetrics;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/** Compares a complete Qwen generation on the Vector API and experimental Java/Tornado prefill. */
public final class QwenFullModelExperiment {
  private static final ModelPrompt SHORT_PROMPT =
      ChatTemplate.CHATML_NO_THINK.render(
          List.of(
              ChatMessage.system("Follow the user's output format exactly."),
              ChatMessage.user("Reply with exactly: JAVA")));
  private static final String TRANSIT_CONTEXT =
      "Route Blue serves Central Station, Museum Square, River Market, and Airport Terminal. "
          + "Trains leave Central Station every twelve minutes from 06:00 through 23:00. "
          + "River Market and Airport Terminal have step-free platforms and working elevators. ";
  private static final SamplingOptions SAMPLING =
      SamplingOptions.builder().temperature(0.0f).maxTokens(4).build();

  private QwenFullModelExperiment() {}

  public static void main(String[] args) {
    if (args.length < 1
        || args.length > 3
        || Arrays.stream(args, 1, args.length)
            .anyMatch(
                argument -> !"--cpu-only".equals(argument) && !"--long-prompt".equals(argument))) {
      throw new IllegalArgumentException(
          "usage: QwenFullModelExperiment <model.gguf> [--cpu-only] [--long-prompt]");
    }
    boolean cpuOnly = Arrays.asList(args).contains("--cpu-only");
    boolean longPrompt = Arrays.asList(args).contains("--long-prompt");
    ModelPrompt prompt = longPrompt ? longPrompt() : SHORT_PROMPT;
    Path model = Path.of(args[0]).toAbsolutePath().normalize();
    if (!Files.isRegularFile(model)) {
      throw new IllegalArgumentException("model is not a regular file: " + model);
    }
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "512");

    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      run("Vector API cold", backend, prompt);
      run("Vector API warm", backend, prompt);
    }
    if (cpuOnly) {
      return;
    }

    TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel();
    try (PureJavaBackend backend = PureJavaBackend.load(model, kernel)) {
      run("Tornado cold", backend, prompt);
      run("Tornado warm", backend, prompt);
      System.out.printf(
          Locale.ROOT,
          "accelerated calls=%d plans=%d cumulative=%.3f s%n",
          kernel.calls(),
          kernel.projectionPlanCount(),
          kernel.totalMillis() / 1_000.0);
    }
  }

  private static ModelPrompt longPrompt() {
    return ChatTemplate.CHATML_NO_THINK.render(
        List.of(
            ChatMessage.system("Use the supplied context and follow the output format exactly."),
            ChatMessage.user(
                "Context: "
                    + TRANSIT_CONTEXT.repeat(4)
                    + "Validation instruction: reply with exactly: JAVA")));
  }

  private static void run(String label, PureJavaBackend backend, ModelPrompt prompt) {
    GenerationLoop loop = new GenerationLoop(backend);
    String output = loop.generate(prompt, SAMPLING);
    GenerationMetrics metrics = loop.lastGenerationMetrics();
    System.out.printf(
        Locale.ROOT,
        "%s output=%s promptTokens=%d completionTokens=%d prefill=%.3f s ttft=%.3f s total=%.3f s%n",
        label,
        output.replace('\n', ' '),
        metrics.usage().promptTokens(),
        metrics.usage().completionTokens(),
        seconds(metrics.prefill()),
        seconds(metrics.timeToFirstToken().orElse(Duration.ZERO)),
        seconds(metrics.total()));
  }

  private static double seconds(Duration duration) {
    return duration.toNanos() / 1_000_000_000.0;
  }
}
