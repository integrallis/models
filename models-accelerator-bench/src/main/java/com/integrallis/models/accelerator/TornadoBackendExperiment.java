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
import com.integrallis.models.backend.tornado.TornadoBackend;
import com.integrallis.models.backend.tornado.TornadoBackendOptions;
import com.integrallis.models.backend.tornado.TornadoBackendRuntime;
import com.integrallis.models.runtime.GenerationLoop;
import com.integrallis.models.runtime.GenerationMetrics;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/** End-to-end release gate for the public automatic accelerator loader and CPU parity. */
public final class TornadoBackendExperiment {
  private static final ModelPrompt PROMPT =
      ChatTemplate.CHATML_NO_THINK.render(
          List.of(
              ChatMessage.system("Follow the user's output format exactly."),
              ChatMessage.user("Reply with exactly: JAVA")));
  private static final SamplingOptions SAMPLING =
      SamplingOptions.builder().temperature(0.0f).maxTokens(4).build();

  private TornadoBackendExperiment() {}

  public static void main(String[] args) {
    if (args.length != 1) {
      throw new IllegalArgumentException("usage: TornadoBackendExperiment <model.gguf>");
    }
    Path model = Path.of(args[0]).toAbsolutePath().normalize();
    if (!Files.isRegularFile(model)) {
      throw new IllegalArgumentException("model is not a regular file: " + model);
    }
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "512");
    String expected;
    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      expected = generate("Vector API", backend);
    }
    TornadoBackendOptions required = new TornadoBackendOptions(true, true, true, 32);
    try (TornadoBackendRuntime runtime = TornadoBackend.open(model, required)) {
      String actual = generate("Automatic accelerator", runtime.backend());
      if (!expected.equals(actual)) {
        throw new IllegalStateException(
            "automatic accelerator changed generated text: " + expected + " != " + actual);
      }
      System.out.printf(
          Locale.ROOT,
          "selected=%s device=%s readiness=%.3f s reason=%s%n",
          runtime.status().accelerated(),
          runtime.status().device(),
          runtime.status().readinessTime().toNanos() / 1_000_000_000.0,
          runtime.status().reason());
    }
    try (PureJavaBackend backend = PureJavaBackend.loadAutomatic(model)) {
      String actual = generate("Service-loaded accelerator", backend);
      if (!expected.equals(actual)) {
        throw new IllegalStateException(
            "service-loaded accelerator changed generated text: " + expected + " != " + actual);
      }
      String implementation =
          backend
              .diagnostics()
              .optimization("grouped-projections")
              .orElseThrow()
              .settings()
              .get("implementation");
      if (!"tornadovm-java-q4-prefill-decode".equals(implementation)) {
        throw new IllegalStateException("automatic provider was not selected: " + implementation);
      }
    }
  }

  private static String generate(String label, PureJavaBackend backend) {
    GenerationLoop loop = new GenerationLoop(backend);
    String output = loop.generate(PROMPT, SAMPLING);
    GenerationMetrics metrics = loop.lastGenerationMetrics();
    System.out.printf(
        Locale.ROOT,
        "%s output=%s promptTokens=%d completionTokens=%d prefill=%.3f s total=%.3f s%n",
        label,
        output.replace('\n', ' '),
        metrics.usage().promptTokens(),
        metrics.usage().completionTokens(),
        metrics.prefill().toNanos() / 1_000_000_000.0,
        metrics.total().toNanos() / 1_000_000_000.0);
    return output;
  }
}
