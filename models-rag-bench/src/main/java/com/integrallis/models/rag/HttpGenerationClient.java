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
package com.integrallis.models.rag;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.nativellm.LocalEngineGeneration;
import com.integrallis.models.backend.nativellm.LocalEngineGenerationModel;
import com.integrallis.models.backend.nativellm.PromptMode;
import java.net.URI;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/** Instrumented RAG adapter around the published local-engine Models backend. */
public final class HttpGenerationClient implements GenerationClient {
  private static final Set<String> BACKENDS = Set.of("ollama", "llama.cpp");

  private final String backend;
  private final String model;
  private final long backendPid;
  private final PromptMode promptMode;
  private final LocalEngineGenerationModel generationModel;
  private double observedLoadMillis;

  public HttpGenerationClient(
      String backend, String model, URI endpoint, int contextLength, int threads, long backendPid) {
    this(backend, model, endpoint, contextLength, threads, backendPid, PromptMode.RAW);
  }

  public HttpGenerationClient(
      String backend,
      String model,
      URI endpoint,
      int contextLength,
      int threads,
      long backendPid,
      PromptMode promptMode) {
    if (!BACKENDS.contains(backend)) {
      throw new IllegalArgumentException("backend must be one of " + BACKENDS);
    }
    if (contextLength < 1 || threads < 1 || backendPid < 0) {
      throw new IllegalArgumentException(
          "contextLength/threads must be positive and pid non-negative");
    }
    this.backend = backend;
    this.model = requireText(model, "model");
    this.backendPid = backendPid;
    this.promptMode = java.util.Objects.requireNonNull(promptMode, "promptMode");
    LocalEngineGenerationModel.Builder builder =
        "ollama".equals(backend)
            ? LocalEngineGenerationModel.ollama(model, endpoint)
            : LocalEngineGenerationModel.llamaCpp(model, endpoint);
    generationModel =
        builder
            .promptMode(promptMode)
            .contextLength(contextLength)
            .threads(threads)
            .requestTimeout(Duration.ofMinutes(30))
            .build();
  }

  @Override
  public String backend() {
    return backend;
  }

  @Override
  public String model() {
    return model;
  }

  @Override
  public BackendDiagnostics diagnostics() {
    return generationModel.diagnostics();
  }

  @Override
  public Map<String, String> generationControls() {
    Map<String, String> common =
        Map.of(
            "temperature", "0",
            "topK", "1",
            "topP", "1",
            "seed", "42",
            "repetitionPenalty", "1");
    LinkedHashMap<String, String> controls = new LinkedHashMap<>(common);
    if (promptMode == PromptMode.NATIVE_CHAT) {
      controls.put("promptTemplate", "model-owned");
    } else if ("llama.cpp".equals(backend)) {
      controls.put("promptCache", "false");
    }
    return Map.copyOf(controls);
  }

  @Override
  public GenerationResult generate(String prompt, int maxTokens) {
    SamplingOptions options =
        SamplingOptions.builder()
            .temperature(0)
            .topK(1)
            .topP(1)
            .maxTokens(maxTokens)
            .seed(42)
            .repetitionPenalty(1)
            .build();
    Duration cpuBefore = ProcessResourceProbe.cpuDuration(backendPid);
    LocalEngineGeneration observed = generationModel.generateObserved(prompt, options);
    if (observed.loadMillis() > 0 && observedLoadMillis == 0) {
      observedLoadMillis = observed.loadMillis();
    }
    Duration cpuAfter = ProcessResourceProbe.cpuDuration(backendPid);
    double cpuMillis = Math.max(0, nanosToMillis(cpuAfter.minus(cpuBefore).toNanos()));
    return new GenerationResult(
        observed.text(),
        observed.inputTokens(),
        observed.outputTokens(),
        observed.firstTokenMillis(),
        observed.totalMillis(),
        observed.prefillTokensPerSecond(),
        observedLoadMillis,
        ProcessResourceProbe.highWaterBytes(backendPid),
        cpuMillis);
  }

  @Override
  public void close() {
    generationModel.close();
  }

  private static String requireText(String value, String name) {
    java.util.Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value;
  }

  private static double nanosToMillis(long nanos) {
    return nanos / 1_000_000.0;
  }
}
