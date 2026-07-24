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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Stream;

/** Benchmark-only streaming client for Ollama and llama.cpp using an identical raw prompt. */
public final class HttpGenerationClient implements GenerationClient {
  private static final Set<String> BACKENDS = Set.of("ollama", "llama.cpp");

  private final String backend;
  private final String model;
  private final URI endpoint;
  private final int contextLength;
  private final int threads;
  private final long backendPid;
  private final HttpClient httpClient;
  private final ObjectMapper mapper;
  private double observedLoadMillis;

  public HttpGenerationClient(
      String backend, String model, URI endpoint, int contextLength, int threads, long backendPid) {
    if (!BACKENDS.contains(backend)) {
      throw new IllegalArgumentException("backend must be one of " + BACKENDS);
    }
    this.backend = backend;
    this.model = requireText(model, "model");
    this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
    if (contextLength < 1 || threads < 1 || backendPid < 0) {
      throw new IllegalArgumentException(
          "contextLength/threads must be positive and pid non-negative");
    }
    this.contextLength = contextLength;
    this.threads = threads;
    this.backendPid = backendPid;
    this.httpClient = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build();
    this.mapper = new ObjectMapper();
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
  public Map<String, String> generationControls() {
    return "llama.cpp".equals(backend)
        ? Map.of(
            "temperature", "0",
            "topK", "1",
            "topP", "1",
            "seed", "42",
            "repetitionPenalty", "1",
            "promptCache", "false")
        : Map.of(
            "temperature", "0",
            "topK", "1",
            "topP", "1",
            "seed", "42",
            "repetitionPenalty", "1",
            "rawPrompt", "true");
  }

  @Override
  public GenerationResult generate(String prompt, int maxTokens) {
    try {
      Duration cpuBefore = ProcessResourceProbe.cpuDuration(backendPid);
      HttpRequest request =
          HttpRequest.newBuilder(requestUri())
              .timeout(Duration.ofMinutes(30))
              .header("Content-Type", "application/json")
              .POST(
                  HttpRequest.BodyPublishers.ofString(
                      mapper.writeValueAsString(requestBody(prompt, maxTokens))))
              .build();
      long start = System.nanoTime();
      HttpResponse<Stream<String>> response =
          withSingleTransportRetry(
              () -> httpClient.send(request, HttpResponse.BodyHandlers.ofLines()));
      if (response.statusCode() < 200 || response.statusCode() >= 300) {
        throw new IllegalStateException("HTTP " + response.statusCode() + " from " + requestUri());
      }

      ParsedStreamingGeneration parsed;
      try (Stream<String> lines = response.body()) {
        parsed = parse(lines, start);
      }
      double totalMillis = nanosToMillis(System.nanoTime() - start);
      if (parsed.loadMillis() > 0 && observedLoadMillis == 0) {
        observedLoadMillis = parsed.loadMillis();
      }
      Duration cpuAfter = ProcessResourceProbe.cpuDuration(backendPid);
      double cpuMillis = Math.max(0, nanosToMillis(cpuAfter.minus(cpuBefore).toNanos()));
      return new GenerationResult(
          parsed.text(),
          parsed.inputTokens(),
          parsed.outputTokens(),
          parsed.firstTokenMillis(),
          totalMillis,
          parsed.prefillTokensPerSecond(),
          observedLoadMillis,
          ProcessResourceProbe.highWaterBytes(backendPid),
          cpuMillis);
    } catch (IOException failure) {
      throw new IllegalStateException(backend + " generation failed", failure);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(backend + " generation interrupted", interrupted);
    }
  }

  ObjectNode requestBody(String prompt, int maxTokens) {
    ObjectNode body = mapper.createObjectNode();
    body.put("prompt", requireText(prompt, "prompt"));
    body.put("stream", true);
    if (maxTokens < 1) {
      throw new IllegalArgumentException("maxTokens must be positive");
    }
    if ("ollama".equals(backend)) {
      body.put("model", model);
      body.put("raw", true);
      body.put("keep_alive", "5m");
      ObjectNode options = body.putObject("options");
      options.put("temperature", 0);
      options.put("top_k", 1);
      options.put("top_p", 1);
      options.put("seed", 42);
      options.put("num_predict", maxTokens);
      options.put("num_ctx", contextLength);
      options.put("num_thread", threads);
      options.put("repeat_penalty", 1);
    } else {
      body.put("n_predict", maxTokens);
      body.put("temperature", 0);
      body.put("top_k", 1);
      body.put("top_p", 1);
      body.put("seed", 42);
      body.put("n_threads", threads);
      body.put("repeat_penalty", 1);
      body.put("cache_prompt", false);
    }
    return body;
  }

  @Override
  public void close() {
    httpClient.close();
  }

  private ParsedStreamingGeneration parse(Stream<String> lines, long start) throws IOException {
    if ("ollama".equals(backend)) {
      OllamaGenerationParser parser = new OllamaGenerationParser(mapper);
      for (String line : (Iterable<String>) lines::iterator) {
        parser.accept(line, nanosToMillis(System.nanoTime() - start));
      }
      return parser.result();
    }
    LlamaCppGenerationParser parser = new LlamaCppGenerationParser(mapper);
    for (String line : (Iterable<String>) lines::iterator) {
      parser.accept(line, nanosToMillis(System.nanoTime() - start));
    }
    return parser.result();
  }

  private URI requestUri() {
    return endpoint.resolve("ollama".equals(backend) ? "/api/generate" : "/completion");
  }

  private static String requireText(String value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value;
  }

  private static double nanosToMillis(long nanos) {
    return nanos / 1_000_000.0;
  }

  static <T> T withSingleTransportRetry(TransportOperation<T> operation)
      throws IOException, InterruptedException {
    try {
      return operation.execute();
    } catch (IOException firstFailure) {
      try {
        return operation.execute();
      } catch (IOException | InterruptedException retryFailure) {
        retryFailure.addSuppressed(firstFailure);
        throw retryFailure;
      }
    }
  }

  @FunctionalInterface
  interface TransportOperation<T> {
    T execute() throws IOException, InterruptedException;
  }
}
