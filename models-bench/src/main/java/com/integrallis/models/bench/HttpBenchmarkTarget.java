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
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.stream.Stream;

/** Common streaming HTTP benchmark target for Ollama and llama.cpp server. */
final class HttpBenchmarkTarget implements BenchmarkTarget {

  private final String backend;
  private final String model;
  private final URI endpoint;
  private final HttpClient client;
  private final ObjectMapper mapper;
  private final int contextLength;
  private final int threads;
  private final BackendIdentity identity;
  private double loadMillis;

  private HttpBenchmarkTarget(
      String backend,
      String model,
      URI endpoint,
      HttpClient client,
      ObjectMapper mapper,
      int contextLength,
      int threads,
      BackendIdentity identity) {
    this.backend = backend;
    this.model = model;
    this.endpoint = endpoint;
    this.client = client;
    this.mapper = mapper;
    this.contextLength = contextLength;
    this.threads = threads;
    this.identity = identity;
  }

  static HttpBenchmarkTarget connect(
      String backend,
      String model,
      URI endpoint,
      BenchmarkArtifact artifact,
      int contextLength,
      int threads)
      throws IOException {
    HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build();
    ObjectMapper mapper = new ObjectMapper();
    try {
      BackendIdentity identity =
          "ollama".equals(backend)
              ? probeOllama(client, mapper, endpoint, model, artifact)
              : probeLlamaCpp(client, mapper, endpoint, artifact, contextLength);
      return new HttpBenchmarkTarget(
          backend, model, endpoint, client, mapper, contextLength, threads, identity);
    } catch (InterruptedException interrupted) {
      client.close();
      Thread.currentThread().interrupt();
      throw new IOException("interrupted while probing " + backend, interrupted);
    } catch (IOException | RuntimeException failure) {
      client.close();
      throw failure;
    }
  }

  @Override
  public TrialMeasurement generate(String prompt, int maxTokens) {
    try {
      ObjectNode requestBody =
          requestBody(mapper, backend, model, prompt, maxTokens, contextLength, threads);
      HttpRequest request =
          HttpRequest.newBuilder(requestUri())
              .timeout(Duration.ofMinutes(30))
              .header("Content-Type", "application/json")
              .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(requestBody)))
              .build();
      long start = System.nanoTime();
      HttpResponse<Stream<String>> response = client.send(request, HttpResponse.BodyHandlers.ofLines());
      if (response.statusCode() < 200 || response.statusCode() >= 300) {
        return TrialMeasurement.failure("HTTP " + response.statusCode() + " from " + requestUri());
      }

      ParsedGeneration parsed;
      try (Stream<String> lines = response.body()) {
        if ("ollama".equals(backend)) {
          OllamaStreamParser parser = new OllamaStreamParser(mapper);
          lines.forEachOrdered(line -> acceptOllama(parser, line, start));
          parsed = parser.result();
        } else {
          LlamaCppStreamParser parser = new LlamaCppStreamParser(mapper);
          lines.forEachOrdered(line -> acceptLlamaCpp(parser, line, start));
          parsed = parser.result();
        }
      }
      double totalMillis = (System.nanoTime() - start) / 1_000_000.0;
      if (parsed.loadMillis() > 0 && loadMillis == 0) {
        loadMillis = parsed.loadMillis();
      }
      return TrialMeasurement.success(
          parsed.firstTokenMillis(),
          totalMillis,
          parsed.prefillTokensPerSecond(),
          parsed.decodeTokensPerSecond(),
          parsed.inputTokens(),
          parsed.outputTokens(),
          0,
          0,
          Hashing.sha256(parsed.text()));
    } catch (IOException failure) {
      return TrialMeasurement.failure(failure.toString());
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      return TrialMeasurement.failure(interrupted.toString());
    }
  }

  @Override
  public LoadMeasurement loadMeasurement() {
    return "ollama".equals(backend)
        ? LoadMeasurement.backendReported(loadMillis)
        : LoadMeasurement.notMeasured();
  }

  @Override
  public BackendIdentity identity() {
    return identity;
  }

  @Override
  public void close() {
    client.close();
  }

  private URI requestUri() {
    return endpoint.resolve("ollama".equals(backend) ? "/api/generate" : "/completion");
  }

  static ObjectNode requestBody(
      ObjectMapper mapper,
      String backend,
      String model,
      String prompt,
      int maxTokens,
      int contextLength,
      int threads) {
    ObjectNode body = mapper.createObjectNode();
    body.put("prompt", prompt);
    body.put("stream", true);
    if ("ollama".equals(backend)) {
      body.put("model", model);
      body.put("keep_alive", "5m");
      body.put("raw", true);
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
      body.put("cache_prompt", false);
      body.put("repeat_penalty", 1);
    }
    return body;
  }

  private static BackendIdentity probeOllama(
      HttpClient client,
      ObjectMapper mapper,
      URI endpoint,
      String model,
      BenchmarkArtifact artifact)
      throws IOException, InterruptedException {
    ObjectNode showBody = mapper.createObjectNode().put("model", model);
    JsonResponse show =
        send(
            client,
            HttpRequest.newBuilder(endpoint.resolve("/api/show"))
                .timeout(Duration.ofSeconds(30))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(showBody)))
                .build());
    JsonResponse version =
        send(
            client,
            HttpRequest.newBuilder(endpoint.resolve("/api/version"))
                .timeout(Duration.ofSeconds(30))
                .GET()
                .build());
    return BackendIdentityProbe.fromOllama(
        model,
        artifact,
        mapper.readTree(show.body()),
        mapper.readTree(version.body()).path("version").asText("unknown"));
  }

  private static BackendIdentity probeLlamaCpp(
      HttpClient client,
      ObjectMapper mapper,
      URI endpoint,
      BenchmarkArtifact artifact,
      int contextLength)
      throws IOException, InterruptedException {
    JsonResponse props =
        send(
            client,
            HttpRequest.newBuilder(endpoint.resolve("/props"))
                .timeout(Duration.ofSeconds(30))
                .GET()
                .build());
    return BackendIdentityProbe.fromLlamaCpp(
        artifact, mapper.readTree(props.body()), contextLength);
  }

  private static JsonResponse send(HttpClient client, HttpRequest request)
      throws IOException, InterruptedException {
    HttpResponse<String> response =
        client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
    if (response.statusCode() < 200 || response.statusCode() >= 300) {
      throw new IOException(
          "HTTP " + response.statusCode() + " while probing " + request.uri());
    }
    return new JsonResponse(response.body());
  }

  private record JsonResponse(String body) {}

  private static void acceptOllama(OllamaStreamParser parser, String line, long start) {
    try {
      parser.accept(line, elapsedMillis(start));
    } catch (IOException failure) {
      throw new StreamParseException(failure);
    }
  }

  private static void acceptLlamaCpp(LlamaCppStreamParser parser, String line, long start) {
    try {
      parser.accept(line, elapsedMillis(start));
    } catch (IOException failure) {
      throw new StreamParseException(failure);
    }
  }

  private static double elapsedMillis(long start) {
    return (System.nanoTime() - start) / 1_000_000.0;
  }

  private static final class StreamParseException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    private StreamParseException(IOException cause) {
      super(cause);
    }
  }
}
