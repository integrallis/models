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
  private final long backendPid;
  private double loadMillis;

  HttpBenchmarkTarget(
      String backend,
      String model,
      URI endpoint,
      int contextLength,
      int threads,
      long backendPid,
      double loadMillis) {
    this.backend = backend;
    this.model = model;
    this.endpoint = endpoint;
    this.contextLength = contextLength;
    this.threads = threads;
    this.backendPid = backendPid;
    this.loadMillis = loadMillis;
    this.client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build();
    this.mapper = new ObjectMapper();
  }

  @Override
  public TrialMeasurement generate(String prompt, int maxTokens) {
    try {
      ProcessMetrics.Snapshot processBefore = ProcessMetrics.capture(backendPid);
      ObjectNode requestBody = requestBody(prompt, maxTokens);
      HttpRequest request =
          HttpRequest.newBuilder(requestUri())
              .timeout(Duration.ofMinutes(30))
              .header("Content-Type", "application/json")
              .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(requestBody)))
              .build();
      long start = System.nanoTime();
      HttpResponse<Stream<String>> response =
          client.send(request, HttpResponse.BodyHandlers.ofLines());
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
      ProcessMetrics.Snapshot processAfter = ProcessMetrics.capture(backendPid);
      return TrialMeasurement.success(
          parsed.firstTokenMillis(),
          totalMillis,
          parsed.prefillTokensPerSecond(),
          parsed.inputTokens(),
          parsed.outputTokens(),
          processAfter.highWaterBytes(),
          processAfter.cpuMillisSince(processBefore),
          Hashing.sha256(parsed.text()));
    } catch (IOException failure) {
      return TrialMeasurement.failure(failure.toString());
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      return TrialMeasurement.failure(interrupted.toString());
    } catch (RuntimeException failure) {
      return TrialMeasurement.failure(failure.toString());
    }
  }

  @Override
  public double loadMillis() {
    return loadMillis;
  }

  @Override
  public void close() {
    client.close();
  }

  private URI requestUri() {
    return endpoint.resolve("ollama".equals(backend) ? "/api/generate" : "/completion");
  }

  ObjectNode requestBody(String prompt, int maxTokens) {
    ObjectNode body = mapper.createObjectNode();
    body.put("prompt", prompt);
    body.put("stream", true);
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
