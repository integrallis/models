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
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.models.api.BackendDiagnostics;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;

/** Direct streaming client for the hosted APIs used by the developer-facing RAG comparison. */
public final class HostedGenerationClient implements GenerationClient {
  private final HostedProvider provider;
  private final String model;
  private final URI endpoint;
  private final String apiKey;
  private final HostedApiPricing pricing;
  private final HttpClient httpClient;
  private final ObjectMapper mapper;

  /** Creates a client using the provider-specific API key environment variable. */
  public HostedGenerationClient(String backend, String model, URI endpoint) {
    this(backend, model, endpoint, apiKey(HostedProvider.parse(backend)));
  }

  HostedGenerationClient(String backend, String model, URI endpoint, String apiKey) {
    this.provider = HostedProvider.parse(backend);
    this.model = requireText(model, "model");
    this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
    this.apiKey = requireText(apiKey, provider.apiKeyEnvironment());
    this.pricing = HostedApiPricing.forModel(backend, model);
    this.httpClient = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build();
    this.mapper = new ObjectMapper();
  }

  @Override
  public String backend() {
    return provider.id();
  }

  @Override
  public String model() {
    return model;
  }

  @Override
  public HostedApiPricing hostedApiPricing() {
    return pricing;
  }

  @Override
  public Map<String, String> generationControls() {
    return provider.generationControls();
  }

  @Override
  public BackendDiagnostics diagnostics() {
    Map<String, String> environment = new LinkedHashMap<>();
    environment.put("executionLocation", "remote");
    environment.put("dataEgress", "true");
    environment.put("endpointHost", endpoint.getHost());
    environment.put("apiKeyEnvironment", provider.apiKeyEnvironment());
    environment.put("pricingEffectiveDate", pricing.effectiveDate());
    environment.put("pricingSource", pricing.sourceUrl());
    environment.put(
        "inputUsdPerMillionTokens", Double.toString(pricing.inputUsdPerMillionTokens()));
    environment.put(
        "cacheReadUsdPerMillionTokens", Double.toString(pricing.cacheReadUsdPerMillionTokens()));
    environment.put(
        "outputUsdPerMillionTokens", Double.toString(pricing.outputUsdPerMillionTokens()));
    environment.put("requestControls", provider.generationControls().toString());
    return new BackendDiagnostics(provider.id(), "hosted-api-v1", environment, java.util.List.of());
  }

  @Override
  public GenerationResult generate(String prompt, int maxTokens) {
    try {
      HttpRequest.Builder request =
          HttpRequest.newBuilder(requestUri())
              .timeout(Duration.ofMinutes(10))
              .header("Accept", "text/event-stream")
              .header("Content-Type", "application/json");
      provider.addAuthentication(request, apiKey);
      HttpRequest built =
          request
              .POST(
                  HttpRequest.BodyPublishers.ofString(
                      mapper.writeValueAsString(requestBody(prompt, maxTokens))))
              .build();

      long start = System.nanoTime();
      HttpResponse<Stream<String>> response =
          httpClient.send(built, HttpResponse.BodyHandlers.ofLines());
      if (response.statusCode() < 200 || response.statusCode() >= 300) {
        String responseBody;
        try (Stream<String> lines = response.body()) {
          responseBody = lines.limit(20).reduce("", (left, right) -> left + right);
        }
        throw new IllegalStateException(
            provider.id()
                + " returned HTTP "
                + response.statusCode()
                + ": "
                + sanitizeError(responseBody));
      }

      ParsedHostedGeneration parsed;
      try (Stream<String> lines = response.body()) {
        parsed = parse(lines, start);
      }
      double totalMillis = nanosToMillis(System.nanoTime() - start);
      double estimatedCost =
          pricing.estimateUsd(
              parsed.inputTokens(),
              parsed.cacheReadInputTokens(),
              parsed.cacheWriteInputTokens(),
              parsed.outputTokens());
      return new GenerationResult(
          parsed.text(),
          parsed.inputTokens(),
          parsed.cacheReadInputTokens(),
          parsed.cacheWriteInputTokens(),
          parsed.outputTokens(),
          parsed.firstTextTokenMillis(),
          totalMillis,
          0,
          0,
          0,
          0,
          estimatedCost);
    } catch (IOException failure) {
      throw new IllegalStateException(provider.id() + " generation failed", failure);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(provider.id() + " generation interrupted", interrupted);
    }
  }

  ObjectNode requestBody(String prompt, int maxTokens) {
    requireText(prompt, "prompt");
    if (maxTokens < 1) {
      throw new IllegalArgumentException("maxTokens must be positive");
    }

    ObjectNode body = mapper.createObjectNode();
    body.put("model", model);
    body.put("stream", true);
    ArrayNode messages = body.putArray("messages");
    ObjectNode message = messages.addObject();
    message.put("role", "user");
    message.put("content", prompt);

    switch (provider) {
      case OPENAI -> {
        body.put("max_completion_tokens", maxTokens);
        body.put("reasoning_effort", "none");
        body.putObject("stream_options").put("include_usage", true);
      }
      case DEEPSEEK -> {
        body.put("max_tokens", maxTokens);
        body.put("temperature", 0);
        body.putObject("thinking").put("type", "disabled");
        body.putObject("stream_options").put("include_usage", true);
      }
      case ANTHROPIC -> {
        body.put("max_tokens", maxTokens);
        body.put("temperature", 0);
      }
    }
    return body;
  }

  @Override
  public void close() {
    httpClient.close();
  }

  private ParsedHostedGeneration parse(Stream<String> lines, long start) throws IOException {
    if (provider == HostedProvider.ANTHROPIC) {
      AnthropicGenerationParser parser = new AnthropicGenerationParser(mapper);
      for (String line : (Iterable<String>) lines::iterator) {
        parser.accept(line, nanosToMillis(System.nanoTime() - start));
      }
      return parser.result();
    }
    OpenAiChatGenerationParser parser = new OpenAiChatGenerationParser(mapper);
    for (String line : (Iterable<String>) lines::iterator) {
      parser.accept(line, nanosToMillis(System.nanoTime() - start));
    }
    return parser.result();
  }

  private URI requestUri() {
    String base = endpoint.toString();
    if (!base.endsWith("/")) {
      base += "/";
    }
    return URI.create(base)
        .resolve(provider == HostedProvider.ANTHROPIC ? "messages" : "chat/completions");
  }

  private static String apiKey(HostedProvider provider) {
    String value = System.getenv(provider.apiKeyEnvironment());
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException(
          provider.apiKeyEnvironment() + " is required for " + provider.id());
    }
    return value;
  }

  private static String sanitizeError(String error) {
    return error.length() > 2_000 ? error.substring(0, 2_000) : error;
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

  private enum HostedProvider {
    OPENAI("openai", "OPENAI_API_KEY", Map.of("reasoningEffort", "none")),
    ANTHROPIC("anthropic", "ANTHROPIC_API_KEY", Map.of("temperature", "0")),
    DEEPSEEK("deepseek", "DEEPSEEK_API_KEY", Map.of("thinking", "disabled", "temperature", "0"));

    private final String id;
    private final String apiKeyEnvironment;
    private final Map<String, String> generationControls;

    HostedProvider(String id, String apiKeyEnvironment, Map<String, String> generationControls) {
      this.id = id;
      this.apiKeyEnvironment = apiKeyEnvironment;
      this.generationControls = Map.copyOf(generationControls);
    }

    private static HostedProvider parse(String value) {
      for (HostedProvider provider : values()) {
        if (provider.id.equals(value)) {
          return provider;
        }
      }
      throw new IllegalArgumentException("hosted backend must be openai, anthropic, or deepseek");
    }

    private void addAuthentication(HttpRequest.Builder request, String key) {
      if (this == ANTHROPIC) {
        request.header("x-api-key", key).header("anthropic-version", "2023-06-01");
      } else {
        request.header("Authorization", "Bearer " + key);
      }
    }

    private String id() {
      return id;
    }

    private String apiKeyEnvironment() {
      return apiKeyEnvironment;
    }

    private Map<String, String> generationControls() {
      return generationControls;
    }
  }
}
