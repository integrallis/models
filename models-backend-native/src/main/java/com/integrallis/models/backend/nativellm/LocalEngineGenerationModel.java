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
package com.integrallis.models.backend.nativellm;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;

/** Streaming Java client for a local llama.cpp server or Ollama daemon. */
public final class LocalEngineGenerationModel implements TextGenerationModel {
  private final LocalEngine engine;
  private final String model;
  private final URI endpoint;
  private final PromptMode promptMode;
  private final boolean thinking;
  private final int contextLength;
  private final int threads;
  private final Duration requestTimeout;
  private final HttpClient client;
  private final ObjectMapper mapper = new ObjectMapper();

  private LocalEngineGenerationModel(Builder builder) {
    engine = builder.engine;
    model = builder.model;
    endpoint = builder.endpoint;
    promptMode = builder.promptMode;
    thinking = builder.thinking;
    contextLength = builder.contextLength;
    threads = builder.threads;
    requestTimeout = builder.requestTimeout;
    client = HttpClient.newBuilder().connectTimeout(builder.connectTimeout).build();
  }

  /** Creates a builder targeting llama.cpp's server API. */
  public static Builder llamaCpp(String model, URI endpoint) {
    return new Builder(LocalEngine.LLAMA_CPP, model, endpoint);
  }

  /** Creates a builder targeting Ollama's local API. */
  public static Builder ollama(String model, URI endpoint) {
    return new Builder(LocalEngine.OLLAMA, model, endpoint);
  }

  @Override
  public String modelName() {
    return model;
  }

  @Override
  public BackendDiagnostics diagnostics() {
    return new BackendDiagnostics(
        engine.id(),
        "local-http-v1",
        Map.of(
            "endpoint", endpoint.toString(),
            "promptMode", promptMode.name(),
            "thinking", Boolean.toString(thinking),
            "contextLength", Integer.toString(contextLength),
            "threads", Integer.toString(threads)),
        java.util.List.of());
  }

  @Override
  public void generate(String prompt, SamplingOptions options, TokenStream stream) {
    Objects.requireNonNull(stream, "stream");
    try {
      execute(prompt, options, stream);
      stream.onComplete();
    } catch (IOException failure) {
      stream.onError(new IllegalStateException(engine.id() + " generation failed", failure));
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      stream.onError(
          new IllegalStateException(engine.id() + " generation interrupted", interrupted));
    } catch (RuntimeException failure) {
      stream.onError(failure);
    }
  }

  /** Generates text and returns engine-provided token and timing observations. */
  public LocalEngineGeneration generateObserved(String prompt, SamplingOptions options) {
    try {
      return execute(prompt, options, NoOpTokenStream.INSTANCE);
    } catch (IOException failure) {
      throw new IllegalStateException(engine.id() + " generation failed", failure);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(engine.id() + " generation interrupted", interrupted);
    }
  }

  @Override
  public void close() {
    client.close();
  }

  ObjectNode requestBody(String prompt, SamplingOptions options) {
    return engine == LocalEngine.OLLAMA
        ? ollamaRequest(prompt, options)
        : llamaRequest(prompt, options);
  }

  URI requestUri() {
    if (engine == LocalEngine.OLLAMA) {
      return endpoint.resolve("/api/generate");
    }
    return endpoint.resolve(
        promptMode == PromptMode.NATIVE_CHAT ? "/v1/chat/completions" : "/completion");
  }

  private ObjectNode ollamaRequest(String prompt, SamplingOptions options) {
    ObjectNode body = mapper.createObjectNode();
    body.put("model", model);
    body.put("prompt", prompt);
    body.put("stream", true);
    body.put("raw", promptMode == PromptMode.RAW);
    if (promptMode == PromptMode.NATIVE_CHAT) {
      body.put("think", thinking);
    }
    body.put("keep_alive", "5m");
    ObjectNode controls = body.putObject("options");
    controls.put("temperature", options.temperature());
    controls.put("top_k", options.topK());
    controls.put("top_p", options.topP());
    controls.put("num_predict", options.maxTokens());
    controls.put("num_ctx", contextLength);
    controls.put("num_thread", threads);
    controls.put("repeat_penalty", options.repetitionPenalty());
    if (options.seed() != null) {
      controls.put("seed", options.seed());
    }
    return body;
  }

  private ObjectNode llamaRequest(String prompt, SamplingOptions options) {
    ObjectNode body = mapper.createObjectNode();
    body.put("stream", true);
    body.put("temperature", options.temperature());
    body.put("top_k", options.topK());
    body.put("top_p", options.topP());
    body.put("repeat_penalty", options.repetitionPenalty());
    if (options.seed() != null) {
      body.put("seed", options.seed());
    }
    if (promptMode == PromptMode.NATIVE_CHAT) {
      body.put("model", model);
      body.put("max_tokens", options.maxTokens());
      body.putObject("chat_template_kwargs").put("enable_thinking", thinking);
      ObjectNode message = body.putArray("messages").addObject();
      message.put("role", "user");
      message.put("content", prompt);
    } else {
      body.put("prompt", prompt);
      body.put("n_predict", options.maxTokens());
      body.put("n_threads", threads);
      body.put("cache_prompt", false);
    }
    return body;
  }

  private LocalEngineGeneration execute(String prompt, SamplingOptions options, TokenStream stream)
      throws IOException, InterruptedException {
    Objects.requireNonNull(prompt, "prompt");
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(stream, "stream");
    if (prompt.isBlank()) {
      throw new IllegalArgumentException("prompt must not be blank");
    }

    HttpRequest request =
        HttpRequest.newBuilder(requestUri())
            .timeout(requestTimeout)
            .header("Accept", "text/event-stream, application/x-ndjson")
            .header("Content-Type", "application/json")
            .POST(
                HttpRequest.BodyPublishers.ofString(
                    mapper.writeValueAsString(requestBody(prompt, options))))
            .build();
    long start = System.nanoTime();
    HttpResponse<Stream<String>> response =
        withSingleTransportRetry(() -> client.send(request, HttpResponse.BodyHandlers.ofLines()));
    if (response.statusCode() < 200 || response.statusCode() >= 300) {
      throw new IOException(
          engine.id() + " returned HTTP " + response.statusCode() + " from " + requestUri());
    }

    GenerationObservation observation = new GenerationObservation(start, stream);
    try (Stream<String> lines = response.body()) {
      if (engine == LocalEngine.OLLAMA) {
        parseOllama(lines, observation);
      } else if (promptMode == PromptMode.NATIVE_CHAT) {
        parseLlamaChat(lines, observation);
      } else {
        parseLlamaCompletion(lines, observation);
      }
    }
    return observation.finish();
  }

  private void parseOllama(Stream<String> lines, GenerationObservation observation)
      throws IOException {
    boolean complete = false;
    for (String line : (Iterable<String>) lines::iterator) {
      if (line.isBlank()) {
        continue;
      }
      JsonNode event = mapper.readTree(line);
      observation.emit(event.path("response").asText(""));
      if (event.path("done").asBoolean(false)) {
        complete = true;
        int inputTokens = event.path("prompt_eval_count").asInt(0);
        int outputTokens = event.path("eval_count").asInt(0);
        long promptNanos = event.path("prompt_eval_duration").asLong(0);
        double prefillTokensPerSecond =
            inputTokens > 0 && promptNanos > 0 ? inputTokens * 1_000_000_000.0 / promptNanos : 0;
        observation.recordMetrics(
            inputTokens,
            outputTokens,
            prefillTokensPerSecond,
            event.path("load_duration").asLong(0) / 1_000_000.0);
      }
    }
    if (!complete) {
      throw new IOException("Ollama stream ended without a completion event");
    }
  }

  private void parseLlamaChat(Stream<String> lines, GenerationObservation observation)
      throws IOException {
    boolean complete = false;
    for (String line : (Iterable<String>) lines::iterator) {
      String payload = ssePayload(line);
      if (payload == null) {
        continue;
      }
      if ("[DONE]".equals(payload)) {
        complete = true;
        continue;
      }
      JsonNode event = mapper.readTree(payload);
      throwIfLlamaError(event);
      JsonNode choice = event.path("choices").path(0);
      JsonNode delta = choice.path("delta");
      if (!delta.path("reasoning_content").asText("").isEmpty()) {
        observation.observeToken();
      }
      observation.emit(delta.path("content").asText(""));
      JsonNode finishReason = choice.path("finish_reason");
      complete |= !finishReason.isMissingNode() && !finishReason.isNull();
      recordLlamaMetrics(event, observation);
    }
    if (!complete) {
      throw new IOException("llama.cpp chat stream ended without a completion event");
    }
  }

  private void parseLlamaCompletion(Stream<String> lines, GenerationObservation observation)
      throws IOException {
    boolean complete = false;
    for (String line : (Iterable<String>) lines::iterator) {
      String payload = ssePayload(line);
      if (payload == null || "[DONE]".equals(payload)) {
        continue;
      }
      JsonNode event = mapper.readTree(payload);
      throwIfLlamaError(event);
      observation.emit(event.path("content").asText(""));
      complete |= event.path("stop").asBoolean(false);
      recordLlamaMetrics(event, observation);
    }
    if (!complete) {
      throw new IOException("llama.cpp completion stream ended without a stop event");
    }
  }

  private static String ssePayload(String line) {
    if (line == null || line.isBlank()) {
      return null;
    }
    return line.startsWith("data:") ? line.substring("data:".length()).trim() : line;
  }

  private static void throwIfLlamaError(JsonNode event) throws IOException {
    JsonNode error = event.path("error");
    if (error.isMissingNode() || error.isNull()) {
      return;
    }
    String message =
        error.isTextual() ? error.asText() : error.path("message").asText(error.toString());
    throw new IOException("llama.cpp stream error: " + message);
  }

  private static void recordLlamaMetrics(JsonNode event, GenerationObservation observation) {
    JsonNode timings = event.path("timings");
    if (!timings.isMissingNode()) {
      int inputTokens = timings.path("prompt_n").asInt(0);
      int outputTokens = timings.path("predicted_n").asInt(0);
      double promptMillis = timings.path("prompt_ms").asDouble(0);
      double prefillTokensPerSecond =
          inputTokens > 0 && promptMillis > 0 ? inputTokens * 1_000.0 / promptMillis : 0;
      observation.recordMetrics(inputTokens, outputTokens, prefillTokensPerSecond, 0);
      return;
    }
    JsonNode usage = event.path("usage");
    if (!usage.isMissingNode()) {
      observation.recordMetrics(
          usage.path("prompt_tokens").asInt(0), usage.path("completion_tokens").asInt(0), 0, 0);
    }
  }

  private static <T> T withSingleTransportRetry(TransportOperation<T> operation)
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

  /** Builder for local inference engine clients. */
  public static final class Builder {
    private final LocalEngine engine;
    private final String model;
    private final URI endpoint;
    private PromptMode promptMode = PromptMode.NATIVE_CHAT;
    private boolean thinking;
    private int contextLength = 2_048;
    private int threads = Runtime.getRuntime().availableProcessors();
    private Duration connectTimeout = Duration.ofSeconds(30);
    private Duration requestTimeout = Duration.ofMinutes(30);

    private Builder(LocalEngine engine, String model, URI endpoint) {
      this.engine = Objects.requireNonNull(engine, "engine");
      this.model = requireText(model, "model");
      this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
      if (!"http".equalsIgnoreCase(endpoint.getScheme())
          && !"https".equalsIgnoreCase(endpoint.getScheme())) {
        throw new IllegalArgumentException("endpoint must use HTTP or HTTPS");
      }
    }

    public Builder promptMode(PromptMode promptMode) {
      this.promptMode = Objects.requireNonNull(promptMode, "promptMode");
      return this;
    }

    /** Enables or disables model-native reasoning for chat requests. */
    public Builder thinking(boolean thinking) {
      this.thinking = thinking;
      return this;
    }

    public Builder contextLength(int contextLength) {
      this.contextLength = positive(contextLength, "contextLength");
      return this;
    }

    public Builder threads(int threads) {
      this.threads = positive(threads, "threads");
      return this;
    }

    public Builder connectTimeout(Duration connectTimeout) {
      this.connectTimeout = positive(connectTimeout, "connectTimeout");
      return this;
    }

    public Builder requestTimeout(Duration requestTimeout) {
      this.requestTimeout = positive(requestTimeout, "requestTimeout");
      return this;
    }

    public LocalEngineGenerationModel build() {
      return new LocalEngineGenerationModel(this);
    }

    private static int positive(int value, String name) {
      if (value < 1) {
        throw new IllegalArgumentException(name + " must be positive");
      }
      return value;
    }

    private static Duration positive(Duration value, String name) {
      Objects.requireNonNull(value, name);
      if (value.isZero() || value.isNegative()) {
        throw new IllegalArgumentException(name + " must be positive");
      }
      return value;
    }
  }

  private static String requireText(String value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value;
  }

  private enum LocalEngine {
    LLAMA_CPP("llama.cpp"),
    OLLAMA("ollama");

    private final String id;

    LocalEngine(String id) {
      this.id = id;
    }

    String id() {
      return id;
    }
  }

  @FunctionalInterface
  private interface TransportOperation<T> {
    T execute() throws IOException, InterruptedException;
  }

  private static final class GenerationObservation {
    private final long start;
    private final TokenStream stream;
    private final StringBuilder text = new StringBuilder();
    private double firstTokenMillis = -1;
    private int inputTokens;
    private int outputTokens;
    private double prefillTokensPerSecond;
    private double loadMillis;

    private GenerationObservation(long start, TokenStream stream) {
      this.start = start;
      this.stream = stream;
    }

    private void emit(String token) {
      if (token.isEmpty()) {
        return;
      }
      if (firstTokenMillis < 0) {
        firstTokenMillis = elapsedMillis();
      }
      text.append(token);
      stream.onToken(token);
    }

    private void observeToken() {
      if (firstTokenMillis < 0) {
        firstTokenMillis = elapsedMillis();
      }
    }

    private void recordMetrics(
        int inputTokens, int outputTokens, double prefillTokensPerSecond, double loadMillis) {
      this.inputTokens = inputTokens;
      this.outputTokens = outputTokens;
      this.prefillTokensPerSecond = prefillTokensPerSecond;
      this.loadMillis = loadMillis;
    }

    private LocalEngineGeneration finish() {
      double totalMillis = elapsedMillis();
      return new LocalEngineGeneration(
          text.toString(),
          inputTokens,
          outputTokens,
          Math.max(0, firstTokenMillis),
          totalMillis,
          prefillTokensPerSecond,
          loadMillis);
    }

    private double elapsedMillis() {
      return (System.nanoTime() - start) / 1_000_000.0;
    }
  }

  private enum NoOpTokenStream implements TokenStream {
    INSTANCE;

    @Override
    public void onToken(String token) {}

    @Override
    public void onComplete() {}

    @Override
    public void onError(Throwable throwable) {}
  }
}
