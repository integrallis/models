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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class LocalEngineGenerationModelTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void streamsLlamaCppNativeChatAndSendsSamplingControls() throws Exception {
    AtomicReference<JsonNode> requestBody = new AtomicReference<>();
    HttpServer server = server("/v1/chat/completions", requestBody, llamaStream());
    try {
      URI endpoint = endpoint(server);
      try (LocalEngineGenerationModel model =
          LocalEngineGenerationModel.llamaCpp("model.gguf", endpoint)
              .promptMode(PromptMode.NATIVE_CHAT)
              .build()) {
        LocalEngineGeneration result =
            model.generateObserved(
                "Use the supplied context.",
                SamplingOptions.builder()
                    .temperature(0)
                    .topK(1)
                    .topP(1)
                    .maxTokens(32)
                    .seed(42)
                    .repetitionPenalty(1)
                    .build());

        assertThat(result.text()).isEqualTo("grounded answer [doc-1]");
        assertThat(result.inputTokens()).isEqualTo(100);
        assertThat(result.outputTokens()).isEqualTo(5);
        assertThat(result.prefillTokensPerSecond()).isEqualTo(250);
        assertThat(result.firstTokenMillis()).isPositive();
        assertThat(result.totalMillis()).isGreaterThanOrEqualTo(result.firstTokenMillis());
        assertThat(requestBody.get().path("model").asText()).isEqualTo("model.gguf");
        assertThat(requestBody.get().path("messages").get(0).path("role").asText())
            .isEqualTo("user");
        assertThat(requestBody.get().path("messages").get(0).path("content").asText())
            .isEqualTo("Use the supplied context.");
        assertThat(requestBody.get().path("max_tokens").asInt()).isEqualTo(32);
        assertThat(requestBody.get().path("top_k").asInt()).isEqualTo(1);
        assertThat(requestBody.get().path("seed").asLong()).isEqualTo(42);
        assertThat(
                requestBody.get().path("chat_template_kwargs").path("enable_thinking").asBoolean())
            .isFalse();
      }
    } finally {
      server.stop(0);
    }
  }

  @Test
  void streamsOllamaWithItsModelOwnedChatTemplate() throws Exception {
    AtomicReference<JsonNode> requestBody = new AtomicReference<>();
    HttpServer server = server("/api/generate", requestBody, ollamaStream());
    try {
      try (LocalEngineGenerationModel model =
          LocalEngineGenerationModel.ollama("qwen3:0.6b", endpoint(server)).build()) {
        LocalEngineGeneration result =
            model.generateObserved(
                "Question", SamplingOptions.builder().temperature(0).maxTokens(16).seed(7).build());

        assertThat(result.text()).isEqualTo("answer [doc-1]");
        assertThat(result.inputTokens()).isEqualTo(80);
        assertThat(result.outputTokens()).isEqualTo(4);
        assertThat(result.prefillTokensPerSecond()).isEqualTo(200);
        assertThat(result.loadMillis()).isEqualTo(7);
        assertThat(requestBody.get().path("raw").asBoolean()).isFalse();
        assertThat(requestBody.get().path("think").asBoolean()).isFalse();
        assertThat(requestBody.get().path("options").path("num_predict").asInt()).isEqualTo(16);
        assertThat(requestBody.get().path("options").path("seed").asLong()).isEqualTo(7);
      }
    } finally {
      server.stop(0);
    }
  }

  @Test
  void streamsRawLlamaCompletionThroughThePublicCallbackApi() throws Exception {
    AtomicReference<JsonNode> requestBody = new AtomicReference<>();
    HttpServer server = server("/completion", requestBody, llamaCompletionStream());
    try {
      try (LocalEngineGenerationModel model =
          LocalEngineGenerationModel.llamaCpp("raw-model.gguf", endpoint(server))
              .promptMode(PromptMode.RAW)
              .contextLength(1_024)
              .threads(2)
              .connectTimeout(Duration.ofSeconds(1))
              .requestTimeout(Duration.ofSeconds(5))
              .build()) {
        RecordingTokenStream stream = new RecordingTokenStream();
        model.generate(
            "raw prompt", SamplingOptions.builder().temperature(0).maxTokens(8).build(), stream);

        assertThat(stream.tokens).containsExactly("raw", " answer");
        assertThat(stream.completed).isTrue();
        assertThat(stream.failure).isNull();
        assertThat(model.modelName()).isEqualTo("raw-model.gguf");
        assertThat(model.diagnostics().environment())
            .containsEntry("promptMode", "RAW")
            .containsEntry("contextLength", "1024")
            .containsEntry("threads", "2");
        assertThat(requestBody.get().path("prompt").asText()).isEqualTo("raw prompt");
        assertThat(requestBody.get().path("cache_prompt").asBoolean()).isFalse();
      }
    } finally {
      server.stop(0);
    }
  }

  @Test
  void canExplicitlyEnableNativeModelThinking() {
    try (LocalEngineGenerationModel model =
        LocalEngineGenerationModel.llamaCpp("thinking-model", URI.create("http://127.0.0.1:8080"))
            .thinking(true)
            .build()) {
      JsonNode body =
          model.requestBody(
              "prompt", SamplingOptions.builder().temperature(0).maxTokens(8).build());

      assertThat(body.path("chat_template_kwargs").path("enable_thinking").asBoolean()).isTrue();
      assertThat(model.diagnostics().environment()).containsEntry("thinking", "true");
    }
  }

  @Test
  void rejectsInvalidBuilderAndObservationValues() {
    URI endpoint = URI.create("http://127.0.0.1:8080");

    assertThatThrownBy(() -> LocalEngineGenerationModel.llamaCpp("", endpoint))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                LocalEngineGenerationModel.llamaCpp("model", URI.create("file:///tmp/model.gguf")))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> LocalEngineGenerationModel.llamaCpp("model", endpoint).contextLength(0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> LocalEngineGenerationModel.llamaCpp("model", endpoint).threads(0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                LocalEngineGenerationModel.llamaCpp("model", endpoint)
                    .connectTimeout(Duration.ZERO))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                LocalEngineGenerationModel.llamaCpp("model", endpoint)
                    .requestTimeout(Duration.ofSeconds(-1)))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new LocalEngineGeneration("text", -1, 1, 1, 2, 3, 4))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new LocalEngineGeneration("text", 1, 1, 3, 2, 3, 4))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void reportsAnIncompleteStreamThroughTheErrorCallback() throws Exception {
    AtomicReference<JsonNode> requestBody = new AtomicReference<>();
    HttpServer server =
        server("/api/generate", requestBody, "{\"response\":\"partial\",\"done\":false}\n");
    try {
      try (LocalEngineGenerationModel model =
          LocalEngineGenerationModel.ollama("model", endpoint(server)).build()) {
        RecordingTokenStream stream = new RecordingTokenStream();

        model.generate(
            "prompt", SamplingOptions.builder().temperature(0).maxTokens(8).build(), stream);

        assertThat(stream.tokens).containsExactly("partial");
        assertThat(stream.completed).isFalse();
        assertThat(stream.failure)
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("generation failed");
      }
    } finally {
      server.stop(0);
    }
  }

  @Test
  void surfacesLlamaCppStreamErrorsInsteadOfReportingAnAmbiguousTruncation() throws Exception {
    AtomicReference<JsonNode> requestBody = new AtomicReference<>();
    HttpServer server =
        server(
            "/v1/chat/completions",
            requestBody,
            "data: {\"error\":{\"message\":\"chat template parser rejected output\","
                + "\"type\":\"server_error\",\"code\":500}}\n\n");
    try {
      try (LocalEngineGenerationModel model =
          LocalEngineGenerationModel.llamaCpp("model", endpoint(server)).build()) {
        assertThatThrownBy(
                () ->
                    model.generateObserved(
                        "prompt", SamplingOptions.builder().temperature(0).maxTokens(8).build()))
            .isInstanceOf(IllegalStateException.class)
            .rootCause()
            .hasMessageContaining("chat template parser rejected output");
      }
    } finally {
      server.stop(0);
    }
  }

  private static HttpServer server(
      String path, AtomicReference<JsonNode> requestBody, String response) throws IOException {
    HttpServer server =
        HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
    server.createContext(
        path,
        exchange -> {
          captureRequest(exchange, requestBody);
          byte[] body = response.getBytes(StandardCharsets.UTF_8);
          exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
          exchange.sendResponseHeaders(200, 0);
          exchange.getResponseBody().write(body);
          exchange.close();
        });
    server.start();
    return server;
  }

  private static void captureRequest(HttpExchange exchange, AtomicReference<JsonNode> requestBody)
      throws IOException {
    requestBody.set(MAPPER.readTree(exchange.getRequestBody()));
  }

  private static URI endpoint(HttpServer server) {
    return URI.create("http://127.0.0.1:" + server.getAddress().getPort());
  }

  private static String llamaStream() {
    return "data: {\"choices\":[{\"delta\":{\"content\":\"grounded answer\"}}]}\n\n"
        + "data: {\"choices\":[{\"delta\":{\"content\":\" [doc-1]\"},"
        + "\"finish_reason\":\"stop\"}],\"timings\":{\"prompt_n\":100,"
        + "\"predicted_n\":5,\"prompt_ms\":400.0}}\n\n"
        + "data: [DONE]\n\n";
  }

  private static String ollamaStream() {
    return "{\"response\":\"answer\",\"done\":false}\n"
        + "{\"response\":\" [doc-1]\",\"done\":true,\"prompt_eval_count\":80,"
        + "\"eval_count\":4,\"prompt_eval_duration\":400000000,"
        + "\"load_duration\":7000000}\n";
  }

  private static String llamaCompletionStream() {
    return "data: {\"content\":\"raw\",\"stop\":false}\n\n"
        + "data: {\"content\":\" answer\",\"stop\":true,\"timings\":{\"prompt_n\":20,"
        + "\"predicted_n\":2,\"prompt_ms\":100.0}}\n\n";
  }

  private static final class RecordingTokenStream implements TokenStream {
    private final List<String> tokens = new ArrayList<>();
    private boolean completed;
    private Throwable failure;

    @Override
    public void onToken(String token) {
      tokens.add(token);
    }

    @Override
    public void onComplete() {
      completed = true;
    }

    @Override
    public void onError(Throwable throwable) {
      failure = throwable;
    }
  }
}
