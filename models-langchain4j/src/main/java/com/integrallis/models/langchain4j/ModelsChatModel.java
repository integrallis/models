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
package com.integrallis.models.langchain4j;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.runtime.GenerationLoop;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.util.Objects;

/** LangChain4J {@link ChatModel} backed by the models runtime generation loop. */
public final class ModelsChatModel implements ChatModel {
  private final InferenceBackend backend;
  private final SamplingOptions defaults;

  public ModelsChatModel(InferenceBackend backend) {
    this(backend, SamplingOptions.builder().build());
  }

  public ModelsChatModel(InferenceBackend backend, SamplingOptions defaults) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.defaults = Objects.requireNonNull(defaults, "defaults");
  }

  /** Returns the execution decisions selected by the wrapped backend. */
  public BackendDiagnostics diagnostics() {
    return backend.diagnostics();
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    Objects.requireNonNull(request, "request");
    String output = new GenerationLoop(backend).generate(prompt(request), options(request));
    return ChatResponse.builder()
        .aiMessage(AiMessage.from(output))
        .modelName(backend.metadata().modelName())
        .build();
  }

  private String prompt(ChatRequest request) {
    StringBuilder prompt = new StringBuilder();
    for (ChatMessage message : request.messages()) {
      if (!prompt.isEmpty()) {
        prompt.append('\n');
      }
      prompt.append(render(message));
    }
    return prompt.toString();
  }

  private String render(ChatMessage message) {
    if (message instanceof UserMessage userMessage && userMessage.hasSingleText()) {
      return userMessage.singleText();
    }
    if (message instanceof SystemMessage systemMessage) {
      return systemMessage.text();
    }
    if (message instanceof AiMessage aiMessage) {
      return aiMessage.text();
    }
    return message.toString();
  }

  private SamplingOptions options(ChatRequest request) {
    SamplingOptions.Builder builder =
        SamplingOptions.builder()
            .temperature(defaults.temperature())
            .topP(defaults.topP())
            .topK(defaults.topK())
            .maxTokens(defaults.maxTokens())
            .repetitionPenalty(defaults.repetitionPenalty());
    if (defaults.seed() != null) {
      builder.seed(defaults.seed());
    }
    if (request.temperature() != null) {
      builder.temperature(request.temperature().floatValue());
    }
    if (request.topP() != null) {
      builder.topP(request.topP().floatValue());
    }
    if (request.topK() != null) {
      builder.topK(request.topK());
    }
    if (request.maxOutputTokens() != null) {
      builder.maxTokens(request.maxOutputTokens());
    }
    return builder.build();
  }
}
