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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.runtime.chat.ChatTemplate;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import java.util.ArrayList;

final class LangChain4jChatRequestMapper {

  private LangChain4jChatRequestMapper() {}

  static ModelPrompt prompt(ChatRequest request, ChatTemplate template) {
    ArrayList<com.integrallis.models.runtime.chat.ChatMessage> messages =
        new ArrayList<>(request.messages().size());
    for (ChatMessage message : request.messages()) {
      messages.add(render(message));
    }
    return template.render(messages);
  }

  static SamplingOptions options(ChatRequest request, SamplingOptions defaults) {
    SamplingOptions.Builder builder =
        SamplingOptions.builder()
            .temperature(defaults.temperature())
            .topP(defaults.topP())
            .topK(defaults.topK())
            .maxTokens(defaults.maxTokens())
            .repetitionPenalty(defaults.repetitionPenalty())
            .stopSequences(defaults.stopSequences());
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
    if (request.stopSequences() != null && !request.stopSequences().isEmpty()) {
      builder.stopSequences(request.stopSequences());
    }
    return builder.build();
  }

  private static com.integrallis.models.runtime.chat.ChatMessage render(ChatMessage message) {
    if (message instanceof UserMessage userMessage && userMessage.hasSingleText()) {
      return com.integrallis.models.runtime.chat.ChatMessage.user(userMessage.singleText());
    }
    if (message instanceof SystemMessage systemMessage) {
      return com.integrallis.models.runtime.chat.ChatMessage.system(systemMessage.text());
    }
    if (message instanceof AiMessage aiMessage) {
      return com.integrallis.models.runtime.chat.ChatMessage.assistant(aiMessage.text());
    }
    if (message instanceof ToolExecutionResultMessage toolMessage) {
      return com.integrallis.models.runtime.chat.ChatMessage.tool(toolMessage.text());
    }
    throw new IllegalArgumentException("Unsupported LangChain4j chat message: " + message.type());
  }
}
