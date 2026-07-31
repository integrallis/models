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
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.runtime.RuntimeTextGenerationModel;
import com.integrallis.models.runtime.chat.ChatTemplate;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.util.Objects;

/** LangChain4J {@link ChatModel} backed by the models runtime generation loop. */
public final class ModelsChatModel implements ChatModel {
  private final TextGenerationModel model;
  private final ChatTemplate template;
  private final SamplingOptions defaults;

  public ModelsChatModel(InferenceBackend backend) {
    this(new RuntimeTextGenerationModel(backend));
  }

  public ModelsChatModel(InferenceBackend backend, SamplingOptions defaults) {
    this(new RuntimeTextGenerationModel(backend), defaults);
  }

  public ModelsChatModel(TextGenerationModel model) {
    this(model, SamplingOptions.builder().build());
  }

  public ModelsChatModel(TextGenerationModel model, SamplingOptions defaults) {
    this(model, ChatTemplate.RAW, defaults);
  }

  public ModelsChatModel(
      InferenceBackend backend, ChatTemplate template, SamplingOptions defaults) {
    this(new RuntimeTextGenerationModel(backend), template, defaults);
  }

  public ModelsChatModel(
      TextGenerationModel model, ChatTemplate template, SamplingOptions defaults) {
    this.model = Objects.requireNonNull(model, "model");
    this.template = Objects.requireNonNull(template, "template");
    this.defaults = Objects.requireNonNull(defaults, "defaults");
  }

  /** Returns the execution decisions selected by the wrapped backend. */
  public BackendDiagnostics diagnostics() {
    return model.diagnostics();
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    Objects.requireNonNull(request, "request");
    String output =
        model.generate(
            LangChain4jChatRequestMapper.prompt(request, template),
            LangChain4jChatRequestMapper.options(request, defaults));
    return ChatResponse.builder()
        .aiMessage(AiMessage.from(output))
        .modelName(model.modelName())
        .build();
  }
}
