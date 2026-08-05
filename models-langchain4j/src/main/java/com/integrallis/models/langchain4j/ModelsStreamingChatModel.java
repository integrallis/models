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
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.runtime.RuntimeTextGenerationModel;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.ToolCallScanner;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/** LangChain4j {@link StreamingChatModel} backed by the Models runtime generation loop. */
public final class ModelsStreamingChatModel implements StreamingChatModel {

  private final TextGenerationModel model;
  private final ChatTemplate template;
  private final SamplingOptions defaults;

  public ModelsStreamingChatModel(InferenceBackend backend) {
    this(new RuntimeTextGenerationModel(backend));
  }

  public ModelsStreamingChatModel(InferenceBackend backend, SamplingOptions defaults) {
    this(new RuntimeTextGenerationModel(backend), defaults);
  }

  public ModelsStreamingChatModel(TextGenerationModel model) {
    this(model, SamplingOptions.builder().build());
  }

  public ModelsStreamingChatModel(TextGenerationModel model, SamplingOptions defaults) {
    this(model, ChatTemplate.RAW, defaults);
  }

  public ModelsStreamingChatModel(
      InferenceBackend backend, ChatTemplate template, SamplingOptions defaults) {
    this(new RuntimeTextGenerationModel(backend), template, defaults);
  }

  public ModelsStreamingChatModel(
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
  public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(handler, "handler");
    StringBuilder accumulated = new StringBuilder();
    AtomicBoolean terminalSignalSent = new AtomicBoolean();
    // A tool call means nothing until it is complete, and its delimiters are not user-facing text.
    // So when tools are declared, deltas are withheld and the result is delivered once, whole.
    boolean toolsDeclared = !LangChain4jChatRequestMapper.tools(request).isEmpty();
    TokenStream stream =
        new TokenStream() {
          @Override
          public void onToken(String token) {
            if (!terminalSignalSent.get()) {
              accumulated.append(token);
              if (!toolsDeclared) {
                handler.onPartialResponse(token);
              }
            }
          }

          @Override
          public void onComplete() {
            if (terminalSignalSent.compareAndSet(false, true)) {
              handler.onCompleteResponse(completed(accumulated.toString(), toolsDeclared));
            }
          }

          @Override
          public void onError(Throwable failure) {
            if (terminalSignalSent.compareAndSet(false, true)) {
              handler.onError(failure);
            }
          }
        };

    try {
      model.generate(
          LangChain4jChatRequestMapper.prompt(request, template),
          LangChain4jChatRequestMapper.options(request, defaults),
          stream);
    } catch (RuntimeException | Error failure) {
      if (terminalSignalSent.compareAndSet(false, true)) {
        handler.onError(failure);
      } else {
        throw failure;
      }
    }
  }

  /** Builds the terminal response, recovering any tool calls the model produced. */
  private ChatResponse completed(String output, boolean toolsDeclared) {
    ToolCallScanner.Result scan =
        toolsDeclared
            ? ToolCallScanner.scan(output, template.toolSyntax())
            : ToolCallScanner.Result.plainText(output);
    if (!scan.hasCalls()) {
      return ChatResponse.builder()
          .aiMessage(AiMessage.from(scan.content()))
          .modelName(model.modelName())
          .build();
    }
    List<ToolExecutionRequest> requests = new ArrayList<>(scan.toolCalls().size());
    for (ToolCall call : scan.toolCalls()) {
      requests.add(
          ToolExecutionRequest.builder()
              .id(call.id())
              .name(call.name())
              .arguments(call.argumentsJson())
              .build());
    }
    return ChatResponse.builder()
        .aiMessage(new AiMessage(scan.content(), requests))
        .finishReason(FinishReason.TOOL_EXECUTION)
        .modelName(model.modelName())
        .build();
  }
}
