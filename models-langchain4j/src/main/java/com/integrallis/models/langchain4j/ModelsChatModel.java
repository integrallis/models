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
import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelGenerationException;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.ConstrainedTextGenerationModel;
import com.integrallis.models.runtime.RuntimeTextGenerationModel;
import com.integrallis.models.runtime.TokenConstraint;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.ToolCallScanner;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

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

  /**
   * Whether this model can take part in tool calling.
   *
   * <p>Narrower than "the family has a tool format". Gemma 4 and MiniCPM5 do, but encode arguments
   * as tagged pairs the runtime cannot yet turn into JSON without the declared schemas.
   */
  public boolean supportsTools() {
    return template.canParseToolCalls();
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    Objects.requireNonNull(request, "request");
    List<ToolSpec> tools = LangChain4jChatRequestMapper.tools(request);
    boolean toolsDeclared = !tools.isEmpty();
    ModelPrompt prompt = LangChain4jChatRequestMapper.prompt(request, template);
    SamplingOptions requested = LangChain4jChatRequestMapper.options(request, defaults);
    GenerationOutput generated = generate(prompt, requested, tools);

    // Without declared tools, output that merely resembles a call is ordinary text.
    ToolCallScanner.Result scan =
        toolsDeclared
            ? ToolCallScanner.scan(generated.text(), template.toolSyntax())
            : ToolCallScanner.Result.plainText(generated.text());
    if (!scan.hasCalls()) {
      var response =
          ChatResponse.builder()
              .aiMessage(AiMessage.from(scan.content()))
              .modelName(model.modelName());
      addUsage(response, generated.usage());
      return response.build();
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
    // LangChain4j drives its tool loop off the finish reason; Spring AI 2.0 keys on the message
    // carrying calls instead, so this has to be set explicitly here and not there.
    var response =
        ChatResponse.builder()
            .aiMessage(new AiMessage(scan.content(), requests))
            .finishReason(FinishReason.TOOL_EXECUTION)
            .modelName(model.modelName());
    addUsage(response, generated.usage());
    return response.build();
  }

  private GenerationOutput generate(
      ModelPrompt prompt, SamplingOptions options, List<ToolSpec> tools) {
    StringBuilder output = new StringBuilder();
    AtomicReference<GenerationUsage> usage = new AtomicReference<>();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    TokenStream stream =
        new TokenStream() {
          @Override
          public void onToken(String token) {
            output.append(token);
          }

          @Override
          public void onComplete() {}

          @Override
          public void onComplete(GenerationUsage completedUsage) {
            usage.set(completedUsage);
          }

          @Override
          public void onError(Throwable generationFailure) {
            failure.compareAndSet(null, generationFailure);
          }
        };
    Optional<TokenConstraint> constraint = toolConstraint(tools);
    if (constraint.isPresent()) {
      ((ConstrainedTextGenerationModel) model).generate(prompt, options, stream, constraint.get());
    } else {
      model.generate(prompt, options, stream);
    }
    throwFailure(failure.get());
    return new GenerationOutput(output.toString(), usage.get());
  }

  private static void addUsage(ChatResponse.Builder response, GenerationUsage usage) {
    if (usage != null) {
      response.tokenUsage(
          new TokenUsage(usage.promptTokens(), usage.completionTokens(), usage.totalTokens()));
    }
  }

  private static void throwFailure(Throwable failure) {
    if (failure instanceof RuntimeException runtimeFailure) {
      throw runtimeFailure;
    }
    if (failure instanceof Error error) {
      throw error;
    }
    if (failure != null) {
      throw new ModelGenerationException("text generation failed", failure);
    }
  }

  private Optional<TokenConstraint> toolConstraint(List<ToolSpec> tools) {
    if (tools.isEmpty() || !(model instanceof ConstrainedTextGenerationModel constrainedModel)) {
      return Optional.empty();
    }
    return LangChain4jToolCallConstraint.compile(
        constrainedModel.tokenizer(), template.toolSyntax(), tools);
  }

  private record GenerationOutput(String text, GenerationUsage usage) {}
}
