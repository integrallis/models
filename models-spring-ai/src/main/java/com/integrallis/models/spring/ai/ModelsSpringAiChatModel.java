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
package com.integrallis.models.spring.ai;

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
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.ToolCallScanner;
import io.micrometer.observation.ObservationRegistry;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.ai.chat.observation.ChatModelObservationDocumentation;
import org.springframework.ai.chat.observation.DefaultChatModelObservationConvention;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

/** Spring AI {@link ChatModel} backed by the Models runtime generation loop. */
public final class ModelsSpringAiChatModel implements ChatModel {
  private static final String PROVIDER = "integrallis";
  private static final ChatModelObservationConvention DEFAULT_OBSERVATION_CONVENTION =
      new DefaultChatModelObservationConvention();

  private final SamplingOptions defaults;
  private final TextGenerationModel model;
  private final String modelName;
  private final ChatTemplate template;
  private final ObservationRegistry observationRegistry;
  private ChatModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

  public ModelsSpringAiChatModel(InferenceBackend backend) {
    this(new RuntimeTextGenerationModel(backend));
  }

  public ModelsSpringAiChatModel(InferenceBackend backend, SamplingOptions defaults) {
    this(new RuntimeTextGenerationModel(backend), defaults);
  }

  public ModelsSpringAiChatModel(TextGenerationModel model) {
    this(model, SamplingOptions.builder().build());
  }

  public ModelsSpringAiChatModel(TextGenerationModel model, SamplingOptions defaults) {
    this(model, ChatTemplate.RAW, defaults);
  }

  public ModelsSpringAiChatModel(
      InferenceBackend backend, ChatTemplate template, SamplingOptions defaults) {
    this(new RuntimeTextGenerationModel(backend), template, defaults);
  }

  public ModelsSpringAiChatModel(
      InferenceBackend backend,
      ChatTemplate template,
      SamplingOptions defaults,
      ObservationRegistry observationRegistry) {
    this(new RuntimeTextGenerationModel(backend), template, defaults, observationRegistry);
  }

  public ModelsSpringAiChatModel(
      TextGenerationModel model, ChatTemplate template, SamplingOptions defaults) {
    this(model, template, defaults, ObservationRegistry.NOOP);
  }

  public ModelsSpringAiChatModel(
      TextGenerationModel model,
      String modelName,
      ChatTemplate template,
      SamplingOptions defaults) {
    this(model, modelName, template, defaults, ObservationRegistry.NOOP);
  }

  public ModelsSpringAiChatModel(
      TextGenerationModel model,
      ChatTemplate template,
      SamplingOptions defaults,
      ObservationRegistry observationRegistry) {
    this(model, null, template, defaults, observationRegistry, true);
  }

  public ModelsSpringAiChatModel(
      TextGenerationModel model,
      String modelName,
      ChatTemplate template,
      SamplingOptions defaults,
      ObservationRegistry observationRegistry) {
    this(model, modelName, template, defaults, observationRegistry, false);
  }

  private ModelsSpringAiChatModel(
      TextGenerationModel model,
      String modelName,
      ChatTemplate template,
      SamplingOptions defaults,
      ObservationRegistry observationRegistry,
      boolean useRuntimeModelName) {
    this.model = Objects.requireNonNull(model, "model");
    this.modelName = useRuntimeModelName ? null : requireModelName(modelName);
    this.template = Objects.requireNonNull(template, "template");
    this.defaults = Objects.requireNonNull(defaults, "defaults");
    this.observationRegistry = Objects.requireNonNull(observationRegistry, "observationRegistry");
  }

  @Override
  public ChatResponse call(Prompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    var context = observationContext(prompt);
    return ChatModelObservationDocumentation.CHAT_MODEL_OPERATION
        .observation(
            observationConvention,
            DEFAULT_OBSERVATION_CONVENTION,
            () -> context,
            observationRegistry)
        .observe(() -> callObserved(prompt, context));
  }

  private ChatResponse callObserved(Prompt prompt, ChatModelObservationContext context) {
    List<ToolSpec> tools = declaredTools(prompt);
    ModelPrompt rendered = render(prompt, tools);
    SamplingOptions requested = options(prompt);
    GenerationOutput generated = generate(rendered, requested, tools);
    ChatResponse response =
        tools.isEmpty()
            ? response(generated.text(), generated.usage())
            : toolAwareResponse(generated.text(), generated.usage());
    context.setResponse(response);
    return response;
  }

  @Override
  public Flux<ChatResponse> stream(Prompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    return Flux.defer(
            () -> {
              var context = observationContext(prompt);
              var observation =
                  ChatModelObservationDocumentation.CHAT_MODEL_OPERATION
                      .observation(
                          observationConvention,
                          DEFAULT_OBSERVATION_CONVENTION,
                          () -> context,
                          observationRegistry)
                      .start();
              return streamObserved(prompt)
                  .doOnNext(context::setResponse)
                  .doOnError(observation::error)
                  .doFinally(ignored -> observation.stop());
            })
        .subscribeOn(Schedulers.boundedElastic());
  }

  private Flux<ChatResponse> streamObserved(Prompt prompt) {
    List<ToolSpec> tools = declaredTools(prompt);
    ModelPrompt rendered = render(prompt, tools);
    SamplingOptions requested = options(prompt);
    return Flux.<ChatResponse>create(
        sink -> {
          // A tool call only means anything once it is complete, so when tools are in play the
          // output is accumulated and emitted as one response rather than surfaced in pieces.
          StringBuilder accumulated = tools.isEmpty() ? null : new StringBuilder();
          TokenStream stream =
              new TokenStream() {
                @Override
                public void onToken(String token) {
                  if (accumulated == null) {
                    sink.next(response(token, null));
                  } else {
                    accumulated.append(token);
                  }
                }

                @Override
                public void onComplete() {
                  complete(null);
                }

                @Override
                public void onComplete(GenerationUsage usage) {
                  complete(usage);
                }

                private void complete(GenerationUsage usage) {
                  if (accumulated != null) {
                    sink.next(toolAwareResponse(accumulated.toString(), usage));
                  } else if (usage != null) {
                    sink.next(response("", usage));
                  }
                  sink.complete();
                }

                @Override
                public void onError(Throwable failure) {
                  sink.error(failure);
                }
              };
          Optional<TokenConstraint> constraint = toolConstraint(tools);
          if (constraint.isPresent()) {
            ((ConstrainedTextGenerationModel) model)
                .generate(rendered, requested, stream, constraint.get());
          } else {
            model.generate(rendered, requested, stream);
          }
        });
  }

  public ChatOptions getOptions() {
    return ChatOptions.builder()
        .model(resolvedModelName())
        .temperature((double) defaults.temperature())
        .topP((double) defaults.topP())
        .topK(defaults.topK())
        .maxTokens(defaults.maxTokens())
        .stopSequences(defaults.stopSequences())
        .build();
  }

  public void setObservationConvention(ChatModelObservationConvention observationConvention) {
    this.observationConvention =
        Objects.requireNonNull(observationConvention, "observationConvention");
  }

  private ChatModelObservationContext observationContext(Prompt prompt) {
    ChatOptions observedOptions = observationOptions(prompt.getOptions());
    var observedPrompt = new Prompt(prompt.getInstructions(), observedOptions);
    return ChatModelObservationContext.builder().prompt(observedPrompt).provider(PROVIDER).build();
  }

  private ChatOptions observationOptions(ChatOptions requested) {
    if (requested == null) {
      return getOptions();
    }
    var builder = ChatOptions.builder().model(resolvedModelName());
    if (requested.getFrequencyPenalty() != null) {
      builder.frequencyPenalty(requested.getFrequencyPenalty());
    }
    if (requested.getMaxTokens() != null) {
      builder.maxTokens(requested.getMaxTokens());
    }
    if (requested.getPresencePenalty() != null) {
      builder.presencePenalty(requested.getPresencePenalty());
    }
    if (requested.getStopSequences() != null) {
      builder.stopSequences(requested.getStopSequences());
    }
    if (requested.getTemperature() != null) {
      builder.temperature(requested.getTemperature());
    }
    if (requested.getTopK() != null) {
      builder.topK(requested.getTopK());
    }
    if (requested.getTopP() != null) {
      builder.topP(requested.getTopP());
    }
    return builder.build();
  }

  private SamplingOptions options(Prompt prompt) {
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

    ChatOptions requested = prompt.getOptions();
    if (requested != null) {
      if (requested.getTemperature() != null) {
        builder.temperature(requested.getTemperature().floatValue());
      }
      if (requested.getTopP() != null) {
        builder.topP(requested.getTopP().floatValue());
      }
      if (requested.getTopK() != null) {
        builder.topK(requested.getTopK());
      }
      if (requested.getMaxTokens() != null) {
        builder.maxTokens(requested.getMaxTokens());
      }
      if (requested.getStopSequences() != null) {
        builder.stopSequences(requested.getStopSequences());
      }
    }
    return builder.build();
  }

  /**
   * Reads the tools the caller declared, if any.
   *
   * <p>Spring AI 2.0 exposes them directly on {@link ToolCallingChatOptions}, so the adapter does
   * not need a {@code ToolCallingManager} — the execution loop lives in the advisor, not here.
   */
  private static List<ToolSpec> declaredTools(Prompt prompt) {
    if (!(prompt.getOptions() instanceof ToolCallingChatOptions toolOptions)) {
      return List.of();
    }
    List<ToolCallback> callbacks = toolOptions.getToolCallbacks();
    if (callbacks == null || callbacks.isEmpty()) {
      return List.of();
    }
    List<ToolSpec> tools = new ArrayList<>(callbacks.size());
    for (ToolCallback callback : callbacks) {
      ToolDefinition definition = callback.getToolDefinition();
      tools.add(
          new ToolSpec(definition.name(), definition.description(), definition.inputSchema()));
    }
    return List.copyOf(tools);
  }

  private ModelPrompt render(Prompt prompt, List<ToolSpec> tools) {
    List<ChatMessage> messages = new ArrayList<>(prompt.getInstructions().size());
    for (org.springframework.ai.chat.messages.Message message : prompt.getInstructions()) {
      switch (message.getMessageType()) {
        case SYSTEM -> messages.add(ChatMessage.system(message.getText()));
        case USER -> messages.add(ChatMessage.user(message.getText()));
        case ASSISTANT -> messages.add(assistantMessage(message));
        case TOOL -> appendToolResponses(messages, message);
      }
    }
    return tools.isEmpty() ? template.render(messages) : template.render(messages, tools);
  }

  private Optional<TokenConstraint> toolConstraint(List<ToolSpec> tools) {
    if (tools.isEmpty() || !(model instanceof ConstrainedTextGenerationModel constrainedModel)) {
      return Optional.empty();
    }
    return SpringAiToolCallConstraint.compile(
        constrainedModel.tokenizer(), template.toolSyntax(), tools);
  }

  /** Carries any tool calls the assistant previously made back into the rendered history. */
  private static ChatMessage assistantMessage(
      org.springframework.ai.chat.messages.Message message) {
    String text = message.getText() == null ? "" : message.getText();
    if (!(message instanceof AssistantMessage assistant) || !assistant.hasToolCalls()) {
      return ChatMessage.assistant(text);
    }
    List<ToolCall> calls = new ArrayList<>(assistant.getToolCalls().size());
    for (AssistantMessage.ToolCall call : assistant.getToolCalls()) {
      calls.add(new ToolCall(call.id(), call.name(), call.arguments()));
    }
    return ChatMessage.assistantToolCalls(text, calls);
  }

  /**
   * Expands one {@link ToolResponseMessage} into a tool turn per response.
   *
   * <p>Spring AI batches results into a single message; every chat template renders them
   * individually, and the ChatML family coalesces consecutive ones back into one user turn.
   */
  private static void appendToolResponses(
      List<ChatMessage> messages, org.springframework.ai.chat.messages.Message message) {
    if (message instanceof ToolResponseMessage responses) {
      for (ToolResponseMessage.ToolResponse response : responses.getResponses()) {
        messages.add(ChatMessage.tool(response.responseData()));
      }
      return;
    }
    messages.add(ChatMessage.tool(message.getText()));
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

  private ChatResponse response(String text, GenerationUsage usage) {
    return new ChatResponse(List.of(new Generation(new AssistantMessage(text))), metadata(usage));
  }

  /** Builds a response that surfaces any recovered tool calls to Spring AI's advisor. */
  private ChatResponse toolAwareResponse(String output, GenerationUsage usage) {
    ToolCallScanner.Result scan = ToolCallScanner.scan(output, template.toolSyntax());
    if (!scan.hasCalls()) {
      return response(scan.content(), usage);
    }
    List<AssistantMessage.ToolCall> calls = new ArrayList<>(scan.toolCalls().size());
    for (ToolCall call : scan.toolCalls()) {
      calls.add(
          new AssistantMessage.ToolCall(call.id(), "function", call.name(), call.argumentsJson()));
    }
    AssistantMessage assistant =
        AssistantMessage.builder().content(scan.content()).toolCalls(calls).build();
    return new ChatResponse(List.of(new Generation(assistant)), metadata(usage));
  }

  private ChatResponseMetadata metadata(GenerationUsage usage) {
    var metadata = ChatResponseMetadata.builder().model(resolvedModelName());
    if (usage != null) {
      metadata.usage(new DefaultUsage(usage.promptTokens(), usage.completionTokens()));
    }
    return metadata.build();
  }

  private String resolvedModelName() {
    return modelName == null ? model.modelName() : modelName;
  }

  private static String requireModelName(String modelName) {
    Objects.requireNonNull(modelName, "modelName");
    if (modelName.isBlank()) {
      throw new IllegalArgumentException("modelName must not be blank");
    }
    return modelName.strip();
  }

  private record GenerationOutput(String text, GenerationUsage usage) {}
}
