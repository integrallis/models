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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.AuxiliaryTextGenerationModel;
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
import com.integrallis.models.runtime.chat.ToolSpecSelector;
import io.micrometer.observation.ObservationRegistry;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
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
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

/** Spring AI {@link ChatModel} backed by the Models runtime generation loop. */
public final class ModelsSpringAiChatModel implements ChatModel {
  private static final String PROVIDER = "integrallis";
  private static final String DEFAULT_NO_APPLICABLE_TOOL_RESPONSE =
      "No applicable tool is available.";
  private static final int MAX_INTERNAL_TOOL_TURNS = 16;
  private static final ObjectMapper JSON = new ObjectMapper();
  private static final ChatModelObservationConvention DEFAULT_OBSERVATION_CONVENTION =
      new DefaultChatModelObservationConvention();

  private final SamplingOptions defaults;
  private final TextGenerationModel model;
  private final String modelName;
  private final ChatTemplate template;
  private final ObservationRegistry observationRegistry;
  private final Set<String> qualifiedCapabilities;
  private final ToolCallingManager toolCallingManager;
  private final ToolSpecSelector toolSelector;
  private final Map<String, TypedToolResultRenderer<?>> toolResultRenderers =
      new ConcurrentHashMap<>();
  private volatile boolean rawToolResults;
  private volatile String noApplicableToolResponse = DEFAULT_NO_APPLICABLE_TOOL_RESPONSE;
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
    this(model, modelName, template, defaults, ObservationRegistry.NOOP, null);
  }

  /**
   * Creates an adapter with an authoritative set of artifact capabilities.
   *
   * <p>When tools are supplied, the adapter rejects the call unless {@code qualifiedCapabilities}
   * contains {@code tool-calling}. ModelJars callers should pass the capabilities from the selected
   * descriptor so a chat-only artifact cannot silently attempt an unqualified tool workflow.
   */
  public ModelsSpringAiChatModel(
      TextGenerationModel model,
      String modelName,
      ChatTemplate template,
      SamplingOptions defaults,
      Set<String> qualifiedCapabilities) {
    this(model, modelName, template, defaults, ObservationRegistry.NOOP, qualifiedCapabilities);
  }

  public ModelsSpringAiChatModel(
      TextGenerationModel model,
      ChatTemplate template,
      SamplingOptions defaults,
      ObservationRegistry observationRegistry) {
    this(model, null, template, defaults, observationRegistry, true, null);
  }

  public ModelsSpringAiChatModel(
      TextGenerationModel model,
      String modelName,
      ChatTemplate template,
      SamplingOptions defaults,
      ObservationRegistry observationRegistry) {
    this(model, modelName, template, defaults, observationRegistry, false, null);
  }

  /** Creates an observed adapter with an authoritative set of artifact capabilities. */
  public ModelsSpringAiChatModel(
      TextGenerationModel model,
      String modelName,
      ChatTemplate template,
      SamplingOptions defaults,
      ObservationRegistry observationRegistry,
      Set<String> qualifiedCapabilities) {
    this(model, modelName, template, defaults, observationRegistry, false, qualifiedCapabilities);
  }

  private ModelsSpringAiChatModel(
      TextGenerationModel model,
      String modelName,
      ChatTemplate template,
      SamplingOptions defaults,
      ObservationRegistry observationRegistry,
      boolean useRuntimeModelName,
      Set<String> qualifiedCapabilities) {
    this.model = Objects.requireNonNull(model, "model");
    this.modelName = useRuntimeModelName ? null : requireModelName(modelName);
    this.template = Objects.requireNonNull(template, "template");
    this.defaults = Objects.requireNonNull(defaults, "defaults");
    this.observationRegistry = Objects.requireNonNull(observationRegistry, "observationRegistry");
    this.qualifiedCapabilities =
        qualifiedCapabilities == null ? null : Set.copyOf(qualifiedCapabilities);
    this.toolCallingManager =
        ToolCallingManager.builder().observationRegistry(observationRegistry).build();
    this.toolSelector =
        model instanceof AuxiliaryTextGenerationModel auxiliary
                && auxiliary.supportsContrastiveEncoding()
            ? new ToolSpecSelector(auxiliary)
            : null;
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
    Prompt current = prompt;
    for (int toolTurn = 0; toolTurn <= MAX_INTERNAL_TOOL_TURNS; toolTurn++) {
      ChatResponse response = callOnce(current);
      if (!response.hasToolCalls() || !internalToolExecutionEnabled(current.getOptions())) {
        context.setResponse(response);
        return response;
      }
      ToolExecutionResult toolResult = toolCallingManager.executeToolCalls(current, response);
      if (toolResult.returnDirect()) {
        ChatResponse direct =
            new ChatResponse(
                ToolExecutionResult.buildGenerations(toolResult), response.getMetadata());
        context.setResponse(direct);
        return direct;
      }
      if (toolTurn == MAX_INTERNAL_TOOL_TURNS) {
        throw new IllegalStateException(
            "tool-calling exceeded " + MAX_INTERNAL_TOOL_TURNS + " consecutive turns");
      }
      current = new Prompt(toolResult.conversationHistory(), current.getOptions());
    }
    throw new IllegalStateException("unreachable tool-calling state");
  }

  private ChatResponse callOnce(Prompt prompt) {
    Optional<String> completedToolResult = completedNeedleToolResult(prompt);
    if (completedToolResult.isPresent()) {
      return response(completedToolResult.orElseThrow(), null);
    }
    List<ToolSpec> declaredTools = declaredTools(prompt);
    requireQualifiedToolCalling(declaredTools);
    List<ToolSpec> tools = selectedTools(prompt, declaredTools);
    ModelPrompt rendered = render(prompt, tools);
    SamplingOptions requested = options(prompt, tools);
    GenerationOutput generated = generate(rendered, requested, tools);
    return tools.isEmpty()
        ? response(generated.text(), generated.usage())
        : toolAwareResponse(generated.text(), generated.usage());
  }

  /**
   * Needle 2 selects and structures actions; it is not a general response-synthesis model. Once
   * Spring has executed those actions, returning their actual result completes the advisor loop
   * instead of asking the selector to choose the same action again.
   */
  private Optional<String> completedNeedleToolResult(Prompt prompt) {
    if (template != ChatTemplate.NEEDLE2 || !(model instanceof ConstrainedTextGenerationModel)) {
      return Optional.empty();
    }
    List<org.springframework.ai.chat.messages.Message> messages = prompt.getInstructions();
    if (messages.isEmpty() || !(messages.getLast() instanceof ToolResponseMessage toolResponses)) {
      return Optional.empty();
    }
    List<String> results = new ArrayList<>();
    for (ToolResponseMessage.ToolResponse result : toolResponses.getResponses()) {
      results.add(renderToolResult(result));
    }
    if (results.isEmpty()) {
      return Optional.empty();
    }
    if (results.size() == 1) {
      return Optional.of(results.getFirst());
    }
    return Optional.of(
        rawToolResults ? "[" + String.join(",", results) + "]" : String.join("\n", results));
  }

  private String renderToolResult(ToolResponseMessage.ToolResponse result) {
    String serialized = Objects.requireNonNullElse(result.responseData(), "");
    TypedToolResultRenderer<?> renderer = toolResultRenderers.get(result.name());
    if (renderer != null) {
      return renderer.render(serialized);
    }
    if (rawToolResults) {
      return serialized;
    }
    throw new IllegalStateException(
        "Needle 2 selected tool '"
            + result.name()
            + "', but it cannot synthesize a conversational answer from the tool result. "
            + "Configure withToolResultRenderer(...) or explicitly opt into serialized results "
            + "withRawToolResults().");
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
    List<ToolSpec> declaredTools = declaredTools(prompt);
    requireQualifiedToolCalling(declaredTools);
    List<ToolSpec> tools = selectedTools(prompt, declaredTools);
    Flux<ChatResponse> response = streamOnce(prompt, tools);
    if (tools.isEmpty() || !internalToolExecutionEnabled(prompt.getOptions())) {
      return response;
    }
    return response.single().flatMapMany(result -> continueToolStream(prompt, result, 0));
  }

  private Flux<ChatResponse> continueToolStream(
      Prompt prompt, ChatResponse response, int toolTurn) {
    if (!response.hasToolCalls()) {
      return Flux.just(response);
    }
    ToolExecutionResult toolResult = toolCallingManager.executeToolCalls(prompt, response);
    if (toolResult.returnDirect()) {
      return Flux.just(
          new ChatResponse(
              ToolExecutionResult.buildGenerations(toolResult), response.getMetadata()));
    }
    if (toolTurn == MAX_INTERNAL_TOOL_TURNS) {
      return Flux.error(
          new IllegalStateException(
              "tool-calling exceeded " + MAX_INTERNAL_TOOL_TURNS + " consecutive turns"));
    }
    Prompt next = new Prompt(toolResult.conversationHistory(), prompt.getOptions());
    List<ToolSpec> tools = selectedTools(next, declaredTools(next));
    return streamOnce(next, tools)
        .single()
        .flatMapMany(result -> continueToolStream(next, result, toolTurn + 1));
  }

  private Flux<ChatResponse> streamOnce(Prompt prompt, List<ToolSpec> tools) {
    Optional<String> completedToolResult = completedNeedleToolResult(prompt);
    if (completedToolResult.isPresent()) {
      return Flux.just(response(completedToolResult.orElseThrow(), null));
    }
    ModelPrompt rendered = render(prompt, tools);
    SamplingOptions requested = options(prompt, tools);
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
    return ToolCallingChatOptions.builder()
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

  /**
   * Renders a completed Needle tool result as user-facing assistant text.
   *
   * <p>Spring AI serializes Java tool return values before adding them to conversation history.
   * This adapter deserializes the matching result to {@code resultType} and invokes {@code
   * renderer}; application code does not need to handle the intermediate JSON string. Configure
   * every tool that may be selected before sharing this adapter between requests.
   */
  public <T> ModelsSpringAiChatModel withToolResultRenderer(
      String toolName, Class<T> resultType, Function<? super T, String> renderer) {
    toolResultRenderers.put(
        requireToolName(toolName),
        new TypedToolResultRenderer<>(
            Objects.requireNonNull(resultType, "resultType"),
            Objects.requireNonNull(renderer, "renderer")));
    return this;
  }

  /**
   * Returns Needle tool results in Spring AI's serialized form.
   *
   * <p>This is an explicit machine-to-machine mode. Without this option or a registered typed
   * renderer, the adapter rejects a completed Needle result rather than presenting JSON as a
   * conversational answer.
   */
  public ModelsSpringAiChatModel withRawToolResults() {
    rawToolResults = true;
    return this;
  }

  /** Configures the user-facing response when Needle selects no applicable action. */
  public ModelsSpringAiChatModel withNoApplicableToolResponse(String response) {
    Objects.requireNonNull(response, "response");
    if (response.isBlank()) {
      throw new IllegalArgumentException("no-applicable-tool response must not be blank");
    }
    noApplicableToolResponse = response.strip();
    return this;
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

  private SamplingOptions options(Prompt prompt, List<ToolSpec> tools) {
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
    // Needle is an action selector, and its qualification contract is deterministic. Sampling can
    // turn an applicable action into a refusal even while the output remains schema-valid.
    if (template == ChatTemplate.NEEDLE2 && !tools.isEmpty()) {
      builder.temperature(0.0f);
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

  private static boolean internalToolExecutionEnabled(ChatOptions options) {
    if (!(options instanceof ToolCallingChatOptions toolOptions)) {
      return false;
    }
    try {
      Object enabled =
          ToolCallingChatOptions.class
              .getMethod("getInternalToolExecutionEnabled")
              .invoke(toolOptions);
      return enabled == null || Boolean.TRUE.equals(enabled);
    } catch (NoSuchMethodException ignored) {
      // Spring AI 2.0 removed provider-owned execution; ToolCallingAdvisor owns the loop.
      return false;
    } catch (ReflectiveOperationException failure) {
      throw new IllegalStateException("cannot read Spring AI tool-execution policy", failure);
    }
  }

  private List<ToolSpec> selectedTools(Prompt prompt, List<ToolSpec> tools) {
    if (toolSelector == null || tools.size() <= ToolSpecSelector.DEFAULT_TOOL_LIMIT) {
      return tools;
    }
    return toolSelector.select(latestUserText(prompt), tools);
  }

  private void requireQualifiedToolCalling(List<ToolSpec> tools) {
    if (tools.isEmpty()
        || qualifiedCapabilities == null
        || qualifiedCapabilities.contains("tool-calling")) {
      return;
    }
    throw new IllegalStateException(
        "Model "
            + resolvedModelName()
            + " is not qualified for tool-calling; qualified capabilities are "
            + qualifiedCapabilities);
  }

  private static String latestUserText(Prompt prompt) {
    List<org.springframework.ai.chat.messages.Message> messages = prompt.getInstructions();
    for (int index = messages.size() - 1; index >= 0; index--) {
      if (messages.get(index) instanceof UserMessage userMessage) {
        return userMessage.getText();
      }
    }
    throw new IllegalArgumentException("tool retrieval requires a text user message");
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
        messages.add(ChatMessage.tool(response.name(), response.responseData()));
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
      if (template == ChatTemplate.NEEDLE2 && isEmptyNeedleToolSelection(output)) {
        return response(noApplicableToolResponse, usage);
      }
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

  private static String requireToolName(String toolName) {
    Objects.requireNonNull(toolName, "toolName");
    if (toolName.isBlank()) {
      throw new IllegalArgumentException("toolName must not be blank");
    }
    return toolName.strip();
  }

  private static boolean isEmptyNeedleToolSelection(String output) {
    String sectionStart = ChatTemplate.NEEDLE2.toolSyntax().sectionStart();
    String sectionEnd = ChatTemplate.NEEDLE2.toolSyntax().sectionEnd();
    int start = output.indexOf(sectionStart);
    if (start < 0) {
      return false;
    }
    int payloadStart = start + sectionStart.length();
    int end = output.indexOf(sectionEnd, payloadStart);
    return end >= payloadStart && output.substring(payloadStart, end).strip().equals("[]");
  }

  private record TypedToolResultRenderer<T>(
      Class<T> resultType, Function<? super T, String> renderer) {

    private String render(String serialized) {
      try {
        T result = JSON.readValue(serialized, resultType);
        return Objects.requireNonNull(renderer.apply(result), "tool result renderer returned null");
      } catch (JsonProcessingException failure) {
        throw new IllegalArgumentException(
            "cannot deserialize tool result as " + resultType.getName(), failure);
      }
    }
  }

  private record GenerationOutput(String text, GenerationUsage usage) {}
}
