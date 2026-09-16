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

import com.integrallis.models.api.AuxiliaryTextGenerationModel;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.StopReason;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.ActivatedToolModel;
import com.integrallis.models.runtime.ConstrainedTextGenerationModel;
import com.integrallis.models.runtime.RuntimeTextGenerationModel;
import com.integrallis.models.runtime.SharedToolTurn;
import com.integrallis.models.runtime.TokenConstraint;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.ToolCallScanner;
import com.integrallis.models.runtime.chat.ToolSpecSelector;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.chat.response.StreamingHandle;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

/** LangChain4j {@link StreamingChatModel} backed by the Models runtime generation loop. */
public final class ModelsStreamingChatModel implements StreamingChatModel, AutoCloseable {

  private final TextGenerationModel model;
  private final ChatTemplate template;
  private final SamplingOptions defaults;
  private final ToolSpecSelector toolSelector;
  private final ActivatedTurnRegistry activatedTurns = new ActivatedTurnRegistry();

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
    this.toolSelector =
        model instanceof AuxiliaryTextGenerationModel auxiliary
                && auxiliary.supportsContrastiveEncoding()
            ? new ToolSpecSelector(auxiliary)
            : null;
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
    List<ToolSpec> tools = selectedTools(request, LangChain4jChatRequestMapper.tools(request));
    boolean toolsDeclared = !tools.isEmpty();
    ModelPrompt prompt = LangChain4jChatRequestMapper.prompt(request, template, tools);
    SamplingOptions requested = LangChain4jChatRequestMapper.options(request, defaults);
    if (model instanceof ActivatedToolModel activatedModel && toolsDeclared) {
      streamActivatedTurn(request, handler, activatedModel, prompt, requested, tools);
      return;
    }
    CancellationHandle cancellation = new CancellationHandle();
    PartialResponseContext partialContext = new PartialResponseContext(cancellation);
    TokenStream stream =
        new TokenStream() {
          @Override
          public void onToken(String token) {
            if (!terminalSignalSent.get() && !cancellation.isCancelled()) {
              accumulated.append(token);
              if (!toolsDeclared) {
                handler.onPartialResponse(new PartialResponse(token), partialContext);
              }
            }
          }

          @Override
          public boolean isCancelled() {
            return cancellation.isCancelled();
          }

          @Override
          public void onComplete() {
            complete(null, null);
          }

          @Override
          public void onComplete(GenerationUsage usage) {
            complete(usage, null);
          }

          @Override
          public void onComplete(GenerationUsage usage, StopReason stopReason) {
            complete(usage, stopReason);
          }

          private void complete(GenerationUsage usage, StopReason stopReason) {
            // LangChain4j's contract is that a cancelled stream receives no further callbacks.
            if (terminalSignalSent.compareAndSet(false, true) && !cancellation.isCancelled()) {
              handler.onCompleteResponse(
                  completed(accumulated.toString(), toolsDeclared, usage, stopReason));
            }
          }

          @Override
          public void onError(Throwable failure) {
            if (terminalSignalSent.compareAndSet(false, true) && !cancellation.isCancelled()) {
              handler.onError(failure);
            }
          }
        };

    try {
      Optional<TokenConstraint> constraint = toolConstraint(tools);
      if (constraint.isPresent()) {
        ((ConstrainedTextGenerationModel) model)
            .generate(prompt, requested, stream, constraint.get());
      } else {
        model.generate(prompt, requested, stream);
      }
    } catch (RuntimeException | Error failure) {
      if (terminalSignalSent.compareAndSet(false, true)) {
        handler.onError(failure);
      } else {
        throw failure;
      }
    }
  }

  private void streamActivatedTurn(
      ChatRequest request,
      StreamingChatResponseHandler handler,
      ActivatedToolModel activatedModel,
      ModelPrompt prompt,
      SamplingOptions options,
      List<ToolSpec> tools) {
    ActivatedTurnRegistry.PendingTurn pending;
    try {
      pending = activatedTurns.take(request);
    } catch (RuntimeException | Error failure) {
      handler.onError(failure);
      return;
    }
    if (pending != null) {
      SharedToolTurn previous = pending.turn;
      SharedToolTurn next;
      try {
        next = previous.continueToolSelection(prompt);
      } catch (RuntimeException | Error failure) {
        handler.onError(failure);
        return;
      } finally {
        previous.close();
      }
      streamActivatedSelection(next, prompt, options, tools, handler);
      return;
    }

    streamActivatedSelection(activatedModel.openToolTurn(prompt), prompt, options, tools, handler);
  }

  private void streamActivatedSelection(
      SharedToolTurn openedTurn,
      ModelPrompt prompt,
      SamplingOptions options,
      List<ToolSpec> tools,
      StreamingChatResponseHandler handler) {
    SharedToolTurn turn = openedTurn;
    boolean retained = false;
    try {
      TokenConstraint constraint = toolConstraint(tools).orElseGet(TokenConstraint::unrestricted);
      String output = turn.generateToolCall(options, constraint);
      GenerationUsage usage = turn.toolMetrics().available() ? turn.toolMetrics().usage() : null;
      ChatResponse response =
          completed(output, true, usage, turn.toolMetrics().stopReason().orElse(null));
      if (response.aiMessage().hasToolExecutionRequests()) {
        List<ToolExecutionRequest> requests =
            activatedTurns.retain(turn, response.aiMessage().toolExecutionRequests());
        retained = true;
        handler.onCompleteResponse(toolResponse(response.aiMessage().text(), requests, usage));
        return;
      }
      SharedToolTurn baseTurn = turn;
      turn = null;
      streamActivatedBase(baseTurn, prompt, options, usage, handler);
    } catch (RuntimeException | Error failure) {
      handler.onError(failure);
    } finally {
      if (!retained && turn != null) {
        turn.close();
      }
    }
  }

  private void streamActivatedBase(
      SharedToolTurn turn,
      ModelPrompt prompt,
      SamplingOptions options,
      GenerationUsage selectionUsage,
      StreamingChatResponseHandler handler) {
    StringBuilder accumulated = new StringBuilder();
    AtomicBoolean terminal = new AtomicBoolean();
    try {
      turn.generateBaseResponse(
          prompt,
          options,
          new TokenStream() {
            @Override
            public void onToken(String token) {
              if (!terminal.get()) {
                accumulated.append(token);
                handler.onPartialResponse(token);
              }
            }

            @Override
            public void onComplete() {
              complete(null, null);
            }

            @Override
            public void onComplete(GenerationUsage usage) {
              complete(usage, null);
            }

            @Override
            public void onComplete(GenerationUsage usage, StopReason stopReason) {
              complete(usage, stopReason);
            }

            private void complete(GenerationUsage usage, StopReason stopReason) {
              if (terminal.compareAndSet(false, true)) {
                try {
                  handler.onCompleteResponse(
                      completed(
                          accumulated.toString(),
                          false,
                          combineUsage(selectionUsage, usage),
                          stopReason));
                } finally {
                  turn.close();
                }
              }
            }

            @Override
            public void onError(Throwable failure) {
              if (terminal.compareAndSet(false, true)) {
                try {
                  handler.onError(failure);
                } finally {
                  turn.close();
                }
              }
            }
          });
    } catch (RuntimeException | Error failure) {
      if (terminal.compareAndSet(false, true)) {
        try {
          handler.onError(failure);
        } finally {
          turn.close();
        }
      } else {
        throw failure;
      }
    }
  }

  /** Builds the terminal response, recovering any tool calls the model produced. */
  private ChatResponse completed(
      String output, boolean toolsDeclared, GenerationUsage usage, StopReason stopReason) {
    ToolCallScanner.Result scan =
        toolsDeclared
            ? ToolCallScanner.scan(output, template.toolSyntax())
            : ToolCallScanner.Result.plainText(output);
    if (!scan.hasCalls()) {
      var response =
          ChatResponse.builder()
              .aiMessage(AiMessage.from(scan.content()))
              .finishReason(LangChain4jFinishReasons.of(stopReason))
              .modelName(model.modelName());
      addUsage(response, usage);
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
    var response =
        ChatResponse.builder()
            .aiMessage(new AiMessage(scan.content(), requests))
            .finishReason(FinishReason.TOOL_EXECUTION)
            .modelName(model.modelName());
    addUsage(response, usage);
    return response.build();
  }

  private ChatResponse toolResponse(
      String content, List<ToolExecutionRequest> requests, GenerationUsage usage) {
    var response =
        ChatResponse.builder()
            .aiMessage(new AiMessage(content, requests))
            .finishReason(FinishReason.TOOL_EXECUTION)
            .modelName(model.modelName());
    addUsage(response, usage);
    return response.build();
  }

  private static void addUsage(ChatResponse.Builder response, GenerationUsage usage) {
    if (usage != null) {
      response.tokenUsage(
          new TokenUsage(usage.promptTokens(), usage.completionTokens(), usage.totalTokens()));
    }
  }

  private static GenerationUsage combineUsage(GenerationUsage first, GenerationUsage second) {
    if (first == null) {
      return second;
    }
    if (second == null) {
      return first;
    }
    return new GenerationUsage(
        Math.addExact(first.promptTokens(), second.promptTokens()),
        Math.addExact(first.completionTokens(), second.completionTokens()));
  }

  private Optional<TokenConstraint> toolConstraint(List<ToolSpec> tools) {
    if (tools.isEmpty() || !(model instanceof ConstrainedTextGenerationModel constrainedModel)) {
      return Optional.empty();
    }
    return LangChain4jToolCallConstraint.compile(
        constrainedModel.tokenizer(),
        template.toolSyntax(),
        tools,
        model instanceof ActivatedToolModel activatedModel
            ? activatedModel.toolAbstentionOutputs()
            : List.of());
  }

  private List<ToolSpec> selectedTools(ChatRequest request, List<ToolSpec> tools) {
    if (toolSelector == null || tools.size() <= ToolSpecSelector.DEFAULT_TOOL_LIMIT) {
      return tools;
    }
    return toolSelector.select(LangChain4jChatRequestMapper.latestUserText(request), tools);
  }

  @Override
  public void close() {
    activatedTurns.close();
    model.close();
  }

  /** LangChain4j cancellation handle polled by the generation loop after each token. */
  private static final class CancellationHandle implements StreamingHandle {
    private volatile boolean cancelled;

    @Override
    public void cancel() {
      cancelled = true;
    }

    @Override
    public boolean isCancelled() {
      return cancelled;
    }
  }
}
