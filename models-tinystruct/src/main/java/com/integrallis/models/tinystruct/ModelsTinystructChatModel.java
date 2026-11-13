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
package com.integrallis.models.tinystruct;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelGenerationException;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.StopReason;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.RuntimeTextGenerationModel;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.ToolCallScanner;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import org.tinystruct.data.component.Builder;
import org.tinystruct.data.component.Builders;

/**
 * In-process {@link TinystructChatModel} backed by the Models runtime generation loop.
 *
 * <p>Formats role-aware {@link ChatMessage} histories and optional {@link ToolSpec} declarations
 * through a {@link ChatTemplate}, scans the output for structured tool calls when tools are
 * declared, and returns results as tinystruct {@link Builder} and {@link Builders} instances.
 */
public final class ModelsTinystructChatModel implements TinystructChatModel, AutoCloseable {

  private final TextGenerationModel model;
  private final ChatTemplate template;
  private final SamplingOptions defaults;

  /** Wraps an {@link InferenceBackend} with {@link ChatTemplate#RAW} and default sampling. */
  public ModelsTinystructChatModel(InferenceBackend backend) {
    this(new RuntimeTextGenerationModel(backend));
  }

  /** Wraps an {@link InferenceBackend} with {@link ChatTemplate#RAW} and explicit defaults. */
  public ModelsTinystructChatModel(InferenceBackend backend, SamplingOptions defaults) {
    this(new RuntimeTextGenerationModel(backend), defaults);
  }

  /** Wraps an {@link InferenceBackend} with the given {@link ChatTemplate} and defaults. */
  public ModelsTinystructChatModel(
      InferenceBackend backend, ChatTemplate template, SamplingOptions defaults) {
    this(new RuntimeTextGenerationModel(backend), template, defaults);
  }

  /** Wraps a {@link TextGenerationModel} with {@link ChatTemplate#RAW} and default sampling. */
  public ModelsTinystructChatModel(TextGenerationModel model) {
    this(model, SamplingOptions.builder().build());
  }

  /** Wraps a {@link TextGenerationModel} with {@link ChatTemplate#RAW} and explicit defaults. */
  public ModelsTinystructChatModel(TextGenerationModel model, SamplingOptions defaults) {
    this(model, ChatTemplate.RAW, defaults);
  }

  /** Wraps a {@link TextGenerationModel} with the given {@link ChatTemplate} and defaults. */
  public ModelsTinystructChatModel(
      TextGenerationModel model, ChatTemplate template, SamplingOptions defaults) {
    this.model = Objects.requireNonNull(model, "model");
    this.template = Objects.requireNonNull(template, "template");
    this.defaults = Objects.requireNonNull(defaults, "defaults");
  }

  /** Returns the execution diagnostics of the underlying backend. */
  public BackendDiagnostics diagnostics() {
    return model.diagnostics();
  }

  /** Returns whether the configured {@link ChatTemplate} supports schema-aware tool calling. */
  public boolean supportsTools() {
    return template.canParseToolCallsWithSchemas();
  }

  @Override
  public Builder chat(List<ChatMessage> messages, SamplingOptions options, List<ToolSpec> tools) {
    return execute(messages, options, tools, ignored -> {});
  }

  @Override
  public Builder stream(
      List<ChatMessage> messages,
      SamplingOptions options,
      List<ToolSpec> tools,
      Consumer<String> onToken) {
    Objects.requireNonNull(onToken, "onToken");
    return execute(messages, options, tools, onToken);
  }

  private Builder execute(
      List<ChatMessage> messages,
      SamplingOptions options,
      List<ToolSpec> tools,
      Consumer<String> onToken) {
    Objects.requireNonNull(messages, "messages");
    if (messages.isEmpty()) {
      throw new IllegalArgumentException("messages must not be empty");
    }
    List<ToolSpec> declaredTools = tools == null ? List.of() : List.copyOf(tools);
    SamplingOptions effectiveOptions = options != null ? options : defaults;
    ModelPrompt prompt =
        declaredTools.isEmpty()
            ? template.render(messages)
            : template.render(messages, declaredTools);

    StringBuilder output = new StringBuilder();
    AtomicReference<GenerationUsage> usage = new AtomicReference<>();
    AtomicReference<StopReason> stopReason = new AtomicReference<>();
    AtomicReference<Throwable> failure = new AtomicReference<>();

    TokenStream stream =
        new TokenStream() {
          @Override
          public void onToken(String token) {
            output.append(token);
            onToken.accept(token);
          }

          @Override
          public void onComplete() {}

          @Override
          public void onComplete(GenerationUsage completedUsage) {
            usage.set(completedUsage);
          }

          @Override
          public void onComplete(GenerationUsage completedUsage, StopReason completedReason) {
            usage.set(completedUsage);
            stopReason.set(completedReason);
          }

          @Override
          public void onError(Throwable generationFailure) {
            failure.compareAndSet(null, generationFailure);
          }
        };

    model.generate(prompt, effectiveOptions, stream);
    throwFailure(failure.get());

    String rawText = output.toString();
    ToolCallScanner.Result scan =
        !declaredTools.isEmpty()
            ? ToolCallScanner.scan(rawText, template.toolSyntax(), declaredTools)
            : ToolCallScanner.Result.plainText(rawText);

    Builder response = new Builder();
    response.put("text", scan.content());
    response.put("content", scan.content());
    response.put("raw_output", rawText);
    response.put("model", model.modelName());
    response.put(
        "finish_reason",
        scan.hasCalls()
            ? "TOOL_CALLS"
            : stopReason.get() != null ? stopReason.get().name() : StopReason.EOS.name());

    GenerationUsage recordedUsage = usage.get();
    if (recordedUsage != null) {
      Builder usageBuilder = new Builder();
      usageBuilder.put("prompt_tokens", recordedUsage.promptTokens());
      usageBuilder.put("completion_tokens", recordedUsage.completionTokens());
      usageBuilder.put("total_tokens", recordedUsage.totalTokens());
      response.put("usage", usageBuilder);
    }

    Builders toolCalls = new Builders();
    for (ToolCall call : scan.toolCalls()) {
      Builder callBuilder = new Builder();
      callBuilder.put("id", call.id());
      callBuilder.put("name", call.name());
      callBuilder.put("arguments", call.argumentsJson());
      toolCalls.add(callBuilder);
    }
    response.put("tool_calls", toolCalls);
    return response;
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

  @Override
  public void close() {
    model.close();
  }
}
