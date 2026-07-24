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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.runtime.GenerationLoop;
import java.util.List;
import java.util.Objects;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

/** Spring AI {@link ChatModel} backed by the Models runtime generation loop. */
public final class ModelsSpringAiChatModel implements ChatModel {
  private final SamplingOptions defaults;
  private final GenerationLoop generationLoop;

  public ModelsSpringAiChatModel(InferenceBackend backend) {
    this(backend, SamplingOptions.builder().build());
  }

  public ModelsSpringAiChatModel(InferenceBackend backend, SamplingOptions defaults) {
    Objects.requireNonNull(backend, "backend");
    this.defaults = Objects.requireNonNull(defaults, "defaults");
    this.generationLoop = new GenerationLoop(backend);
  }

  @Override
  public ChatResponse call(Prompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    String output = generationLoop.generate(prompt.getContents(), options(prompt));
    return response(output);
  }

  @Override
  public Flux<ChatResponse> stream(Prompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    return Flux.create(
        sink ->
            generationLoop.generate(
                prompt.getContents(),
                options(prompt),
                new TokenStream() {
                  @Override
                  public void onToken(String token) {
                    sink.next(response(token));
                  }

                  @Override
                  public void onComplete() {
                    sink.complete();
                  }

                  @Override
                  public void onError(Throwable failure) {
                    sink.error(failure);
                  }
                }));
  }

  @Override
  public ChatOptions getOptions() {
    return ChatOptions.builder()
        .temperature((double) defaults.temperature())
        .topP((double) defaults.topP())
        .topK(defaults.topK())
        .maxTokens(defaults.maxTokens())
        .build();
  }

  private SamplingOptions options(Prompt prompt) {
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
    }
    return builder.build();
  }

  private static ChatResponse response(String text) {
    return new ChatResponse(List.of(new Generation(new AssistantMessage(text))));
  }
}
