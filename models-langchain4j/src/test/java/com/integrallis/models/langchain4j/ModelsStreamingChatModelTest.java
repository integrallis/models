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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.runtime.chat.ChatTemplate;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ModelsStreamingChatModelTest {

  @Test
  void streamsDeltasAndCompletesWithTheAccumulatedResponse() {
    RecordingStreamingModel delegate = new RecordingStreamingModel(List.of("mapped ", "answer"));
    SamplingOptions defaults =
        SamplingOptions.builder()
            .temperature(0.8f)
            .topP(0.7f)
            .topK(20)
            .maxTokens(100)
            .repetitionPenalty(1.2f)
            .seed(42L)
            .stopSequences(List.of("DEFAULT_STOP"))
            .build();
    ModelsStreamingChatModel model =
        new ModelsStreamingChatModel(delegate, ChatTemplate.CHATML, defaults);
    ChatRequest request =
        ChatRequest.builder()
            .messages(List.of(SystemMessage.from("system"), UserMessage.from("question")))
            .temperature(0.2)
            .topP(0.3)
            .topK(7)
            .maxOutputTokens(19)
            .stopSequences(List.of("REQUEST_STOP"))
            .build();
    List<String> partials = new ArrayList<>();
    AtomicReference<ChatResponse> completed = new AtomicReference<>();
    AtomicReference<Throwable> failed = new AtomicReference<>();

    model.doChat(request, handler(partials, completed, failed));

    assertThat(partials).containsExactly("mapped ", "answer");
    assertThat(completed.get().aiMessage().text()).isEqualTo("mapped answer");
    assertThat(completed.get().modelName()).isEqualTo("RecordingStreamingModel");
    assertThat(failed).hasValue(null);
    assertThat(delegate.prompt)
        .isEqualTo(
            """
            <|im_start|>system
            system<|im_end|>
            <|im_start|>user
            question<|im_end|>
            <|im_start|>assistant
            """);
    assertThat(delegate.options.temperature()).isEqualTo(0.2f);
    assertThat(delegate.options.topP()).isEqualTo(0.3f);
    assertThat(delegate.options.topK()).isEqualTo(7);
    assertThat(delegate.options.maxTokens()).isEqualTo(19);
    assertThat(delegate.options.repetitionPenalty()).isEqualTo(1.2f);
    assertThat(delegate.options.seed()).isEqualTo(42L);
    assertThat(delegate.options.stopSequences()).containsExactly("REQUEST_STOP");
    assertThat(model.diagnostics().backend()).isEqualTo("stream-recording");
  }

  @Test
  void forwardsGenerationFailureWithoutCompleting() {
    IllegalStateException failure = new IllegalStateException("generation failed");
    TextGenerationModel delegate =
        new RecordingStreamingModel(List.of()) {
          @Override
          public void generate(String prompt, SamplingOptions options, TokenStream stream) {
            stream.onError(failure);
          }
        };
    ModelsStreamingChatModel model = new ModelsStreamingChatModel(delegate);
    AtomicReference<ChatResponse> completed = new AtomicReference<>();
    AtomicReference<Throwable> failed = new AtomicReference<>();

    model.doChat(
        ChatRequest.builder().messages(UserMessage.from("question")).build(),
        handler(new ArrayList<>(), completed, failed));

    assertThat(failed).hasValue(failure);
    assertThat(completed).hasValue(null);
  }

  @Test
  void rejectsNullDependenciesAndRequests() {
    RecordingStreamingModel delegate = new RecordingStreamingModel(List.of("answer"));
    assertThatNullPointerException()
        .isThrownBy(() -> new ModelsStreamingChatModel((TextGenerationModel) null));
    assertThatNullPointerException().isThrownBy(() -> new ModelsStreamingChatModel(delegate, null));
    assertThatNullPointerException()
        .isThrownBy(
            () ->
                new ModelsStreamingChatModel(
                    delegate, (ChatTemplate) null, SamplingOptions.builder().build()));
    ModelsStreamingChatModel model = new ModelsStreamingChatModel(delegate);
    assertThatNullPointerException()
        .isThrownBy(
            () ->
                model.doChat(
                    null,
                    handler(new ArrayList<>(), new AtomicReference<>(), new AtomicReference<>())));
    assertThatNullPointerException()
        .isThrownBy(
            () ->
                model.doChat(
                    ChatRequest.builder().messages(UserMessage.from("question")).build(), null));
  }

  private static StreamingChatResponseHandler handler(
      List<String> partials,
      AtomicReference<ChatResponse> completed,
      AtomicReference<Throwable> failed) {
    return new StreamingChatResponseHandler() {
      @Override
      public void onPartialResponse(String partialResponse) {
        partials.add(partialResponse);
      }

      @Override
      public void onCompleteResponse(ChatResponse response) {
        completed.set(response);
      }

      @Override
      public void onError(Throwable error) {
        failed.set(error);
      }
    };
  }

  private static class RecordingStreamingModel implements TextGenerationModel {
    private final List<String> tokens;
    private String prompt;
    private SamplingOptions options;

    private RecordingStreamingModel(List<String> tokens) {
      this.tokens = tokens;
    }

    @Override
    public String modelName() {
      return "RecordingStreamingModel";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("stream-recording");
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      this.prompt = prompt;
      this.options = options;
      tokens.forEach(stream::onToken);
      stream.onComplete();
    }
  }
}
