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

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.StopReason;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.runtime.chat.ChatTemplate;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** The runtime's typed stop reason reaches LangChain4j as a {@link FinishReason}. */
@Tag("unit")
class LangChain4jStopReasonTest {

  private static final ChatRequest QUESTION =
      ChatRequest.builder().messages(UserMessage.from("question")).build();

  @ParameterizedTest
  @CsvSource({
    "EOS, STOP",
    "STOP_SEQUENCE, STOP",
    "CONSTRAINT_COMPLETE, STOP",
    "MAX_TOKENS, LENGTH",
    "REPETITION_LOOP, OTHER",
    "CANCELLED, OTHER"
  })
  void chatModelMapsEveryStopReason(StopReason stopReason, FinishReason expected) {
    ModelsChatModel model =
        new ModelsChatModel(new ScriptedModel(List.of("an", "swer"), stopReason));

    ChatResponse response = model.chat(QUESTION);

    assertThat(response.aiMessage().text()).isEqualTo("answer");
    assertThat(response.finishReason()).isEqualTo(expected);
  }

  @ParameterizedTest
  @CsvSource({
    "EOS, STOP",
    "STOP_SEQUENCE, STOP",
    "CONSTRAINT_COMPLETE, STOP",
    "MAX_TOKENS, LENGTH",
    "REPETITION_LOOP, OTHER",
    "CANCELLED, OTHER"
  })
  void streamingChatModelMapsEveryStopReason(StopReason stopReason, FinishReason expected) {
    ModelsStreamingChatModel model =
        new ModelsStreamingChatModel(new ScriptedModel(List.of("an", "swer"), stopReason));
    AtomicReference<ChatResponse> completed = new AtomicReference<>();

    model.chat(QUESTION, completingHandler(completed));

    assertThat(completed.get().aiMessage().text()).isEqualTo("answer");
    assertThat(completed.get().finishReason()).isEqualTo(expected);
  }

  @Test
  void engineWithoutStopReasonsKeepsTheFinishReasonAbsent() {
    ModelsChatModel model = new ModelsChatModel(new ScriptedModel(List.of("answer"), null));

    assertThat(model.chat(QUESTION).finishReason()).isNull();
  }

  @Test
  void toolCallsStillFinishWithToolExecutionWhateverTheStopReason() {
    ToolSpecification weather =
        ToolSpecification.builder()
            .name("get_weather")
            .description("Look up the forecast")
            .parameters(
                JsonObjectSchema.builder()
                    .addStringProperty("city", "the city")
                    .required("city")
                    .build())
            .build();
    ModelsChatModel model =
        new ModelsChatModel(
            new ScriptedModel(
                List.of(
                    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":"
                        + " \"Austin\"}}\n</tool_call>"),
                StopReason.EOS),
            ChatTemplate.CHATML,
            SamplingOptions.builder().build());

    ChatResponse response =
        model.chat(
            ChatRequest.builder()
                .messages(UserMessage.from("weather?"))
                .toolSpecifications(weather)
                .build());

    assertThat(response.aiMessage().hasToolExecutionRequests()).isTrue();
    assertThat(response.finishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
  }

  @Test
  void cancellingTheStreamingHandleStopsGenerationWithoutFurtherCallbacks() {
    ScriptedModel delegate = new ScriptedModel(List.of("one", "two", "three"), StopReason.EOS);
    ModelsStreamingChatModel model = new ModelsStreamingChatModel(delegate);
    List<String> partials = new ArrayList<>();
    AtomicReference<ChatResponse> completed = new AtomicReference<>();
    AtomicReference<Throwable> failed = new AtomicReference<>();

    model.chat(
        QUESTION,
        new StreamingChatResponseHandler() {
          @Override
          public void onPartialResponse(PartialResponse partial, PartialResponseContext context) {
            partials.add(partial.text());
            context.streamingHandle().cancel();
            assertThat(context.streamingHandle().isCancelled()).isTrue();
          }

          @Override
          public void onCompleteResponse(ChatResponse response) {
            completed.set(response);
          }

          @Override
          public void onError(Throwable error) {
            failed.set(error);
          }
        });

    assertThat(partials).containsExactly("one");
    assertThat(delegate.emitted).isEqualTo(1);
    assertThat(delegate.reported).isEqualTo(StopReason.CANCELLED);
    assertThat(completed.get()).isNull();
    assertThat(failed.get()).isNull();
  }

  private static StreamingChatResponseHandler completingHandler(
      AtomicReference<ChatResponse> completed) {
    return new StreamingChatResponseHandler() {
      @Override
      public void onPartialResponse(String partialResponse) {}

      @Override
      public void onCompleteResponse(ChatResponse response) {
        completed.set(response);
      }

      @Override
      public void onError(Throwable error) {
        throw new AssertionError(error);
      }
    };
  }

  /** Streams fixed fragments, honours cancellation, and reports a fixed stop reason. */
  private static final class ScriptedModel implements TextGenerationModel {
    private final List<String> tokens;
    private final StopReason stopReason;
    private int emitted;
    private StopReason reported;

    private ScriptedModel(List<String> tokens, StopReason stopReason) {
      this.tokens = tokens;
      this.stopReason = stopReason;
    }

    @Override
    public String modelName() {
      return "scripted";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("scripted");
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      for (String token : tokens) {
        stream.onToken(token);
        emitted++;
        if (stream.isCancelled()) {
          reported = StopReason.CANCELLED;
          stream.onComplete(new GenerationUsage(1, emitted), StopReason.CANCELLED);
          return;
        }
      }
      if (stopReason == null) {
        stream.onComplete(new GenerationUsage(1, emitted));
        return;
      }
      reported = stopReason;
      stream.onComplete(new GenerationUsage(1, emitted), stopReason);
    }
  }
}
