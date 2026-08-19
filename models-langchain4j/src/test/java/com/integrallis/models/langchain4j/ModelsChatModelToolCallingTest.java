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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.runtime.ConstrainedTextGenerationModel;
import com.integrallis.models.runtime.TokenConstraint;
import com.integrallis.models.runtime.chat.ChatTemplate;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ModelsChatModelToolCallingTest {

  private static final ToolSpecification WEATHER =
      ToolSpecification.builder()
          .name("get_weather")
          .description("Look up the forecast")
          .parameters(
              JsonObjectSchema.builder()
                  .addStringProperty("city", "the city")
                  .required("city")
                  .build())
          .build();

  private static final ToolSpecification MODE =
      ToolSpecification.builder()
          .name("set_mode")
          .description("Set the HVAC mode")
          .parameters(
              JsonObjectSchema.builder()
                  .addProperty(
                      "mode", JsonEnumSchema.builder().enumValues(List.of("cool", "heat")).build())
                  .required("mode")
                  .build())
          .build();

  private static final class ScriptedModel implements TextGenerationModel {
    private final String completion;
    private final AtomicReference<String> lastPrompt = new AtomicReference<>();

    ScriptedModel(String completion) {
      this.completion = completion;
    }

    @Override
    public String modelName() {
      return "ScriptedModel";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("scripted");
    }

    @Override
    public String generate(String prompt, SamplingOptions options) {
      lastPrompt.set(prompt);
      return completion;
    }

    @Override
    public String generate(ModelPrompt prompt, SamplingOptions options) {
      lastPrompt.set(prompt.text());
      return completion;
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      lastPrompt.set(prompt);
      stream.onToken(completion);
      stream.onComplete();
    }

    @Override
    public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
      lastPrompt.set(prompt.text());
      stream.onToken(completion);
      stream.onComplete();
    }

    String lastPrompt() {
      return lastPrompt.get();
    }
  }

  private static ModelsChatModel chatModel(ScriptedModel model, ChatTemplate template) {
    return new ModelsChatModel(model, template, SamplingOptions.builder().build());
  }

  @Nested
  static class Declaration {

    @Test
    void serializesToolSpecificationsIntoThePrompt() {
      // LangChain4j models parameters as a typed tree, so the adapter must flatten it.
      ScriptedModel model = new ScriptedModel("no call");
      ChatRequest request =
          ChatRequest.builder()
              .messages(UserMessage.from("weather?"))
              .toolSpecifications(WEATHER)
              .build();

      chatModel(model, ChatTemplate.CHATML).chat(request);

      assertThat(model.lastPrompt()).contains("# Tools");
      assertThat(model.lastPrompt()).contains("get_weather");
      assertThat(model.lastPrompt()).contains("\"type\":\"object\"");
      assertThat(model.lastPrompt()).contains("\"city\"");
      assertThat(model.lastPrompt()).contains("\"required\":[\"city\"]");
    }

    @Test
    void withoutToolsRendersThePlainEnvelope() {
      ScriptedModel model = new ScriptedModel("hi");
      ChatRequest request = ChatRequest.builder().messages(UserMessage.from("hello")).build();

      chatModel(model, ChatTemplate.CHATML).chat(request);

      assertThat(model.lastPrompt()).doesNotContain("# Tools");
    }

    @Test
    void refusesToolsWhenTheTemplateCannotExpressThem() {
      ScriptedModel model = new ScriptedModel("x");
      ChatRequest request =
          ChatRequest.builder().messages(UserMessage.from("q")).toolSpecifications(WEATHER).build();

      assertThatThrownBy(() -> chatModel(model, ChatTemplate.GEMMA).chat(request))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("gemma");
    }
  }

  @Nested
  static class Recovery {

    @Test
    void surfacesToolExecutionRequestsAndTheFinishReason() {
      // Unlike Spring AI 2.0, LangChain4j needs FinishReason.TOOL_EXECUTION set explicitly.
      ScriptedModel model =
          new ScriptedModel(
              "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Austin\"}}\n</tool_call>");
      ChatRequest request =
          ChatRequest.builder()
              .messages(UserMessage.from("weather?"))
              .toolSpecifications(WEATHER)
              .build();

      ChatResponse response = chatModel(model, ChatTemplate.CHATML).chat(request);

      assertThat(response.aiMessage().hasToolExecutionRequests()).isTrue();
      assertThat(response.finishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
      assertThat(response.aiMessage().toolExecutionRequests())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("get_weather");
                assertThat(call.arguments()).isEqualTo("{\"city\": \"Austin\"}");
                assertThat(call.id()).isEqualTo("000000000");
              });
    }

    @Test
    void keepsProseAlongsideTheCall() {
      ScriptedModel model =
          new ScriptedModel(
              "Checking.\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>");
      ChatRequest request =
          ChatRequest.builder()
              .messages(UserMessage.from("weather?"))
              .toolSpecifications(WEATHER)
              .build();

      ChatResponse response = chatModel(model, ChatTemplate.CHATML).chat(request);

      assertThat(response.aiMessage().text()).isEqualTo("Checking.");
      assertThat(response.aiMessage().hasToolExecutionRequests()).isTrue();
    }

    @Test
    void plainAnswersCarryNoToolRequests() {
      ScriptedModel model = new ScriptedModel("It is 88 degrees.");
      ChatRequest request =
          ChatRequest.builder()
              .messages(UserMessage.from("weather?"))
              .toolSpecifications(WEATHER)
              .build();

      ChatResponse response = chatModel(model, ChatTemplate.CHATML).chat(request);

      assertThat(response.aiMessage().hasToolExecutionRequests()).isFalse();
      assertThat(response.aiMessage().text()).isEqualTo("It is 88 degrees.");
      assertThat(response.finishReason()).isNotEqualTo(FinishReason.TOOL_EXECUTION);
    }

    @Test
    void doesNotScanForCallsWhenNoToolsWereDeclared() {
      ScriptedModel model =
          new ScriptedModel("<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>");
      ChatRequest request = ChatRequest.builder().messages(UserMessage.from("hi")).build();

      ChatResponse response = chatModel(model, ChatTemplate.CHATML).chat(request);

      assertThat(response.aiMessage().hasToolExecutionRequests()).isFalse();
    }
  }

  @Nested
  static class ConstrainedDecoding {

    @Test
    void passesASchemaConstraintToConstrainedModelsWhenToolsAreDeclared() {
      ConstraintRecordingModel model = new ConstraintRecordingModel();
      ChatRequest request =
          ChatRequest.builder()
              .messages(UserMessage.from("switch to cooling"))
              .toolSpecifications(MODE)
              .build();

      ChatResponse response =
          new ModelsChatModel(model, ChatTemplate.CHATML, SamplingOptions.builder().build())
              .chat(request);

      assertThat(model.constrainedCalls).isEqualTo(1);
      assertThat(model.unconstrainedCalls).isZero();
      assertThat(response.finishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
      assertThat(response.aiMessage().toolExecutionRequests())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("set_mode");
                assertThat(call.arguments()).isEqualTo("{\"mode\":\"cool\"}");
              });
    }

    @Test
    void fallsBackWhenASchemaRequiresOpenTextArguments() {
      ConstraintRecordingModel model = new ConstraintRecordingModel();
      ChatRequest request =
          ChatRequest.builder()
              .messages(UserMessage.from("weather in Austin?"))
              .toolSpecifications(WEATHER)
              .build();

      ChatResponse response =
          new ModelsChatModel(model, ChatTemplate.CHATML, SamplingOptions.builder().build())
              .chat(request);

      assertThat(model.constrainedCalls).isZero();
      assertThat(model.unconstrainedCalls).isEqualTo(1);
      assertThat(response.aiMessage().hasToolExecutionRequests()).isFalse();
      assertThat(response.aiMessage().text()).isEqualTo("unconstrained");
    }

    @Test
    @Tag("integration")
    void constrainsRuntimeGenerationFromLangChain4jToolSchemas() {
      ModelsChatModel chat =
          new ModelsChatModel(
              new CharacterLogitBackend(),
              ChatTemplate.CHATML,
              SamplingOptions.builder().temperature(0.0f).maxTokens(80).build());
      ChatRequest request =
          ChatRequest.builder()
              .messages(UserMessage.from("switch to cooling"))
              .toolSpecifications(MODE)
              .build();

      ChatResponse response = chat.chat(request);

      assertThat(response.finishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
      assertThat(response.aiMessage().toolExecutionRequests())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("set_mode");
                assertThat(call.arguments()).isEqualTo("{\"mode\":\"cool\"}");
              });
    }
  }

  @Nested
  static class RoundTrip {

    @Test
    void rendersAPriorCallAndItsResultBackIntoHistory() {
      ScriptedModel model = new ScriptedModel("It is 88 degrees.");
      ChatRequest request =
          ChatRequest.builder()
              .messages(
                  UserMessage.from("weather?"),
                  AiMessage.from(
                      ToolExecutionRequest.builder()
                          .id("000000000")
                          .name("get_weather")
                          .arguments("{\"city\":\"Austin\"}")
                          .build()),
                  ToolExecutionResultMessage.from("000000000", "get_weather", "{\"tempF\":88}"))
              .toolSpecifications(WEATHER)
              .build();

      chatModel(model, ChatTemplate.CHATML).chat(request);

      String prompt = model.lastPrompt();
      assertThat(prompt).contains("<tool_call>");
      assertThat(prompt).contains("\"name\": \"get_weather\"");
      assertThat(prompt).contains("<tool_response>");
      assertThat(prompt).contains("{\"tempF\":88}");
    }

    @Test
    void acceptsAnAssistantTurnThatIsOnlyACall() {
      // AiMessage.from(request) carries no text at all; rendering must not reject it.
      ScriptedModel model = new ScriptedModel("done");
      ChatRequest request =
          ChatRequest.builder()
              .messages(
                  UserMessage.from("q"),
                  AiMessage.from(
                      ToolExecutionRequest.builder()
                          .id("000000000")
                          .name("ping")
                          .arguments("{}")
                          .build()),
                  ToolExecutionResultMessage.from("000000000", "ping", "pong"))
              .toolSpecifications(WEATHER)
              .build();

      assertThat(chatModel(model, ChatTemplate.CHATML).chat(request)).isNotNull();
      assertThat(model.lastPrompt()).contains("\"name\": \"ping\"");
    }
  }

  @Nested
  static class Capabilities {

    @Test
    void reportsWhetherToolsCanBeUsed() {
      ScriptedModel model = new ScriptedModel("x");

      assertThat(chatModel(model, ChatTemplate.CHATML).supportsTools()).isTrue();
      assertThat(chatModel(model, ChatTemplate.LLAMA3).supportsTools()).isTrue();
      assertThat(chatModel(model, ChatTemplate.GEMMA).supportsTools()).isFalse();
      // Gemma 4 has a real tool format, but a tagged one this runtime cannot yet decode.
      assertThat(chatModel(model, ChatTemplate.GEMMA4).supportsTools()).isFalse();
    }
  }

  private static final class ConstraintRecordingModel implements ConstrainedTextGenerationModel {
    private int constrainedCalls;
    private int unconstrainedCalls;
    private final Tokenizer tokenizer =
        new Tokenizer() {
          @Override
          public int[] encode(String text) {
            return text.chars().toArray();
          }

          @Override
          public String decode(int[] tokens) {
            StringBuilder decoded = new StringBuilder();
            for (int token : tokens) {
              decoded.append(decode(token));
            }
            return decoded.toString();
          }

          @Override
          public String decode(int token) {
            return String.valueOf((char) token);
          }

          @Override
          public int vocabSize() {
            return Character.MAX_VALUE + 1;
          }

          @Override
          public int bosToken() {
            return 0;
          }

          @Override
          public int eosToken() {
            return 1;
          }
        };

    @Override
    public String modelName() {
      return "ConstraintRecordingModel";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("constraint-recording");
    }

    @Override
    public Tokenizer tokenizer() {
      return tokenizer;
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      unconstrainedCalls++;
      stream.onToken("unconstrained");
      stream.onComplete();
    }

    @Override
    public void generate(
        ModelPrompt prompt,
        SamplingOptions options,
        TokenStream stream,
        TokenConstraint constraint) {
      constrainedCalls++;
      String output =
          "<tool_call>{\"name\":\"set_mode\",\"arguments\":{\"mode\":\"cool\"}}</tool_call>";
      for (int index = 0; index < output.length(); index++) {
        int token = output.charAt(index);
        if (!constraint.allows(token)) {
          stream.onError(new AssertionError("constraint rejected token " + token));
          return;
        }
        constraint.accept(token);
        stream.onToken(String.valueOf((char) token));
      }
      stream.onComplete();
    }
  }

  private static final class CharacterLogitBackend implements InferenceBackend {
    private final Tokenizer tokenizer =
        new Tokenizer() {
          @Override
          public int[] encode(String text) {
            return text.chars().toArray();
          }

          @Override
          public int[] encode(ModelPrompt prompt) {
            return prompt.text().chars().toArray();
          }

          @Override
          public int[] encodeControl(String text) {
            return encode(text);
          }

          @Override
          public String decode(int[] tokens) {
            StringBuilder decoded = new StringBuilder();
            for (int token : tokens) {
              decoded.append(decode(token));
            }
            return decoded.toString();
          }

          @Override
          public String decode(int token) {
            return String.valueOf((char) token);
          }

          @Override
          public int vocabSize() {
            return Character.MAX_VALUE + 1;
          }

          @Override
          public int bosToken() {
            return 0;
          }

          @Override
          public int eosToken() {
            return 1;
          }
        };

    @Override
    public String name() {
      return "character-logit";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("test", "CharacterLogit", 256, tokenizer.vocabSize(), 8, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return tokenizer;
    }

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      return logits();
    }

    @Override
    public float[] forward(int token, int position) {
      return logits();
    }

    @Override
    public void close() {}

    private static float[] logits() {
      float[] logits = new float[Character.MAX_VALUE + 1];
      logits['x'] = 100.0f;
      return logits;
    }
  }
}
