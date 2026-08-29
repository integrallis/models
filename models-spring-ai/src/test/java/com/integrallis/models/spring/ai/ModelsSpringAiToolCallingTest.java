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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.AuxiliaryTextGenerationModel;
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
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;

@Tag("unit")
class ModelsSpringAiToolCallingTest {

  private static final String SCHEMA =
      "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}";
  private static final String REQUIRED_CITY_SCHEMA =
      "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}";
  private static final String MODE_SCHEMA =
      "{\"type\":\"object\",\"properties\":{\"mode\":{\"type\":\"string\",\"enum\":[\"cool\",\"heat\"]}},\"required\":[\"mode\"]}";

  /** Records the rendered prompt and replays a canned completion. */
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

  private static ToolCallback weatherCallback() {
    return weatherCallback(SCHEMA);
  }

  private static ToolCallback requiredWeatherCallback() {
    return weatherCallback(REQUIRED_CITY_SCHEMA);
  }

  private static ToolCallback weatherCallback(String inputSchema) {
    ToolDefinition definition =
        DefaultToolDefinition.builder()
            .name("get_weather")
            .description("Look up the forecast")
            .inputSchema(inputSchema)
            .build();
    return new ToolCallback() {
      @Override
      public ToolDefinition getToolDefinition() {
        return definition;
      }

      @Override
      public String call(String toolInput) {
        return "{\"tempF\":88}";
      }
    };
  }

  private static ToolCallback modeCallback() {
    ToolDefinition definition =
        DefaultToolDefinition.builder()
            .name("set_mode")
            .description("Set the HVAC mode")
            .inputSchema(MODE_SCHEMA)
            .build();
    return new ToolCallback() {
      @Override
      public ToolDefinition getToolDefinition() {
        return definition;
      }

      @Override
      public String call(String toolInput) {
        return "{}";
      }
    };
  }

  private static Prompt promptWithTools(
      List<org.springframework.ai.chat.messages.Message> messages) {
    return new Prompt(messages, toolOptions(List.of(weatherCallback()), false));
  }

  private static Prompt promptWithModeTool(
      List<org.springframework.ai.chat.messages.Message> messages) {
    return new Prompt(messages, toolOptions(List.of(modeCallback()), false));
  }

  private static Prompt promptWithRequiredWeatherTool(
      List<org.springframework.ai.chat.messages.Message> messages) {
    return new Prompt(messages, toolOptions(List.of(requiredWeatherCallback()), false));
  }

  private static ToolCallingChatOptions toolOptions(
      List<ToolCallback> callbacks, boolean internalExecution) {
    ToolCallingChatOptions options =
        ToolCallingChatOptions.builder().toolCallbacks(callbacks).build();
    try {
      ToolCallingChatOptions.class
          .getMethod("setInternalToolExecutionEnabled", Boolean.class)
          .invoke(options, internalExecution);
    } catch (NoSuchMethodException ignored) {
      // Spring AI 2.0 always delegates execution to ToolCallingAdvisor.
    } catch (ReflectiveOperationException failure) {
      throw new AssertionError("cannot configure Spring AI tool-execution policy", failure);
    }
    return options;
  }

  private static boolean hasLegacyInternalToolExecution() {
    try {
      ToolCallingChatOptions.class.getMethod("getInternalToolExecutionEnabled");
      return true;
    } catch (NoSuchMethodException ignored) {
      return false;
    }
  }

  private static List<ToolCallback> numberedCallbacks(int count) {
    List<ToolCallback> callbacks = new ArrayList<>(count);
    for (int index = 0; index < count; index++) {
      ToolDefinition definition =
          DefaultToolDefinition.builder()
              .name("tool-" + index)
              .description("A numbered test tool " + index)
              .inputSchema("{\"type\":\"object\",\"properties\":{}}")
              .build();
      callbacks.add(
          new ToolCallback() {
            @Override
            public ToolDefinition getToolDefinition() {
              return definition;
            }

            @Override
            public String call(String toolInput) {
              return "{}";
            }
          });
    }
    return List.copyOf(callbacks);
  }

  @Nested
  static class Declaration {

    @Test
    void rendersDeclaredToolsIntoThePrompt() {
      ScriptedModel model = new ScriptedModel("no call here");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      chat.call(promptWithTools(List.of(new UserMessage("weather in Austin?"))));

      assertThat(model.lastPrompt()).contains("# Tools");
      assertThat(model.lastPrompt()).contains("get_weather");
      assertThat(model.lastPrompt()).contains(SCHEMA);
    }

    @Test
    void withoutToolsRendersThePlainEnvelope() {
      ScriptedModel model = new ScriptedModel("hi");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      chat.call(new Prompt(List.of(new UserMessage("hello"))));

      assertThat(model.lastPrompt()).doesNotContain("# Tools");
    }

    @Test
    void retrievesFiveRelevantToolsAndCachesTheirSchemaEmbeddings() {
      RetrievalModel model = new RetrievalModel();
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());
      Prompt prompt =
          new Prompt(
              List.of(new UserMessage("use tool-6")), toolOptions(numberedCallbacks(7), false));

      chat.call(prompt);
      chat.call(prompt);

      assertThat(model.lastPrompt()).contains("tool-6").doesNotContain("tool-4", "tool-5");
      assertThat(model.encoded).hasSize(9);
    }

    @Test
    void retrievesToolsBeforeRenderingAStreamingRequest() {
      RetrievalModel model = new RetrievalModel();
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());
      Prompt prompt =
          new Prompt(
              List.of(new UserMessage("use tool-6")), toolOptions(numberedCallbacks(7), false));

      chat.stream(prompt).collectList().block();

      assertThat(model.lastPrompt()).contains("tool-6").doesNotContain("tool-4", "tool-5");
      assertThat(model.encoded).hasSize(8);
    }

    @Test
    void refusesToolsWhenTheTemplateCannotExpressThem() {
      ScriptedModel model = new ScriptedModel("x");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(model, ChatTemplate.GEMMA, SamplingOptions.builder().build());

      assertThatThrownBy(() -> chat.call(promptWithTools(List.of(new UserMessage("q")))))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("gemma");
    }

    @Test
    void refusesToolsWhenTheArtifactWasNotQualifiedForToolCalling() {
      ScriptedModel model = new ScriptedModel("ordinary text");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model,
              "qwen-chat-only",
              ChatTemplate.CHATML,
              SamplingOptions.builder().build(),
              Set.of("chat", "text-generation"));

      assertThatThrownBy(() -> chat.call(promptWithTools(List.of(new UserMessage("weather?")))))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("qwen-chat-only")
          .hasMessageContaining("not qualified for tool-calling");
      assertThat(model.lastPrompt()).isNull();
    }

    @Test
    void acceptsToolsWhenTheArtifactWasQualifiedForToolCalling() {
      ScriptedModel model = new ScriptedModel("ordinary text");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model,
              "needle-qualified",
              ChatTemplate.NEEDLE2,
              SamplingOptions.builder().build(),
              Set.of("chat", "text-generation", "tool-calling"));

      chat.call(promptWithTools(List.of(new UserMessage("weather?"))));

      assertThat(model.lastPrompt()).contains("get_weather");
    }
  }

  @Nested
  static class ChatClientExecution {

    @Test
    void stopsARepeatingInternalToolLoop() {
      assumeTrue(hasLegacyInternalToolExecution());
      ScriptedModel model =
          new ScriptedModel(
              "<tool_call>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Austin\"}}]</tool_call>");
      ModelsSpringAiChatModel adapter =
          new ModelsSpringAiChatModel(
              model,
              "needle-qualified",
              ChatTemplate.NEEDLE2,
              SamplingOptions.builder().build(),
              Set.of("chat", "text-generation", "tool-calling"));
      Prompt prompt =
          new Prompt(
              List.of(new UserMessage("weather in Austin?")),
              toolOptions(List.of(weatherCallback()), true));

      assertThatThrownBy(() -> adapter.call(prompt))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("16 consecutive turns");
    }

    @Test
    void chatClientExecutesTheToolAndReturnsTheFollowUpAnswer() {
      SequentialScriptedModel model =
          new SequentialScriptedModel(
              "<tool_call>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Austin\"}}]</tool_call><|im_end|>",
              "It is 88 degrees.");
      ModelsSpringAiChatModel adapter =
          new ModelsSpringAiChatModel(
              model,
              "needle-qualified",
              ChatTemplate.NEEDLE2,
              SamplingOptions.builder().build(),
              Set.of("chat", "text-generation", "tool-calling"));
      ChatClient client = ChatClient.create(adapter);

      String answer =
          client.prompt().user("weather in Austin?").tools(new WeatherTools()).call().content();

      assertThat(answer).isEqualTo("It is 88 degrees.");
      assertThat(model.prompts()).hasSize(2);
      assertThat(model.prompts().get(1)).contains("{\"tempF\":88}");
    }

    @Test
    void defaultToolsExecutesTheUsersHyphenatedZipcodeToolAndReturnsTheFollowUpAnswer() {
      SequentialScriptedModel model =
          new SequentialScriptedModel(
              "<tool_call>[{\"name\":\"get-weather-for-zipcode\",\"arguments\":{\"zipcode\":\"88252\"}}]</tool_call><|im_end|>",
              "It is raining cats and dogs and 78 degrees in 88252.");
      ModelsSpringAiChatModel adapter =
          new ModelsSpringAiChatModel(
              model,
              "needle-qualified",
              ChatTemplate.NEEDLE2,
              SamplingOptions.builder().build(),
              Set.of("chat", "text-generation", "tool-calling"));
      ZipcodeWeatherTools weatherTools = new ZipcodeWeatherTools();
      ChatClient client = ChatClient.builder(adapter).defaultTools(weatherTools).build();

      String answer = client.prompt().user("What is the weather for 88252?").call().content();

      assertThat(answer).isEqualTo("It is raining cats and dogs and 78 degrees in 88252.");
      assertThat(weatherTools.invocations).hasValue(1);
      assertThat(model.prompts()).hasSize(2);
      assertThat(model.prompts().get(1)).contains("88252", "Raining cats and dogs", "78");
    }

    @Test
    void streamingChatClientExecutesTheToolAndReturnsTheFollowUpAnswer() {
      SequentialScriptedModel model =
          new SequentialScriptedModel(
              "<tool_call>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Austin\"}}]</tool_call><|im_end|>",
              "It is 88 degrees.");
      ModelsSpringAiChatModel adapter =
          new ModelsSpringAiChatModel(
              model,
              "needle-qualified",
              ChatTemplate.NEEDLE2,
              SamplingOptions.builder().build(),
              Set.of("chat", "text-generation", "tool-calling"));
      ChatClient client = ChatClient.create(adapter);

      String answer =
          String.join(
              "",
              client.prompt().user("weather in Austin?").tools(new WeatherTools()).stream()
                  .content()
                  .collectList()
                  .block());

      assertThat(answer).isEqualTo("It is 88 degrees.");
      assertThat(model.prompts()).hasSize(2);
      assertThat(model.prompts().get(1)).contains("{\"tempF\":88}");
    }
  }

  private static final class SequentialScriptedModel implements TextGenerationModel {
    private final ArrayDeque<String> completions;
    private final List<String> prompts = new ArrayList<>();

    SequentialScriptedModel(String... completions) {
      this.completions = new ArrayDeque<>(List.of(completions));
    }

    @Override
    public String modelName() {
      return "SequentialScriptedModel";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("scripted");
    }

    @Override
    public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
      prompts.add(prompt.text());
      stream.onToken(completions.removeFirst());
      stream.onComplete();
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      prompts.add(prompt);
      stream.onToken(completions.removeFirst());
      stream.onComplete();
    }

    List<String> prompts() {
      return List.copyOf(prompts);
    }
  }

  private static final class WeatherTools {

    @Tool(name = "get_weather", description = "Get the current weather for a city")
    String getWeather(@ToolParam(description = "City name") String city) {
      return "{\"tempF\":88}";
    }
  }

  private static final class ZipcodeWeatherTools {
    private final AtomicInteger invocations = new AtomicInteger();

    @Tool(name = "get-weather-for-zipcode", description = "Gets weather for a given zipcode")
    Weather getWeatherForZipcode(
        @ToolParam(description = "The zipcode to get weather for") String zipcode) {
      invocations.incrementAndGet();
      return new Weather(zipcode, "Raining cats and dogs", 78);
    }
  }

  private record Weather(String zipcode, String conditions, int temperature) {}

  private static final class RetrievalModel implements AuxiliaryTextGenerationModel {
    private final List<String> encoded = new ArrayList<>();
    private String lastPrompt;

    @Override
    public boolean supportsContrastiveEncoding() {
      return true;
    }

    @Override
    public int contrastiveDimension() {
      return 8;
    }

    @Override
    public float[] encodeContrastive(ModelPrompt prompt) {
      String text = prompt.text();
      encoded.add(text);
      float[] vector = new float[contrastiveDimension()];
      for (int index = 0; index < vector.length; index++) {
        if (text.contains("tool-" + index)) {
          vector[index] = 1.0f;
        }
      }
      return vector;
    }

    @Override
    public boolean supportsConfidenceScoring() {
      return false;
    }

    @Override
    public float scoreConfidence(ModelPrompt sequence) {
      throw new UnsupportedOperationException();
    }

    @Override
    public String modelName() {
      return "RetrievalModel";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("retrieval");
    }

    @Override
    public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
      lastPrompt = prompt.text();
      stream.onToken("done");
      stream.onComplete();
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      lastPrompt = prompt;
      stream.onToken("done");
      stream.onComplete();
    }

    String lastPrompt() {
      return lastPrompt;
    }
  }

  @Nested
  static class Recovery {

    @Test
    void surfacesAToolCallOnTheAssistantMessage() {
      // Spring AI 2.0 keys tool execution on hasToolCalls(), not on a finish reason.
      ScriptedModel model =
          new ScriptedModel(
              "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Austin\"}}\n</tool_call>");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      ChatResponse response = chat.call(promptWithTools(List.of(new UserMessage("weather?"))));

      assertThat(response.hasToolCalls()).isTrue();
      AssistantMessage assistant = response.getResult().getOutput();
      assertThat(assistant.getToolCalls())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("get_weather");
                assertThat(call.arguments()).isEqualTo("{\"city\": \"Austin\"}");
                assertThat(call.type()).isEqualTo("function");
                assertThat(call.id()).isEqualTo("000000000");
              });
    }

    @Test
    void keepsProseAlongsideTheCall() {
      ScriptedModel model =
          new ScriptedModel(
              "Let me check.\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      ChatResponse response = chat.call(promptWithTools(List.of(new UserMessage("weather?"))));

      assertThat(response.getResult().getOutput().getText()).isEqualTo("Let me check.");
      assertThat(response.hasToolCalls()).isTrue();
    }

    @Test
    void plainAnswersCarryNoToolCalls() {
      ScriptedModel model = new ScriptedModel("It is 88 degrees.");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      ChatResponse response = chat.call(promptWithTools(List.of(new UserMessage("weather?"))));

      assertThat(response.hasToolCalls()).isFalse();
      assertThat(response.getResult().getOutput().getText()).isEqualTo("It is 88 degrees.");
    }

    @Test
    void doesNotScanForCallsWhenNoToolsWereDeclared() {
      // Text that merely looks like a call must stay text when the caller offered no tools.
      ScriptedModel model =
          new ScriptedModel("<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      ChatResponse response = chat.call(new Prompt(List.of(new UserMessage("hi"))));

      assertThat(response.hasToolCalls()).isFalse();
    }
  }

  @Nested
  static class ConstrainedDecoding {

    @Test
    void passesASchemaConstraintToConstrainedModelsWhenToolsAreDeclared() {
      ConstraintRecordingModel model = new ConstraintRecordingModel();
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      ChatResponse response =
          chat.call(promptWithModeTool(List.of(new UserMessage("switch to cooling"))));

      assertThat(model.constrainedCalls).isEqualTo(1);
      assertThat(model.unconstrainedCalls).isZero();
      assertThat(response.hasToolCalls()).isTrue();
      assertThat(response.getResult().getOutput().getToolCalls())
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
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      ChatResponse response =
          chat.call(promptWithRequiredWeatherTool(List.of(new UserMessage("weather in Austin?"))));

      assertThat(model.constrainedCalls).isZero();
      assertThat(model.unconstrainedCalls).isEqualTo(1);
      assertThat(response.hasToolCalls()).isFalse();
      assertThat(response.getResult().getOutput().getText()).isEqualTo("unconstrained");
    }

    @Test
    @Tag("integration")
    void constrainsRuntimeGenerationFromSpringToolSchemas() {
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              new CharacterLogitBackend(),
              ChatTemplate.CHATML,
              SamplingOptions.builder().temperature(0.0f).maxTokens(80).build());

      ChatResponse response =
          chat.call(promptWithModeTool(List.of(new UserMessage("switch to cooling"))));

      assertThat(response.hasToolCalls()).isTrue();
      assertThat(response.getResult().getOutput().getToolCalls())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("set_mode");
                assertThat(call.arguments()).isEqualTo("{\"mode\":\"cool\"}");
              });
    }
  }

  @Nested
  static class ResultsComingBack {

    @Test
    void mapsToolResponsesIntoTheConversation() {
      ScriptedModel model = new ScriptedModel("It is 88 degrees.");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      chat.call(
          promptWithTools(
              List.of(
                  new UserMessage("weather?"),
                  ToolResponseMessage.builder()
                      .responses(
                          List.of(
                              new ToolResponseMessage.ToolResponse(
                                  "000000000", "get_weather", "{\"tempF\":88}")))
                      .build())));

      assertThat(model.lastPrompt()).contains("<tool_response>");
      assertThat(model.lastPrompt()).contains("{\"tempF\":88}");
    }

    @Test
    void returnsNeedleToolResultsDirectlyWithoutStartingAnotherToolSelectionTurn() {
      ConstraintRecordingModel model = new ConstraintRecordingModel();
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.NEEDLE2, SamplingOptions.builder().build());
      ToolResponseMessage result =
          ToolResponseMessage.builder()
              .responses(
                  List.of(
                      new ToolResponseMessage.ToolResponse(
                          "000000000",
                          "get-weather-for-zipcode",
                          "{\"zipcode\":\"88252\",\"conditions\":\"Raining cats and dogs\","
                              + "\"temperature\":78}")))
              .build();

      ChatResponse response =
          chat.call(
              promptWithTools(List.of(new UserMessage("What is the weather for 88252?"), result)));

      assertThat(response.hasToolCalls()).isFalse();
      assertThat(response.getResult().getOutput().getText())
          .isEqualTo(
              "{\"zipcode\":\"88252\",\"conditions\":\"Raining cats and dogs\","
                  + "\"temperature\":78}");
      assertThat(model.constrainedCalls).isZero();
      assertThat(model.unconstrainedCalls).isZero();
    }

    @Test
    void doesNotReuseAnOldNeedleToolResultForANewUserTurn() {
      ConstraintRecordingModel model = new ConstraintRecordingModel(true);
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.NEEDLE2, SamplingOptions.builder().build());
      ToolResponseMessage oldResult =
          ToolResponseMessage.builder()
              .responses(
                  List.of(
                      new ToolResponseMessage.ToolResponse(
                          "000000000", "set_mode", "{\"mode\":\"cool\"}")))
              .build();

      ChatResponse response =
          chat.call(
              promptWithModeTool(
                  List.of(
                      new UserMessage("switch to cooling"),
                      oldResult,
                      new UserMessage("switch to cooling again"))));

      assertThat(response.hasToolCalls()).isTrue();
      assertThat(model.constrainedCalls).isOne();
      assertThat(model.unconstrainedCalls).isZero();
    }

    @Test
    void streamsTheCompletedNeedleToolResultWithoutSelectingAnotherAction() {
      ConstraintRecordingModel model = new ConstraintRecordingModel(true);
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.NEEDLE2, SamplingOptions.builder().build());
      ToolResponseMessage result =
          ToolResponseMessage.builder()
              .responses(
                  List.of(
                      new ToolResponseMessage.ToolResponse(
                          "000000000", "set_mode", "{\"mode\":\"cool\"}")))
              .build();

      ChatResponse response =
          chat.stream(promptWithModeTool(List.of(new UserMessage("switch to cooling"), result)))
              .blockLast();

      assertThat(response).isNotNull();
      assertThat(response.hasToolCalls()).isFalse();
      assertThat(response.getResult().getOutput().getText()).isEqualTo("{\"mode\":\"cool\"}");
      assertThat(model.constrainedCalls).isZero();
      assertThat(model.unconstrainedCalls).isZero();
    }
  }

  private static final class ConstraintRecordingModel implements ConstrainedTextGenerationModel {
    private int constrainedCalls;
    private int unconstrainedCalls;
    private final boolean arrayWrapped;
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

    ConstraintRecordingModel() {
      this(false);
    }

    ConstraintRecordingModel(boolean arrayWrapped) {
      this.arrayWrapped = arrayWrapped;
    }

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
          arrayWrapped
              ? "<tool_call>[{\"name\":\"set_mode\",\"arguments\":{\"mode\":\"cool\"}}]</tool_call>"
              : "<tool_call>{\"name\":\"set_mode\",\"arguments\":{\"mode\":\"cool\"}}</tool_call>";
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
