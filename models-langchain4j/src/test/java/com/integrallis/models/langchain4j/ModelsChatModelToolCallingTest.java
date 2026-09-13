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

import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.AuxiliaryTextGenerationModel;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.runtime.ActivatedToolModel;
import com.integrallis.models.runtime.ConstrainedTextGenerationModel;
import com.integrallis.models.runtime.GenerationMetrics;
import com.integrallis.models.runtime.PromptCacheMetrics;
import com.integrallis.models.runtime.SharedToolTurn;
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
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
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

  private static final class ActivatedScriptedModel implements ActivatedToolModel {
    private final ArrayDeque<String> toolOutputs;
    private final String responseOutput;
    private final Tokenizer tokenizer = new CharacterTokenizer();
    private int openedTurns;
    private int closedTurns;
    private int ordinaryGenerations;
    private int extendedTurns;
    private String responsePrompt;
    private final List<TokenConstraint> constraints = new ArrayList<>();

    private ActivatedScriptedModel(String toolOutput, String responseOutput) {
      this(List.of(toolOutput, "<tool_call>[]</tool_call>"), responseOutput);
    }

    private ActivatedScriptedModel(List<String> toolOutputs, String responseOutput) {
      this.toolOutputs = new ArrayDeque<>(toolOutputs);
      this.responseOutput = responseOutput;
    }

    @Override
    public ActivatedAdapterMetadata adapter() {
      return new ActivatedAdapterMetadata(
          "test/base",
          "a".repeat(40),
          "b".repeat(64),
          java.util.Map.of("tokenizer.json", "d".repeat(64)),
          "c".repeat(64),
          1,
          1,
          List.of(1),
          testProvenance());
    }

    @Override
    public List<String> toolAbstentionOutputs() {
      return List.of("<tool_call>\n[]\n</tool_call>");
    }

    private static ActivatedAdapterMetadata.TrainingProvenance testProvenance() {
      return new ActivatedAdapterMetadata.TrainingProvenance(
          "test/dataset",
          "e".repeat(40),
          "train.jsonl",
          "f".repeat(64),
          "1".repeat(64),
          "2".repeat(64),
          "3".repeat(64),
          "formatter.java",
          "4".repeat(64));
    }

    @Override
    public SharedToolTurn openToolTurn(ModelPrompt renderedToolPrompt) {
      int turnIndex = openedTurns++;
      String toolOutput = toolOutputs.removeFirst();
      return new SharedToolTurn() {
        @Override
        public String generateToolCall(SamplingOptions options, TokenConstraint constraint) {
          constraints.add(constraint);
          return toolOutput;
        }

        @Override
        public void generateToolCall(
            SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
          constraints.add(constraint);
          stream.onToken(toolOutput);
          stream.onComplete();
        }

        @Override
        public SharedToolTurn continueToolSelection(ModelPrompt nextToolPrompt) {
          extendedTurns++;
          close();
          return ActivatedScriptedModel.this.openToolTurn(nextToolPrompt);
        }

        @Override
        public String generateBaseResponse(ModelPrompt prompt, SamplingOptions options) {
          responsePrompt = prompt.text();
          return responseOutput;
        }

        @Override
        public void generateBaseResponse(
            ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
          responsePrompt = prompt.text();
          stream.onToken(responseOutput);
          stream.onComplete();
        }

        @Override
        public int sharedPrefixTokens() {
          return 10;
        }

        @Override
        public long sharedPrefixBytes() {
          return 1_024;
        }

        @Override
        public boolean physicallySharesPrefix() {
          return true;
        }

        @Override
        public GenerationMetrics toolMetrics() {
          return metrics(100 + turnIndex, 5);
        }

        @Override
        public GenerationMetrics responseMetrics() {
          return metrics(200, 10);
        }

        @Override
        public void close() {
          if (!closed) {
            closed = true;
            closedTurns++;
          }
        }

        private boolean closed;
      };
    }

    @Override
    public String modelName() {
      return "activated-scripted";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("activated-scripted");
    }

    @Override
    public Tokenizer tokenizer() {
      return tokenizer;
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      ordinaryGenerations++;
      stream.onError(new AssertionError("ordinary generation must not serve a tool turn"));
    }

    @Override
    public void generate(
        ModelPrompt prompt,
        SamplingOptions options,
        TokenStream stream,
        TokenConstraint constraint) {
      ordinaryGenerations++;
      stream.onError(new AssertionError("ordinary generation must not serve a tool turn"));
    }
  }

  private static final class CharacterTokenizer implements Tokenizer {
    @Override
    public int[] encode(String text) {
      return text.chars().toArray();
    }

    @Override
    public String decode(int[] tokens) {
      return "";
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
  }

  private static ModelsChatModel chatModel(ScriptedModel model, ChatTemplate template) {
    return new ModelsChatModel(model, template, SamplingOptions.builder().build());
  }

  private static List<ToolSpecification> numberedTools(int count) {
    List<ToolSpecification> tools = new ArrayList<>(count);
    for (int index = 0; index < count; index++) {
      tools.add(
          ToolSpecification.builder()
              .name("tool-" + index)
              .description("A numbered test tool " + index)
              .parameters(JsonObjectSchema.builder().build())
              .build());
    }
    return List.copyOf(tools);
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
      assertThat(model.lastPrompt()).contains("\"type\": \"object\"");
      assertThat(model.lastPrompt()).contains("\"city\"");
      assertThat(model.lastPrompt()).contains("\"required\": [\"city\"]");
    }

    @Test
    void withoutToolsRendersThePlainEnvelope() {
      ScriptedModel model = new ScriptedModel("hi");
      ChatRequest request = ChatRequest.builder().messages(UserMessage.from("hello")).build();

      chatModel(model, ChatTemplate.CHATML).chat(request);

      assertThat(model.lastPrompt()).doesNotContain("# Tools");
    }

    @Test
    void retrievesFiveRelevantToolsAndCachesTheirSchemaEmbeddings() {
      RetrievalModel model = new RetrievalModel();
      ModelsChatModel chat =
          new ModelsChatModel(model, ChatTemplate.CHATML, SamplingOptions.builder().build());
      ChatRequest request =
          ChatRequest.builder()
              .messages(UserMessage.from("use tool-6"))
              .toolSpecifications(numberedTools(7))
              .build();

      chat.chat(request);
      chat.chat(request);

      assertThat(model.lastPrompt()).contains("tool-6").doesNotContain("tool-4", "tool-5");
      assertThat(model.encoded).hasSize(9);
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
    void recoversMiniCpmTaggedArgumentsThroughTheDeclaredLangChain4jSchema() {
      ScriptedModel model =
          new ScriptedModel(
              "<function name=\"get_weather\"><param name=\"city\">Austin</param></function>");
      ChatRequest request =
          ChatRequest.builder()
              .messages(UserMessage.from("weather?"))
              .toolSpecifications(WEATHER)
              .build();

      ChatResponse response = chatModel(model, ChatTemplate.MINICPM5_NO_THINK).chat(request);

      assertThat(response.finishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
      assertThat(response.aiMessage().toolExecutionRequests())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("get_weather");
                assertThat(call.arguments()).isEqualTo("{\"city\":\"Austin\"}");
              });
    }

    @Test
    void recoversAGptOssHarmonyCallFromTheGeneratedHeaderContinuation() {
      ScriptedModel model =
          new ScriptedModel(
              "<|channel|>commentary to=functions.get_weather "
                  + "<|constrain|>json<|message|>{\"city\":\"Austin\"}<|call|>");
      ChatRequest request =
          ChatRequest.builder()
              .messages(UserMessage.from("weather?"))
              .toolSpecifications(WEATHER)
              .build();

      ChatResponse response = chatModel(model, ChatTemplate.GPT_OSS).chat(request);

      assertThat(response.finishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
      assertThat(response.aiMessage().toolExecutionRequests())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("get_weather");
                assertThat(call.arguments()).isEqualTo("{\"city\":\"Austin\"}");
              });
    }

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
    void activatedAdapterFiniteGrammarDoesNotForceAnIrrelevantToolCall() {
      String abstention = "<tool_call>\n[]\n</tool_call>";
      ActivatedScriptedModel model = new ActivatedScriptedModel(abstention, "Hello from the base.");
      ModelsChatModel adapter =
          new ModelsChatModel(model, ChatTemplate.CHATML, SamplingOptions.builder().build());
      ChatRequest request =
          ChatRequest.builder()
              .messages(UserMessage.from("Hello, how are you?"))
              .toolSpecifications(MODE)
              .build();

      ChatResponse response = adapter.chat(request);

      assertThat(response.aiMessage().text()).isEqualTo("Hello from the base.");
      TokenConstraint constraint = model.constraints.getFirst();
      for (int token : abstention.chars().toArray()) {
        assertThat(constraint.allows(token)).isTrue();
        constraint.accept(token);
      }
      assertThat(constraint.isComplete()).isTrue();
    }

    @Test
    void activatedAdapterRetainsItsPhysicalBaseBranchAcrossTheLangChainToolRoundTrip() {
      ActivatedScriptedModel model =
          new ActivatedScriptedModel(
              "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Austin\"}}</tool_call>",
              "It is 88 degrees.");
      ModelsChatModel adapter =
          new ModelsChatModel(model, ChatTemplate.CHATML, SamplingOptions.builder().build());
      ChatRequest initial =
          ChatRequest.builder()
              .messages(UserMessage.from("weather?"))
              .toolSpecifications(WEATHER)
              .build();

      ChatResponse toolSelection = adapter.chat(initial);
      ToolExecutionRequest call = toolSelection.aiMessage().toolExecutionRequests().getFirst();
      ChatRequest followUp =
          ChatRequest.builder()
              .messages(
                  UserMessage.from("weather?"),
                  toolSelection.aiMessage(),
                  ToolExecutionResultMessage.from(call.id(), call.name(), "{\"tempF\":88}"))
              .toolSpecifications(WEATHER)
              .build();

      ChatResponse answer = adapter.chat(followUp);

      assertThat(answer.aiMessage().text()).isEqualTo("It is 88 degrees.");
      assertThat(model.openedTurns).isEqualTo(2);
      assertThat(model.extendedTurns).isEqualTo(1);
      assertThat(model.responsePrompt).contains("{\"tempF\":88}");
      assertThat(model.closedTurns).isEqualTo(2);
      assertThat(model.ordinaryGenerations).isZero();
    }

    @Test
    void activatedAdapterReportsTheFinalSelectionAndBaseGenerationUsage() {
      ActivatedScriptedModel model =
          new ActivatedScriptedModel(
              "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Austin\"}}</tool_call>",
              "It is 88 degrees.");
      ModelsChatModel adapter =
          new ModelsChatModel(model, ChatTemplate.CHATML, SamplingOptions.builder().build());
      ChatRequest initial =
          ChatRequest.builder()
              .messages(UserMessage.from("weather?"))
              .toolSpecifications(WEATHER)
              .build();

      ChatResponse toolSelection = adapter.chat(initial);
      ToolExecutionRequest call = toolSelection.aiMessage().toolExecutionRequests().getFirst();
      ChatResponse answer =
          adapter.chat(
              ChatRequest.builder()
                  .messages(
                      UserMessage.from("weather?"),
                      toolSelection.aiMessage(),
                      ToolExecutionResultMessage.from(call.id(), call.name(), "{\"tempF\":88}"))
                  .toolSpecifications(WEATHER)
                  .build());

      assertThat(toolSelection.tokenUsage().inputTokenCount()).isEqualTo(100);
      assertThat(toolSelection.tokenUsage().outputTokenCount()).isEqualTo(5);
      assertThat(answer.tokenUsage().inputTokenCount()).isEqualTo(301);
      assertThat(answer.tokenUsage().outputTokenCount()).isEqualTo(15);
      assertThat(answer.tokenUsage().totalTokenCount()).isEqualTo(316);
    }

    @Test
    void activatedAdapterCanRequestASecondToolBeforeSynthesizingTheAnswer() {
      ActivatedScriptedModel model =
          new ActivatedScriptedModel(
              List.of(
                  "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Austin\"}}</tool_call>",
                  "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Dallas\"}}</tool_call>",
                  "<tool_call>[]</tool_call>"),
              "Austin is 88; Dallas is 90.");
      ModelsChatModel adapter =
          new ModelsChatModel(model, ChatTemplate.CHATML, SamplingOptions.builder().build());
      ChatRequest initial =
          ChatRequest.builder()
              .messages(UserMessage.from("compare weather"))
              .toolSpecifications(WEATHER)
              .build();

      ChatResponse firstSelection = adapter.chat(initial);
      ToolExecutionRequest firstCall =
          firstSelection.aiMessage().toolExecutionRequests().getFirst();
      ChatRequest secondRequest =
          ChatRequest.builder()
              .messages(
                  UserMessage.from("compare weather"),
                  firstSelection.aiMessage(),
                  ToolExecutionResultMessage.from(
                      firstCall.id(), firstCall.name(), "{\"tempF\":88}"))
              .toolSpecifications(WEATHER)
              .build();
      ChatResponse secondSelection = adapter.chat(secondRequest);
      ToolExecutionRequest secondCall =
          secondSelection.aiMessage().toolExecutionRequests().getFirst();
      ChatRequest finalRequest =
          ChatRequest.builder()
              .messages(
                  UserMessage.from("compare weather"),
                  firstSelection.aiMessage(),
                  ToolExecutionResultMessage.from(
                      firstCall.id(), firstCall.name(), "{\"tempF\":88}"),
                  secondSelection.aiMessage(),
                  ToolExecutionResultMessage.from(
                      secondCall.id(), secondCall.name(), "{\"tempF\":90}"))
              .toolSpecifications(WEATHER)
              .build();

      ChatResponse answer = adapter.chat(finalRequest);

      assertThat(answer.aiMessage().text()).isEqualTo("Austin is 88; Dallas is 90.");
      assertThat(model.openedTurns).isEqualTo(3);
      assertThat(model.extendedTurns).isEqualTo(2);
      assertThat(model.closedTurns).isEqualTo(3);
    }

    @Test
    void preservesTheToolNameForGptOssHarmonyResults() {
      ScriptedModel model = new ScriptedModel("It is raining.");
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
                  ToolExecutionResultMessage.from(
                      "000000000", "get_weather", "{\"condition\":\"rain\"}"))
              .toolSpecifications(WEATHER)
              .build();

      chatModel(model, ChatTemplate.GPT_OSS).chat(request);

      assertThat(model.lastPrompt())
          .contains(
              "<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>"
                  + "{\"condition\":\"rain\"}<|end|>");
    }

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
      assertThat(chatModel(model, ChatTemplate.MINICPM5_NO_THINK).supportsTools()).isTrue();
      assertThat(chatModel(model, ChatTemplate.GEMMA).supportsTools()).isFalse();
      // Gemma 4 has a real tool format, but a tagged one this runtime cannot yet decode.
      assertThat(chatModel(model, ChatTemplate.GEMMA4).supportsTools()).isFalse();
    }
  }

  private static GenerationMetrics metrics(int promptTokens, int completionTokens) {
    return new GenerationMetrics(
        true,
        true,
        Duration.ZERO,
        Duration.ZERO,
        Duration.ZERO,
        Optional.empty(),
        Duration.ZERO,
        Duration.ZERO,
        new com.integrallis.models.api.GenerationUsage(promptTokens, completionTokens),
        new PromptCacheMetrics(true, promptTokens, 0, promptTokens));
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
