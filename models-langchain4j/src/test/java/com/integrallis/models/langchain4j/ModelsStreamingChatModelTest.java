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

import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.AuxiliaryTextGenerationModel;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.runtime.ActivatedToolModel;
import com.integrallis.models.runtime.ConstrainedTextGenerationModel;
import com.integrallis.models.runtime.GenerationMetrics;
import com.integrallis.models.runtime.SharedToolTurn;
import com.integrallis.models.runtime.TokenConstraint;
import com.integrallis.models.runtime.chat.ChatTemplate;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ModelsStreamingChatModelTest {

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
            .minP(0.05f)
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
    assertThat(completed.get().tokenUsage()).isNull();
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
    assertThat(delegate.options.minP()).isEqualTo(0.05f);
    assertThat(delegate.options.seed()).isEqualTo(42L);
    assertThat(delegate.options.stopSequences()).containsExactly("REQUEST_STOP");
    assertThat(model.diagnostics().backend()).isEqualTo("stream-recording");
  }

  @Test
  void reportsPromptAndCompletionUsageOnTheTerminalResponse() {
    TextGenerationModel delegate =
        new RecordingStreamingModel(List.of()) {
          @Override
          public void generate(String prompt, SamplingOptions options, TokenStream stream) {
            stream.onToken("mapped ");
            stream.onToken("answer");
            stream.onComplete(new GenerationUsage(5, 2));
          }
        };
    ModelsStreamingChatModel model = new ModelsStreamingChatModel(delegate);
    AtomicReference<ChatResponse> completed = new AtomicReference<>();

    model.doChat(
        ChatRequest.builder().messages(UserMessage.from("question")).build(),
        handler(new ArrayList<>(), completed, new AtomicReference<>()));

    assertThat(completed.get().tokenUsage().inputTokenCount()).isEqualTo(5);
    assertThat(completed.get().tokenUsage().outputTokenCount()).isEqualTo(2);
    assertThat(completed.get().tokenUsage().totalTokenCount()).isEqualTo(7);
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
  void usesASchemaConstraintForStreamingToolCalls() {
    ConstraintRecordingStreamingModel delegate = new ConstraintRecordingStreamingModel();
    ModelsStreamingChatModel model =
        new ModelsStreamingChatModel(
            delegate, ChatTemplate.CHATML, SamplingOptions.builder().build());
    ChatRequest request =
        ChatRequest.builder()
            .messages(UserMessage.from("switch to cooling"))
            .toolSpecifications(MODE)
            .build();
    List<String> partials = new ArrayList<>();
    AtomicReference<ChatResponse> completed = new AtomicReference<>();
    AtomicReference<Throwable> failed = new AtomicReference<>();

    model.doChat(request, handler(partials, completed, failed));

    assertThat(delegate.constrainedCalls).isEqualTo(1);
    assertThat(delegate.unconstrainedCalls).isZero();
    assertThat(partials).isEmpty();
    assertThat(failed).hasValue(null);
    assertThat(completed.get().finishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
    assertThat(completed.get().aiMessage().toolExecutionRequests())
        .singleElement()
        .satisfies(
            call -> {
              assertThat(call.name()).isEqualTo("set_mode");
              assertThat(call.arguments()).isEqualTo("{\"mode\":\"cool\"}");
            });
  }

  @Test
  void activatedAdapterStreamsItsRetainedBaseBranchAfterTheToolResult() {
    ActivatedStreamingModel delegate = new ActivatedStreamingModel();
    ModelsStreamingChatModel model =
        new ModelsStreamingChatModel(
            delegate, ChatTemplate.CHATML, SamplingOptions.builder().build());
    ChatRequest initial =
        ChatRequest.builder()
            .messages(UserMessage.from("switch to cooling"))
            .toolSpecifications(MODE)
            .build();
    AtomicReference<ChatResponse> selected = new AtomicReference<>();

    model.doChat(initial, handler(new ArrayList<>(), selected, new AtomicReference<>()));

    ToolExecutionRequest call = selected.get().aiMessage().toolExecutionRequests().getFirst();
    ChatRequest followUp =
        ChatRequest.builder()
            .messages(
                UserMessage.from("switch to cooling"),
                selected.get().aiMessage(),
                ToolExecutionResultMessage.from(call.id(), call.name(), "cooling enabled"))
            .toolSpecifications(MODE)
            .build();
    List<String> partials = new ArrayList<>();
    AtomicReference<ChatResponse> completed = new AtomicReference<>();

    model.doChat(followUp, handler(partials, completed, new AtomicReference<>()));

    assertThat(partials).containsExactly("Cooling ", "is enabled.");
    assertThat(completed.get().aiMessage().text()).isEqualTo("Cooling is enabled.");
    assertThat(delegate.responsePrompt).contains("cooling enabled");
    assertThat(delegate.openedTurns).isEqualTo(2);
    assertThat(delegate.extendedTurns).isEqualTo(1);
    assertThat(delegate.closedTurns).isEqualTo(2);
    assertThat(delegate.ordinaryGenerations).isZero();
  }

  @Test
  void activatedAdapterStreamingFiniteGrammarPreservesItsNoToolSentinel() {
    String abstention = "<tool_call>\n[]\n</tool_call>";
    ActivatedStreamingModel delegate = new ActivatedStreamingModel(true);
    ModelsStreamingChatModel model =
        new ModelsStreamingChatModel(
            delegate, ChatTemplate.CHATML, SamplingOptions.builder().build());
    ChatRequest request =
        ChatRequest.builder()
            .messages(UserMessage.from("Hello, how are you?"))
            .toolSpecifications(MODE)
            .build();
    AtomicReference<ChatResponse> completed = new AtomicReference<>();

    model.doChat(request, handler(new ArrayList<>(), completed, new AtomicReference<>()));

    assertThat(completed.get().aiMessage().text()).isEqualTo("Cooling is enabled.");
    TokenConstraint constraint = delegate.constraints.getFirst();
    for (int token : abstention.chars().toArray()) {
      assertThat(constraint.allows(token)).isTrue();
      constraint.accept(token);
    }
    assertThat(constraint.isComplete()).isTrue();
  }

  @Test
  void retrievesToolsBeforeRenderingAStreamingRequest() {
    RetrievalStreamingModel delegate = new RetrievalStreamingModel();
    ModelsStreamingChatModel model =
        new ModelsStreamingChatModel(
            delegate, ChatTemplate.CHATML, SamplingOptions.builder().build());
    List<ToolSpecification> tools = new ArrayList<>();
    for (int index = 0; index < 7; index++) {
      tools.add(
          ToolSpecification.builder()
              .name("tool-" + index)
              .description("A numbered test tool " + index)
              .parameters(JsonObjectSchema.builder().build())
              .build());
    }
    ChatRequest request =
        ChatRequest.builder()
            .messages(UserMessage.from("use tool-6"))
            .toolSpecifications(tools)
            .build();

    model.doChat(
        request, handler(new ArrayList<>(), new AtomicReference<>(), new AtomicReference<>()));

    assertThat(delegate.prompt).contains("tool-6").doesNotContain("tool-4", "tool-5");
    assertThat(delegate.encoded).hasSize(8);
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

  private static final class RetrievalStreamingModel implements AuxiliaryTextGenerationModel {
    private final List<String> encoded = new ArrayList<>();
    private String prompt;

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
      return "RetrievalStreamingModel";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("retrieval-streaming");
    }

    @Override
    public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
      this.prompt = prompt.text();
      stream.onToken("done");
      stream.onComplete();
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      this.prompt = prompt;
      stream.onToken("done");
      stream.onComplete();
    }
  }

  private static final class ConstraintRecordingStreamingModel
      implements ConstrainedTextGenerationModel {
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
      return "ConstraintRecordingStreamingModel";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("stream-constraint-recording");
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

  private static final class ActivatedStreamingModel implements ActivatedToolModel {
    private final Tokenizer tokenizer = new CharacterTokenizer();
    private final boolean abstainImmediately;
    private final List<TokenConstraint> constraints = new ArrayList<>();
    private int openedTurns;
    private int extendedTurns;
    private int closedTurns;
    private int ordinaryGenerations;
    private String responsePrompt;

    private ActivatedStreamingModel() {
      this(false);
    }

    private ActivatedStreamingModel(boolean abstainImmediately) {
      this.abstainImmediately = abstainImmediately;
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
      return new SharedToolTurn() {
        @Override
        public String generateToolCall(SamplingOptions options, TokenConstraint constraint) {
          constraints.add(constraint);
          return turnIndex == 0 && !abstainImmediately
              ? "<tool_call>{\"name\":\"set_mode\",\"arguments\":{\"mode\":\"cool\"}}</tool_call>"
              : "<tool_call>\n[]\n</tool_call>";
        }

        @Override
        public void generateToolCall(
            SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
          throw new AssertionError("streaming adapter may accumulate the tool selection directly");
        }

        @Override
        public SharedToolTurn continueToolSelection(ModelPrompt nextToolPrompt) {
          extendedTurns++;
          close();
          return ActivatedStreamingModel.this.openToolTurn(nextToolPrompt);
        }

        @Override
        public String generateBaseResponse(ModelPrompt prompt, SamplingOptions options) {
          throw new AssertionError("streaming adapter must stream the base response");
        }

        @Override
        public void generateBaseResponse(
            ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
          responsePrompt = prompt.text();
          stream.onToken("Cooling ");
          stream.onToken("is enabled.");
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
          return GenerationMetrics.unavailable();
        }

        @Override
        public GenerationMetrics responseMetrics() {
          return GenerationMetrics.unavailable();
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
      return "activated-streaming";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("activated-streaming");
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
}
