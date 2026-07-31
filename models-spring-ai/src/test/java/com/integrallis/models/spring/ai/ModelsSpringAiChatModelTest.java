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
import static org.assertj.core.api.Assertions.assertThatNullPointerException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;

@Tag("unit")
class ModelsSpringAiChatModelTest {

  @Test
  void springAiChatModelUsesModelsRuntime() {
    ChatModel model =
        new ModelsSpringAiChatModel(
            backendGenerating(new int[] {3, 4, 1}),
            SamplingOptions.builder().temperature(0.0f).maxTokens(10).build());

    assertThat(model.call("hello")).isEqualTo(" world");
  }

  @Test
  void springAiStreamingChatModelEmitsGeneratedTokens() {
    ChatModel model =
        new ModelsSpringAiChatModel(
            backendGenerating(new int[] {3, 4, 5, 1}),
            SamplingOptions.builder().temperature(0.0f).maxTokens(10).build());

    String streamed =
        model.stream(new Prompt("hello"))
            .map(ChatResponse::getResult)
            .map(generation -> generation.getOutput().getText())
            .collectList()
            .blockOptional()
            .orElseThrow()
            .stream()
            .reduce("", String::concat);

    assertThat(streamed).isEqualTo(" world!");
  }

  @Test
  void springAiChatModelAcceptsTheSharedLocalEngineContract() {
    ChatModel model = new ModelsSpringAiChatModel(highLevelModel("local answer"));

    assertThat(model.call("hello")).isEqualTo("local answer");
  }

  @Test
  void springAiChatModelMapsRequestedOptionsAndExposesDefaults() {
    RecordingModel delegate = new RecordingModel("mapped answer", null);
    SamplingOptions defaults =
        SamplingOptions.builder()
            .temperature(0.8f)
            .topP(0.7f)
            .topK(40)
            .maxTokens(200)
            .repetitionPenalty(1.2f)
            .seed(42L)
            .stopSequences(List.of("DEFAULT_STOP"))
            .build();
    ModelsSpringAiChatModel model =
        new ModelsSpringAiChatModel(delegate, ChatTemplate.CHATML, defaults);
    ChatOptions requested =
        ChatOptions.builder()
            .temperature(0.2)
            .topP(0.3)
            .topK(7)
            .maxTokens(19)
            .stopSequences(List.of("REQUEST_STOP"))
            .build();

    ChatResponse response =
        model.call(
            new Prompt(
                List.of(
                    new SystemMessage("system"),
                    new UserMessage("question"),
                    new AssistantMessage("prior answer")),
                requested));

    assertThat(delegate.prompt)
        .isEqualTo(
            """
            <|im_start|>system
            system<|im_end|>
            <|im_start|>user
            question<|im_end|>
            <|im_start|>assistant
            prior answer<|im_end|>
            <|im_start|>assistant
            """);
    assertThat(delegate.modelPrompt.segments())
        .anySatisfy(
            segment -> {
              assertThat(segment.kind()).isEqualTo(ModelPrompt.SegmentKind.TEXT);
              assertThat(segment.text()).isEqualTo("question");
            });
    assertThat(delegate.options.temperature()).isEqualTo(0.2f);
    assertThat(delegate.options.topP()).isEqualTo(0.3f);
    assertThat(delegate.options.topK()).isEqualTo(7);
    assertThat(delegate.options.maxTokens()).isEqualTo(19);
    assertThat(delegate.options.repetitionPenalty()).isEqualTo(1.2f);
    assertThat(delegate.options.seed()).isEqualTo(42L);
    assertThat(delegate.options.stopSequences()).containsExactly("REQUEST_STOP");
    assertThat(response.getResult().getOutput().getText()).isEqualTo("mapped answer");
    assertThat(model.getOptions().getTemperature()).isEqualTo((double) defaults.temperature());
    assertThat(model.getOptions().getTopP()).isEqualTo((double) defaults.topP());
    assertThat(model.getOptions().getTopK()).isEqualTo(40);
    assertThat(model.getOptions().getMaxTokens()).isEqualTo(200);
    assertThat(model.getOptions().getStopSequences()).containsExactly("DEFAULT_STOP");
  }

  @Test
  void springAiChatModelRetainsDefaultsWhenPromptHasNoOptions() {
    RecordingModel delegate = new RecordingModel("answer", null);
    SamplingOptions defaults =
        SamplingOptions.builder()
            .temperature(0.4f)
            .topP(0.6f)
            .topK(11)
            .maxTokens(32)
            .repetitionPenalty(1.1f)
            .stopSequences(List.of("END"))
            .build();
    ModelsSpringAiChatModel model = new ModelsSpringAiChatModel(delegate, defaults);

    model.call(new Prompt("question"));

    assertThat(delegate.options).isEqualTo(defaults);
  }

  @Test
  void springAiStreamingChatModelPropagatesGenerationFailure() {
    IllegalStateException failure = new IllegalStateException("generation failed");
    ChatModel model = new ModelsSpringAiChatModel(new RecordingModel("", failure));

    assertThatThrownBy(() -> model.stream(new Prompt("question")).blockLast()).isSameAs(failure);
  }

  @Test
  void springAiStreamingRunsBlockingInferenceOffTheSubscriberThread() {
    Thread subscriberThread = Thread.currentThread();
    AtomicReference<Thread> generationThread = new AtomicReference<>();
    TextGenerationModel delegate =
        new TextGenerationModel() {
          @Override
          public String modelName() {
            return "ThreadRecordingModel";
          }

          @Override
          public BackendDiagnostics diagnostics() {
            return BackendDiagnostics.unavailable("thread-recording");
          }

          @Override
          public void generate(String prompt, SamplingOptions options, TokenStream stream) {
            generationThread.set(Thread.currentThread());
            stream.onToken("answer");
            stream.onComplete();
          }
        };

    new ModelsSpringAiChatModel(delegate).stream(new Prompt("question")).blockLast();

    assertThat(generationThread.get()).isNotSameAs(subscriberThread);
    assertThat(generationThread.get().getName()).contains("boundedElastic");
  }

  @Test
  void springAiChatModelRejectsNullDependenciesAndPrompts() {
    assertThatNullPointerException()
        .isThrownBy(() -> new ModelsSpringAiChatModel((TextGenerationModel) null));
    assertThatNullPointerException()
        .isThrownBy(() -> new ModelsSpringAiChatModel(highLevelModel("answer"), null));
    assertThatNullPointerException()
        .isThrownBy(
            () ->
                new ModelsSpringAiChatModel(
                    highLevelModel("answer"),
                    (ChatTemplate) null,
                    SamplingOptions.builder().build()));
    ModelsSpringAiChatModel model = new ModelsSpringAiChatModel(highLevelModel("answer"));
    assertThatNullPointerException().isThrownBy(() -> model.call((Prompt) null));
    assertThatNullPointerException().isThrownBy(() -> model.stream((Prompt) null));
  }

  private static InferenceBackend backendGenerating(int[] generatedSequence) {
    Tokenizer tokenizer = testTokenizer();
    return new InferenceBackend() {
      @Override
      public String name() {
        return "test";
      }

      @Override
      public ModelMetadata metadata() {
        return new ModelMetadata("test", "SpringAiSmokeModel", 64, 6, 16, 1, 1, 1);
      }

      @Override
      public Tokenizer tokenizer() {
        return tokenizer;
      }

      @Override
      public float[] forward(int token, int position) {
        float[] logits = new float[6];
        int generationIndex = position - tokenizer.encode("hello").length + 1;
        if (generationIndex >= 0 && generationIndex < generatedSequence.length) {
          logits[generatedSequence[generationIndex]] = 100.0f;
        } else {
          logits[1] = 100.0f;
        }
        return logits;
      }

      @Override
      public void close() {}
    };
  }

  private static TextGenerationModel highLevelModel(String answer) {
    return new TextGenerationModel() {
      @Override
      public String modelName() {
        return "LocalEngineModel";
      }

      @Override
      public BackendDiagnostics diagnostics() {
        return BackendDiagnostics.unavailable("local-test");
      }

      @Override
      public void generate(String prompt, SamplingOptions options, TokenStream stream) {
        stream.onToken(answer);
        stream.onComplete();
      }
    };
  }

  private static final class RecordingModel implements TextGenerationModel {
    private final String answer;
    private final RuntimeException failure;
    private String prompt;
    private ModelPrompt modelPrompt;
    private SamplingOptions options;

    private RecordingModel(String answer, RuntimeException failure) {
      this.answer = answer;
      this.failure = failure;
    }

    @Override
    public String modelName() {
      return "RecordingModel";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("recording");
    }

    @Override
    public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
      this.modelPrompt = prompt;
      generate(prompt.text(), options, stream);
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      this.prompt = prompt;
      this.options = options;
      if (failure != null) {
        stream.onError(failure);
        return;
      }
      stream.onToken(answer);
      stream.onComplete();
    }
  }

  private static Tokenizer testTokenizer() {
    return new Tokenizer() {
      private final String[] vocab = {"<s>", "</s>", "hello", " ", "world", "!"};

      @Override
      public int[] encode(String text) {
        List<Integer> tokens = new ArrayList<>();
        String remaining = text;
        while (!remaining.isEmpty()) {
          boolean found = false;
          for (int i = vocab.length - 1; i >= 2; i--) {
            if (remaining.startsWith(vocab[i])) {
              tokens.add(i);
              remaining = remaining.substring(vocab[i].length());
              found = true;
              break;
            }
          }
          if (!found) {
            remaining = remaining.substring(1);
          }
        }
        return tokens.stream().mapToInt(Integer::intValue).toArray();
      }

      @Override
      public String decode(int[] tokens) {
        StringBuilder builder = new StringBuilder();
        for (int token : tokens) {
          builder.append(decode(token));
        }
        return builder.toString();
      }

      @Override
      public String decode(int token) {
        return token >= 0 && token < vocab.length ? vocab[token] : "";
      }

      @Override
      public int vocabSize() {
        return vocab.length;
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
  }
}
