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
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.CustomMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;

@Tag("unit")
class ModelsChatModelTest {

  @Test
  void langChain4jChatModelUsesModelsRuntime() {
    ModelsChatModel model =
        new ModelsChatModel(
            backendGenerating(new int[] {3, 4, 1}),
            SamplingOptions.builder().temperature(0.0f).maxTokens(10).build());

    assertThat(model.chat("hello")).isEqualTo(" world");
    assertThat(model.diagnostics().backend()).isEqualTo("test");
    assertThat(model.diagnostics().planVersion()).isEqualTo("unavailable");
  }

  @Test
  void langChain4jChatModelAcceptsTheSharedLocalEngineContract() {
    ModelsChatModel model = new ModelsChatModel(highLevelModel("local answer"));

    assertThat(model.chat("hello")).isEqualTo("local answer");
    assertThat(model.diagnostics().backend()).isEqualTo("local-test");
  }

  @Test
  void langChain4jChatModelMapsMessagesAndRequestOptions() {
    RecordingModel delegate = new RecordingModel("mapped answer");
    ModelsChatModel model =
        new ModelsChatModel(
            delegate,
            SamplingOptions.builder()
                .temperature(0.9f)
                .topP(0.8f)
                .topK(40)
                .maxTokens(200)
                .repetitionPenalty(1.2f)
                .seed(42L)
                .build());
    CustomMessage custom = CustomMessage.from(Map.of("source", "tool"));
    ChatRequest request =
        ChatRequest.builder()
            .messages(
                List.of(
                    SystemMessage.from("system"),
                    UserMessage.from("question"),
                    AiMessage.from("prior answer"),
                    custom))
            .temperature(0.2)
            .topP(0.3)
            .topK(7)
            .maxOutputTokens(19)
            .build();

    var response = model.doChat(request);

    assertThat(delegate.prompt).isEqualTo("system\nquestion\nprior answer\n" + custom);
    assertThat(delegate.options.temperature()).isEqualTo(0.2f);
    assertThat(delegate.options.topP()).isEqualTo(0.3f);
    assertThat(delegate.options.topK()).isEqualTo(7);
    assertThat(delegate.options.maxTokens()).isEqualTo(19);
    assertThat(delegate.options.repetitionPenalty()).isEqualTo(1.2f);
    assertThat(delegate.options.seed()).isEqualTo(42L);
    assertThat(response.aiMessage().text()).isEqualTo("mapped answer");
    assertThat(response.modelName()).isEqualTo("RecordingModel");
  }

  @Test
  void langChain4jChatModelRetainsDefaultsWhenRequestHasNoOverrides() {
    RecordingModel delegate = new RecordingModel("answer");
    SamplingOptions defaults =
        SamplingOptions.builder()
            .temperature(0.4f)
            .topP(0.7f)
            .topK(12)
            .maxTokens(31)
            .repetitionPenalty(1.1f)
            .build();
    ModelsChatModel model = new ModelsChatModel(delegate, defaults);

    model.doChat(ChatRequest.builder().messages(UserMessage.from("question")).build());

    assertThat(delegate.options).isEqualTo(defaults);
  }

  @Test
  void langChain4jChatModelRejectsNullDependenciesAndRequests() {
    assertThatNullPointerException()
        .isThrownBy(() -> new ModelsChatModel((TextGenerationModel) null));
    assertThatNullPointerException()
        .isThrownBy(() -> new ModelsChatModel(highLevelModel("answer"), null));
    ModelsChatModel model = new ModelsChatModel(highLevelModel("answer"));
    assertThatNullPointerException().isThrownBy(() -> model.doChat(null));
  }

  @Test
  void langChain4jAppCanSeeModelJarsMarkerCatalog() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath()
            .resolve(
                ModelJarRequirement.forSource("hf://ggml-org/Qwen3-0.6B-GGUF")
                    .variant("q4_0")
                    .backend("pure-java")
                    .build())
            .orElseThrow();

    assertThat(descriptor.sourceId()).isEqualTo("hf://ggml-org/Qwen3-0.6B-GGUF");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/Qwen3-0.6B-Q4_0.gguf");
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
        return new ModelMetadata("test", "LangChainSmokeModel", 64, 6, 16, 1, 1, 1);
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
    private String prompt;
    private SamplingOptions options;

    private RecordingModel(String answer) {
      this.answer = answer;
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
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      this.prompt = prompt;
      this.options = options;
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
