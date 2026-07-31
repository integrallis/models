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
package com.integrallis.models.spring.boot;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.spring.ai.ModelsSpringAiChatModel;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.spring.ai.JavaVectorsVectorStore;
import com.integrallis.vectors.spring.boot.JavaVectorsAutoConfiguration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Tag("unit")
class ModelsAutoConfigurationTest {

  private final ApplicationContextRunner runner =
      new ApplicationContextRunner()
          .withConfiguration(AutoConfigurations.of(ModelsAutoConfiguration.class));

  @Test
  void createsSpringAiChatModelFromModelsRuntime() {
    runner
        .withUserConfiguration(LocalModelConfiguration.class)
        .run(
            context -> {
              assertThat(context).hasSingleBean(ModelsSpringAiChatModel.class);
              assertThat(context).hasSingleBean(ChatModel.class);
              assertThat(context.getBean(ChatModel.class).call("question"))
                  .isEqualTo("local answer");
            });
  }

  @Test
  void modelFromLaterAutoConfigurationStillWiresChatModel() {
    new ApplicationContextRunner()
        .withConfiguration(
            AutoConfigurations.of(
                ModelsAutoConfiguration.class, LateLocalModelAutoConfiguration.class))
        .run(
            context -> {
              assertThat(context).hasNotFailed();
              assertThat(context).hasSingleBean(ModelsSpringAiChatModel.class);
            });
  }

  @Test
  void startsWithoutChatModelWhenNoModelsRuntimeIsProvided() {
    runner.run(
        context -> {
          assertThat(context).hasNotFailed();
          assertThat(context).doesNotHaveBean(ModelsSpringAiChatModel.class);
        });
  }

  @Test
  void userProvidedModelsAdapterReplacesTheDefault() {
    runner
        .withUserConfiguration(UserProvidedAdapterConfiguration.class)
        .run(
            context -> {
              assertThat(context).hasSingleBean(ModelsSpringAiChatModel.class);
              assertThat(context.getBean(ModelsSpringAiChatModel.class))
                  .isSameAs(
                      context.getBean(
                          UserProvidedAdapterConfiguration.USER_ADAPTER_BEAN_NAME,
                          ModelsSpringAiChatModel.class));
              assertThat(context)
                  .doesNotHaveBean(ModelsAutoConfiguration.MODELS_CHAT_MODEL_BEAN_NAME);
            });
  }

  @Test
  void localModelsChatModelCoexistsWithOtherSpringAiChatModels() {
    runner
        .withUserConfiguration(LocalModelConfiguration.class, RemoteChatModelConfiguration.class)
        .run(
            context ->
                assertThat(context.getBeansOfType(ChatModel.class))
                    .containsKeys(
                        ModelsAutoConfiguration.MODELS_CHAT_MODEL_BEAN_NAME,
                        RemoteChatModelConfiguration.REMOTE_CHAT_MODEL_BEAN_NAME));
  }

  @Test
  void modelsAndVectorsStartersCoexistAndHonorTheirOwnConfiguration() {
    AtomicReference<String> prompt = LocalModelConfiguration.PROMPT;
    AtomicReference<SamplingOptions> options = LocalModelConfiguration.OPTIONS;
    prompt.set(null);
    options.set(null);

    new ApplicationContextRunner()
        .withConfiguration(
            AutoConfigurations.of(
                ModelsAutoConfiguration.class, JavaVectorsAutoConfiguration.class))
        .withUserConfiguration(LocalModelConfiguration.class, EmbeddingConfiguration.class)
        .withPropertyValues(
            "java-vectors.dimension=2",
            "integrallis.models.chat-template=chatml",
            "integrallis.models.sampling.temperature=0.0",
            "integrallis.models.sampling.max-tokens=17")
        .run(
            context -> {
              assertThat(context).hasNotFailed();
              assertThat(context).hasSingleBean(ModelsSpringAiChatModel.class);
              assertThat(context).hasSingleBean(JavaVectorsVectorStore.class);
              assertThat(context).hasSingleBean(VectorCollection.class);

              JavaVectorsVectorStore store = context.getBean(JavaVectorsVectorStore.class);
              store.add(
                  List.of(
                      new Document(
                          "deployment",
                          "The deployment region is Phoenix.",
                          Map.of("kind", "fact"))));
              assertThat(
                      store.similaritySearch(
                          SearchRequest.builder().query("deployment region").topK(1).build()))
                  .extracting(Document::getText)
                  .containsExactly("The deployment region is Phoenix.");

              String answer =
                  context
                      .getBean(ModelsSpringAiChatModel.class)
                      .call(new Prompt(List.of(new UserMessage("Use the retrieved context."))))
                      .getResult()
                      .getOutput()
                      .getText();
              assertThat(answer).isEqualTo("local answer");
              assertThat(prompt.get())
                  .isEqualTo(
                      """
                      <|im_start|>user
                      Use the retrieved context.<|im_end|>
                      <|im_start|>assistant
                      """);
              assertThat(options.get().temperature()).isZero();
              assertThat(options.get().maxTokens()).isEqualTo(17);
            });
  }

  @Configuration(proxyBeanMethods = false)
  static class LocalModelConfiguration {
    static final AtomicReference<String> PROMPT = new AtomicReference<>();
    static final AtomicReference<SamplingOptions> OPTIONS = new AtomicReference<>();

    @Bean
    TextGenerationModel textGenerationModel() {
      return localModel();
    }

    static TextGenerationModel localModel() {
      return new TextGenerationModel() {
        @Override
        public String modelName() {
          return "spring-interoperability-test";
        }

        @Override
        public BackendDiagnostics diagnostics() {
          return BackendDiagnostics.unavailable("spring-interoperability-test");
        }

        @Override
        public void generate(String prompt, SamplingOptions options, TokenStream stream) {
          PROMPT.set(prompt);
          OPTIONS.set(options);
          stream.onToken("local answer");
          stream.onComplete();
        }
      };
    }
  }

  @AutoConfiguration
  @AutoConfigureAfter(ModelsAutoConfiguration.class)
  static class LateLocalModelAutoConfiguration {
    @Bean
    TextGenerationModel textGenerationModel() {
      return LocalModelConfiguration.localModel();
    }
  }

  @Configuration(proxyBeanMethods = false)
  static class UserProvidedAdapterConfiguration {
    static final String USER_ADAPTER_BEAN_NAME = "customModelsChatModel";

    @Bean(USER_ADAPTER_BEAN_NAME)
    ModelsSpringAiChatModel customModelsChatModel() {
      return new ModelsSpringAiChatModel(LocalModelConfiguration.localModel());
    }
  }

  @Configuration(proxyBeanMethods = false)
  static class RemoteChatModelConfiguration {
    static final String REMOTE_CHAT_MODEL_BEAN_NAME = "remoteChatModel";

    @Bean(REMOTE_CHAT_MODEL_BEAN_NAME)
    ChatModel remoteChatModel() {
      return mock(ChatModel.class);
    }
  }

  @Configuration(proxyBeanMethods = false)
  static class EmbeddingConfiguration {
    @Bean
    EmbeddingModel embeddingModel() {
      return new FixedEmbeddingModel();
    }

    @Bean
    BatchingStrategy batchingStrategy() {
      return documents -> List.of(documents);
    }
  }

  static final class FixedEmbeddingModel implements EmbeddingModel {
    private static final float[] VECTOR = {1.0f, 0.0f};

    @Override
    public float[] embed(Document document) {
      return VECTOR.clone();
    }

    @Override
    public List<float[]> embed(
        List<Document> documents, EmbeddingOptions options, BatchingStrategy batchingStrategy) {
      return documents.stream().map(document -> VECTOR.clone()).toList();
    }

    @Override
    public EmbeddingResponse call(EmbeddingRequest request) {
      List<Embedding> embeddings =
          java.util.stream.IntStream.range(0, request.getInstructions().size())
              .mapToObj(index -> new Embedding(VECTOR.clone(), index))
              .toList();
      return new EmbeddingResponse(embeddings);
    }

    @Override
    public int dimensions() {
      return VECTOR.length;
    }
  }
}
