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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.List;
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
