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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.Tokenizer;
import java.util.List;
import org.junit.jupiter.api.Test;

class PureJavaGenerationClientTest {

  @Test
  void measuresInProcessModelsGeneration() {
    try (PureJavaGenerationClient client = new PureJavaGenerationClient(backend(), 12.5)) {
      GenerationResult result = client.generate("hello", 8);

      assertThat(client.diagnostics().planVersion()).isEqualTo("fixture-v1");
      assertThat(result.text()).isEqualTo(" world");
      assertThat(result.inputTokens()).isEqualTo(1);
      assertThat(result.outputTokens()).isEqualTo(2);
      assertThat(result.ttftMillis()).isGreaterThanOrEqualTo(0);
      assertThat(result.totalMillis()).isGreaterThanOrEqualTo(result.ttftMillis());
      assertThat(result.loadMillis()).isEqualTo(12.5);
    }
  }

  private static InferenceBackend backend() {
    Tokenizer tokenizer = tokenizer();
    return new InferenceBackend() {
      @Override
      public String name() {
        return "pure-java-test";
      }

      @Override
      public ModelMetadata metadata() {
        return new ModelMetadata("test", "Tiny", 16, 5, 8, 1, 1, 1);
      }

      @Override
      public BackendDiagnostics diagnostics() {
        return new BackendDiagnostics(name(), "fixture-v1", java.util.Map.of(), List.of());
      }

      @Override
      public Tokenizer tokenizer() {
        return tokenizer;
      }

      @Override
      public float[] forward(int token, int position) {
        float[] logits = new float[5];
        logits[position == 0 ? 3 : position == 1 ? 4 : 1] = 100;
        return logits;
      }

      @Override
      public void close() {}
    };
  }

  private static Tokenizer tokenizer() {
    return new Tokenizer() {
      private final List<String> vocab = List.of("<s>", "</s>", "hello", " ", "world");

      @Override
      public int[] encode(String text) {
        return new int[] {2};
      }

      @Override
      public String decode(int[] tokens) {
        return "";
      }

      @Override
      public String decode(int token) {
        return vocab.get(token);
      }

      @Override
      public int vocabSize() {
        return vocab.size();
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
