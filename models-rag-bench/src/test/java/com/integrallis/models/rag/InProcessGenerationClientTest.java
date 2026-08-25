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
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import java.util.List;
import org.junit.jupiter.api.Test;

class InProcessGenerationClientTest {

  @Test
  void measuresInProcessModelsGeneration() {
    RagSamplingProfile sampling = new RagSamplingProfile(0.7, 0.95, 4, 1729L, 1.05);
    try (InProcessGenerationClient client =
        new InProcessGenerationClient("rust-ffm", backend(), 12.5, sampling)) {
      GenerationResult result = client.generate("hello", 8);

      assertThat(client.backend()).isEqualTo("rust-ffm");
      assertThat(client.diagnostics().planVersion()).isEqualTo("fixture-v1");
      assertThat(client.generationControls())
          .containsEntry("temperature", "0.7")
          .containsEntry("topP", "0.95")
          .containsEntry("topK", "4")
          .containsEntry("seed", "1729")
          .containsEntry("repetitionPenalty", "1.05");
      assertThat(result.text()).isEqualTo(" world");
      assertThat(result.inputTokens()).isEqualTo(1);
      assertThat(result.outputTokens()).isEqualTo(2);
      assertThat(result.ttftMillis()).isGreaterThanOrEqualTo(0);
      assertThat(result.totalMillis()).isGreaterThanOrEqualTo(result.ttftMillis());
      assertThat(result.loadMillis()).isEqualTo(12.5);
    }
  }

  @Test
  void preservesAndReportsPromptCacheReuseThroughTimingInstrumentation() {
    PrefixBackend backend = new PrefixBackend();
    try (InProcessGenerationClient client =
        new InProcessGenerationClient(
            "rust-ffm", backend, 12.5, RagSamplingProfile.deterministic())) {
      client.generate("first", 1);
      GenerationResult result = client.generate("second", 1);

      assertThat(result.inputTokens()).isEqualTo(3);
      assertThat(result.cacheReadInputTokens()).isEqualTo(2);
      assertThat(result.cacheWriteInputTokens()).isEqualTo(1);
      assertThat(backend.rewindCheckpoints).containsExactly(2);
    }
  }

  @Test
  void preservesStructuredTemplateControlsForTheModelsTokenizer() {
    ModelPrompt prompt =
        ModelPrompt.builder().control("<|im_start|>").text("hello<|im_end|>").build();
    try (InProcessGenerationClient client =
        new InProcessGenerationClient(
            "pure-java", structuredPromptBackend(prompt), 1.0, RagSamplingProfile.deterministic())) {
      GenerationResult result = client.generate(prompt, 1);

      assertThat(result.inputTokens()).isEqualTo(1);
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

  private static InferenceBackend structuredPromptBackend(ModelPrompt expectedPrompt) {
    Tokenizer tokenizer =
        new Tokenizer() {
          @Override
          public int[] encode(String text) {
            return new int[] {2, 2, 2};
          }

          @Override
          public int[] encode(ModelPrompt prompt) {
            assertThat(prompt.segments()).containsExactlyElementsOf(expectedPrompt.segments());
            return new int[] {2};
          }

          @Override
          public String decode(int[] tokens) {
            return "";
          }

          @Override
          public String decode(int token) {
            return token == 3 ? "answer" : "";
          }

          @Override
          public int vocabSize() {
            return 4;
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
    return new InferenceBackend() {
      @Override
      public String name() {
        return "structured-prompt-stub";
      }

      @Override
      public ModelMetadata metadata() {
        return new ModelMetadata("test", "StructuredPromptStub", 16, 4, 8, 1, 1, 1);
      }

      @Override
      public Tokenizer tokenizer() {
        return tokenizer;
      }

      @Override
      public float[] forward(int token, int position) {
        float[] logits = new float[4];
        logits[3] = 100;
        return logits;
      }

      @Override
      public void close() {}
    };
  }

  private static final class PrefixBackend implements RewindableInferenceBackend {
    private final List<Integer> rewindCheckpoints = new java.util.ArrayList<>();
    private int nextPosition;

    @Override
    public String name() {
      return "prefix-stub";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("test", "PrefixStub", 16, 6, 8, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return new Tokenizer() {
        @Override
        public int[] encode(String text) {
          return "first".equals(text) ? new int[] {2, 3, 4} : new int[] {2, 3, 5};
        }

        @Override
        public String decode(int[] tokens) {
          return "";
        }

        @Override
        public String decode(int token) {
          return token == 4 ? "answer" : "";
        }

        @Override
        public int vocabSize() {
          return 6;
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

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      assertThat(startPosition).isEqualTo(nextPosition);
      nextPosition += tokens.length;
      float[] logits = new float[6];
      logits[4] = 100;
      return logits;
    }

    @Override
    public float[] forward(int token, int position) {
      throw new UnsupportedOperationException("maxTokens=1 does not forward the final token");
    }

    @Override
    public int checkpoint() {
      return nextPosition;
    }

    @Override
    public void rewind(int checkpoint) {
      rewindCheckpoints.add(checkpoint);
      nextPosition = checkpoint;
    }

    @Override
    public void reset() {
      nextPosition = 0;
    }

    @Override
    public void close() {}
  }
}
