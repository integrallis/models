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
package com.integrallis.models.runtime;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GenerationLoopTest {

  /** Simple mock tokenizer with tokens: 0=BOS, 1=EOS, 2="hello", 3=" ", 4="world", 5="!" */
  private static final Tokenizer MOCK_TOKENIZER =
      new Tokenizer() {
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
          StringBuilder sb = new StringBuilder();
          for (int t : tokens) sb.append(decode(t));
          return sb.toString();
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

  /** Creates a mock backend that returns logits favoring a specific token sequence. */
  private InferenceBackend mockBackend(int[] generatedSequence) {
    return new InferenceBackend() {
      private int callCount = 0;

      @Override
      public String name() {
        return "mock";
      }

      @Override
      public ModelMetadata metadata() {
        return new ModelMetadata("mock", "MockModel", 64, 6, 16, 1, 1, 1);
      }

      @Override
      public Tokenizer tokenizer() {
        return MOCK_TOKENIZER;
      }

      @Override
      public float[] forward(int token, int position) {
        float[] logits = new float[6];
        // After prompt prefill, return logits that strongly favor the next token in sequence
        int promptLen = MOCK_TOKENIZER.encode("hello").length;
        int genIdx = position - promptLen + 1;
        if (genIdx >= 0 && genIdx < generatedSequence.length) {
          logits[generatedSequence[genIdx]] = 100.0f;
        } else {
          logits[1] = 100.0f; // EOS
        }
        callCount++;
        return logits;
      }

      @Override
      public void close() {}
    };
  }

  @Nested
  class BasicGeneration {

    @Test
    void usesOneBackendPrefillForThePrompt() {
      AtomicInteger prefillCalls = new AtomicInteger();
      AtomicInteger forwardCalls = new AtomicInteger();
      InferenceBackend backend =
          new InferenceBackend() {
            @Override
            public String name() {
              return "prefill-tracking";
            }

            @Override
            public ModelMetadata metadata() {
              return new ModelMetadata("mock", "MockModel", 64, 6, 16, 1, 1, 1);
            }

            @Override
            public Tokenizer tokenizer() {
              return MOCK_TOKENIZER;
            }

            @Override
            public float[] prefill(int[] tokens, int startPosition) {
              prefillCalls.incrementAndGet();
              float[] logits = new float[6];
              logits[5] = 100.0f;
              return logits;
            }

            @Override
            public float[] forward(int token, int position) {
              forwardCalls.incrementAndGet();
              float[] logits = new float[6];
              logits[1] = 100.0f;
              return logits;
            }

            @Override
            public void close() {}
          };

      String result =
          new GenerationLoop(backend)
              .generate(
                  "hello world", SamplingOptions.builder().temperature(0.0f).maxTokens(2).build());

      assertThat(result).isEqualTo("!");
      assertThat(prefillCalls).hasValue(1);
      assertThat(forwardCalls).hasValue(1);
    }

    @Test
    void generatesUntilEos() {
      // Will generate " world" then EOS
      InferenceBackend backend = mockBackend(new int[] {3, 4, 1}); // " ", "world", EOS
      GenerationLoop loop = new GenerationLoop(backend);

      String result =
          loop.generate("hello", SamplingOptions.builder().temperature(0.0f).maxTokens(10).build());

      assertThat(result).isEqualTo(" world");
    }

    @Test
    void generatesUntilAnyEndOfGenerationToken() {
      Tokenizer tokenizer =
          new Tokenizer() {
            @Override
            public int[] encode(String text) {
              return new int[] {2};
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
              return switch (token) {
                case 2 -> "hello";
                case 3 -> " world";
                case 4 -> "<|im_end|>";
                case 5 -> " leaked";
                default -> "";
              };
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

            @Override
            public boolean isEndOfGeneration(int token) {
              return token == 1 || token == 4;
            }
          };
      InferenceBackend backend =
          new InferenceBackend() {
            private int generated;

            @Override
            public String name() {
              return "multi-eog";
            }

            @Override
            public ModelMetadata metadata() {
              return new ModelMetadata("mock", "MultiEog", 64, 6, 16, 1, 1, 1);
            }

            @Override
            public Tokenizer tokenizer() {
              return tokenizer;
            }

            @Override
            public float[] prefill(int[] tokens, int startPosition) {
              return logitsFor(3);
            }

            @Override
            public float[] forward(int token, int position) {
              return logitsFor(++generated == 1 ? 4 : 5);
            }

            @Override
            public void close() {}

            private float[] logitsFor(int token) {
              float[] logits = new float[6];
              logits[token] = 100.0f;
              return logits;
            }
          };

      String result =
          new GenerationLoop(backend)
              .generate("hello", SamplingOptions.builder().temperature(0.0f).maxTokens(5).build());

      assertThat(result).isEqualTo(" world");
    }

    @Test
    void generatesUntilMaxTokens() {
      // Will generate "!!!!..." until maxTokens
      InferenceBackend backend = mockBackend(new int[] {5, 5, 5, 5, 5, 5, 5, 5, 5, 5});
      GenerationLoop loop = new GenerationLoop(backend);

      String result =
          loop.generate("hello", SamplingOptions.builder().temperature(0.0f).maxTokens(3).build());

      assertThat(result).isEqualTo("!!!");
    }

    @Test
    void doesNotForwardTheFinalTokenWhenTheLimitIsReached() {
      AtomicInteger forwardCalls = new AtomicInteger();
      InferenceBackend backend =
          new InferenceBackend() {
            @Override
            public String name() {
              return "limit-tracking";
            }

            @Override
            public ModelMetadata metadata() {
              return new ModelMetadata("mock", "MockModel", 64, 6, 16, 1, 1, 1);
            }

            @Override
            public Tokenizer tokenizer() {
              return MOCK_TOKENIZER;
            }

            @Override
            public float[] prefill(int[] tokens, int startPosition) {
              return exclamationLogits();
            }

            @Override
            public float[] forward(int token, int position) {
              forwardCalls.incrementAndGet();
              return exclamationLogits();
            }

            @Override
            public void close() {}

            private float[] exclamationLogits() {
              float[] logits = new float[6];
              logits[5] = 100.0f;
              return logits;
            }
          };

      String result =
          new GenerationLoop(backend)
              .generate("hello", SamplingOptions.builder().temperature(0.0f).maxTokens(3).build());

      assertThat(result).isEqualTo("!!!");
      assertThat(forwardCalls).hasValue(2);
    }
  }

  @Nested
  class Streaming {

    @Test
    void streamingCallbackReceivesEachToken() {
      InferenceBackend backend = mockBackend(new int[] {3, 4, 1}); // " ", "world", EOS
      GenerationLoop loop = new GenerationLoop(backend);

      List<String> tokens = new ArrayList<>();
      boolean[] completed = {false};

      loop.generate(
          "hello",
          SamplingOptions.builder().temperature(0.0f).maxTokens(10).build(),
          new TokenStream() {
            @Override
            public void onToken(String token) {
              tokens.add(token);
            }

            @Override
            public void onComplete() {
              completed[0] = true;
            }

            @Override
            public void onError(Throwable t) {}
          });

      assertThat(tokens).containsExactly(" ", "world");
      assertThat(completed[0]).isTrue();
    }
  }

  @Nested
  class ErrorHandling {

    @Test
    void emptyPromptThrows() {
      InferenceBackend backend = mockBackend(new int[] {1});
      GenerationLoop loop = new GenerationLoop(backend);

      assertThatThrownBy(
              () ->
                  loop.generate(
                      "", SamplingOptions.builder().temperature(0.0f).maxTokens(10).build()))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("prompt");
    }
  }
}
