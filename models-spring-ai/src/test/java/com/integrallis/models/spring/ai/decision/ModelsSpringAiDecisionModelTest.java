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
package com.integrallis.models.spring.ai.decision;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.HiddenStateInferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.decisions.AnswerSpace;
import com.integrallis.models.decisions.Choice;
import com.integrallis.models.decisions.DecisionHead;
import com.integrallis.models.decisions.Verdict;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.springframework.ai.decision.DecisionPrompt;
import org.springframework.ai.decision.answer.ChoiceAnswer;
import org.springframework.ai.decision.question.ChoiceQuestion;

/** Tests for the Spring AI 2.1 {@code DecisionModel} adapter. */
class ModelsSpringAiDecisionModelTest {

  @Test
  void rejectsBackendWithoutHiddenStateSupport() {
    var backend =
        new FakeBackend() {
          @Override
          public boolean supportsHiddenState() {
            return false;
          }
        };
    assertThatThrownBy(
            () -> new ModelsSpringAiDecisionModel(backend, Map.of("q", dummyHead())))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("does not support hidden states");
  }

  @Test
  void textInputProducesChoiceAnswerWithProbabilities() {
    var backend = new FakeBackend();
    var head =
        new DecisionHead() {
          @Override
          public AnswerSpace space() {
            return new Choice("sentiment", java.util.List.of("positive", "negative"));
          }

          @Override
          public double[] logits(float[] hiddenState) {
            // Higher logit for "positive"
            return new double[] {2.0, 1.0};
          }
        };

    var model = new ModelsSpringAiDecisionModel(backend, Map.of("sentiment", head), 1.0);

    DecisionPrompt prompt =
        DecisionPrompt.builder()
            .text("Movie review: The film was great")
            .question(
                "sentiment", ChoiceQuestion.of("What is the sentiment of this review?", "positive", "negative"))
            .build();

    var response = model.call(prompt);

    assertThat(response.hasAnswer("sentiment")).isTrue();
    ChoiceAnswer answer = response.getChoice("sentiment");
    assertThat(answer.getValue()).isEqualTo("positive");
    assertThat(answer.getConfidence()).isBetween(0.0, 1.0);
    assertThat(answer.getProbabilities()).containsKeys("positive", "negative");
    assertThat(answer.getProbability("positive")).isGreaterThan(answer.getProbability("negative"));
    assertThat(answer.getProbability("positive")).isGreaterThan(0.5);
  }

  @Test
  void missingHeadThrows() {
    var backend = new FakeBackend();
    var model = new ModelsSpringAiDecisionModel(backend, Map.of());

    DecisionPrompt prompt =
        DecisionPrompt.builder()
            .text("hello")
            .question("unknown", ChoiceQuestion.of("question?", "a", "b"))
            .build();

    assertThatThrownBy(() -> model.call(prompt))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("No DecisionHead configured");
  }

  static DecisionHead dummyHead() {
    return new DecisionHead() {
      @Override
      public AnswerSpace space() {
        return new Choice("yes or no", java.util.List.of("yes", "no"));
      }

      @Override
      public double[] logits(float[] hiddenState) {
        return new double[] {0, 0};
      }
    };
  }

  /** A fake backend that tokenizes text by ASCII codes and returns deterministic hidden states. */
  static class FakeBackend implements HiddenStateInferenceBackend {

    @Override
    public String name() {
      return "fake";
    }

    @Override
    public ModelMetadata metadata() {
      return null;
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("fake");
    }

    @Override
    public Tokenizer tokenizer() {
      return new Tokenizer() {
        @Override
        public int[] encode(String text) {
          if (text == null || text.isEmpty()) {
            return new int[0];
          }
          int[] tokens = new int[text.length()];
          for (int i = 0; i < text.length(); i++) {
            tokens[i] = text.charAt(i);
          }
          return tokens;
        }

        @Override
        public int[] encode(com.integrallis.models.api.ModelPrompt prompt) {
          throw new UnsupportedOperationException();
        }

        @Override
        public String decode(int[] tokens) {
          return "";
        }

        @Override
        public String decode(int token) {
          return "";
        }

        @Override
        public int tokenId(String token) {
          return 0;
        }

        @Override
        public int vocabSize() {
          return 256;
        }

        @Override
        public int bosToken() {
          return 1;
        }

        @Override
        public int eosToken() {
          return 2;
        }

        @Override
        public int[] endOfGenerationTokenIds() {
          return new int[] {2};
        }

        @Override
        public boolean isEndOfGeneration(int token) {
          return token == 2;
        }
      };
    }

    @Override
    public float[] forward(int token, int position) {
      return new float[0];
    }

    @Override
    public boolean supportsHiddenState() {
      return true;
    }

    @Override
    public float[] forwardHiddenState(int token, int position) {
      // Deterministic: the hidden state is a vector where the first element is the token id
      return new float[] {token, position};
    }

    @Override
    public float[] prefillHiddenState(int[] tokens, int startPosition) {
      // Return hidden state of the last token
      if (tokens.length == 0) {
        return new float[] {0, 0};
      }
      return forwardHiddenState(tokens[tokens.length - 1], startPosition + tokens.length - 1);
    }

    @Override
    public void close() {}
  }
}
