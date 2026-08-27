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

import com.integrallis.models.api.AuxiliaryInferenceBackend;
import com.integrallis.models.api.AuxiliaryTextGenerationModel;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.Tokenizer;
import org.junit.jupiter.api.Test;

class RuntimeAuxiliaryInferenceTest {

  @Test
  void exposesOptionalInModelHeadsWithoutLosingSegmentedTokenization() {
    AuxiliaryTextGenerationModel model = new RuntimeTextGenerationModel(new FakeBackend());
    ModelPrompt prompt = ModelPrompt.builder().control("<control>").text("query").build();

    assertThat(model.supportsContrastiveEncoding()).isTrue();
    assertThat(model.contrastiveDimension()).isEqualTo(2);
    assertThat(model.encodeContrastive(prompt)).containsExactly(7.0f, 8.0f);
    assertThat(model.supportsConfidenceScoring()).isTrue();
    assertThat(model.scoreConfidence(prompt)).isEqualTo(0.78f);
  }

  private static final class FakeBackend implements AuxiliaryInferenceBackend {
    private static final Tokenizer TOKENIZER =
        new Tokenizer() {
          @Override
          public int[] encode(String text) {
            return new int[] {1};
          }

          @Override
          public int[] encode(ModelPrompt prompt) {
            return new int[] {7, 8};
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
          public int vocabSize() {
            return 16;
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
      return "fake";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("fake", "fake", 16, 16, 2, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return TOKENIZER;
    }

    @Override
    public float[] forward(int token, int position) {
      throw new AssertionError("generation is not used");
    }

    @Override
    public boolean supportsContrastiveEncoding() {
      return true;
    }

    @Override
    public int contrastiveDimension() {
      return 2;
    }

    @Override
    public float[] encodeContrastive(int[] tokens) {
      return new float[] {tokens[0], tokens[1]};
    }

    @Override
    public boolean supportsConfidenceScoring() {
      return true;
    }

    @Override
    public float scoreConfidence(int[] tokens) {
      return (tokens[0] * 10 + tokens[1]) / 100.0f;
    }

    @Override
    public void close() {}
  }
}
