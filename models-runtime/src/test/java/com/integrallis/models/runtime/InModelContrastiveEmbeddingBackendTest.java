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
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.Tokenizer;
import org.junit.jupiter.api.Test;

class InModelContrastiveEmbeddingBackendTest {

  @Test
  void tokenizesTextAndDelegatesToTheModelsOwnHead() {
    try (var embeddings = new InModelContrastiveEmbeddingBackend(new FakeAuxiliaryBackend())) {
      assertThat(embeddings.dimension()).isEqualTo(2);
      assertThat(embeddings.embed("abcd")).containsExactly(4.0f, 14.0f);
    }
  }

  private static final class FakeAuxiliaryBackend implements AuxiliaryInferenceBackend {
    private static final Tokenizer TOKENIZER =
        new Tokenizer() {
          @Override
          public int[] encode(String text) {
            return new int[] {text.length(), text.length() + 10};
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
            return 32;
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
      return new ModelMetadata("fake", "fake", 32, 32, 2, 1, 1, 1);
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
      return false;
    }

    @Override
    public float scoreConfidence(int[] tokens) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }
}
