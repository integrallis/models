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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.Tokenizer;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class JsonSchemaConstraintTest {

  private static final Tokenizer JSON_TOKENIZER =
      new Tokenizer() {
        private final String[] vocab = {
          "<s>", "</s>", "prompt", "{", "\"status\"", ":", "\"ok\"", "\"failed\"", "}", " nope"
        };

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
          return vocab[token];
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

  @Test
  void allowsTokenFragmentsThatKeepOutputOnASchemaAlternative() {
    JsonSchemaConstraint constraint =
        JsonSchemaConstraint.requiredStringEnums(
            JSON_TOKENIZER, Map.of("status", List.of("ok", "failed")));

    assertThat(constraint.allows(3)).isTrue();
    assertThat(constraint.allows(9)).isFalse();
    constraint.accept(3);
    constraint.accept(4);
    constraint.accept(5);

    assertThat(constraint.allows(6)).isTrue();
    assertThat(constraint.allows(9)).isFalse();
  }

  @Test
  void generationCanForceACanonicalJsonObjectDespiteInvalidLogitBias() {
    Map<String, List<String>> schema = new LinkedHashMap<>();
    schema.put("status", List.of("ok"));
    GenerationLoop loop = new GenerationLoop(jsonBackendFavoringInvalidTokens());

    String generated =
        loop.generate(
            "prompt",
            SamplingOptions.builder().temperature(0.0f).maxTokens(8).build(),
            JsonSchemaConstraint.requiredStringEnums(JSON_TOKENIZER, schema));

    assertThat(generated).isEqualTo("{\"status\":\"ok\"}");
  }

  private static InferenceBackend jsonBackendFavoringInvalidTokens() {
    return new InferenceBackend() {
      @Override
      public String name() {
        return "json-schema-stub";
      }

      @Override
      public ModelMetadata metadata() {
        return new ModelMetadata(
            "stub", "JsonSchemaStub", 64, JSON_TOKENIZER.vocabSize(), 16, 1, 1, 1);
      }

      @Override
      public Tokenizer tokenizer() {
        return JSON_TOKENIZER;
      }

      @Override
      public float[] forward(int token, int position) {
        return logits();
      }

      @Override
      public float[] prefill(int[] tokens, int startPosition) {
        return logits();
      }

      @Override
      public void close() {}

      private float[] logits() {
        float[] logits = new float[JSON_TOKENIZER.vocabSize()];
        logits[9] = 100.0f;
        for (int token = 3; token <= 8; token++) {
          logits[token] = 10.0f - token;
        }
        return logits;
      }
    };
  }
}
