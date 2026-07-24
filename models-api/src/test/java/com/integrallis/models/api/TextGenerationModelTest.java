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
package com.integrallis.models.api;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class TextGenerationModelTest {

  @Test
  void defaultCollectorJoinsStreamedTokens() {
    try (TextGenerationModel model = modelGenerating("grounded", " answer")) {
      assertThat(model.generate("prompt", SamplingOptions.builder().build()))
          .isEqualTo("grounded answer");
      assertThat(model.modelName()).isEqualTo("test-model");
      assertThat(model.diagnostics().backend()).isEqualTo("test");
    }
  }

  @Test
  void defaultCollectorPropagatesStreamFailure() {
    TextGenerationModel model =
        new TextGenerationModel() {
          @Override
          public String modelName() {
            return "test-model";
          }

          @Override
          public BackendDiagnostics diagnostics() {
            return BackendDiagnostics.unavailable("test");
          }

          @Override
          public void generate(String prompt, SamplingOptions options, TokenStream stream) {
            stream.onError(new IllegalArgumentException("bad generation"));
          }
        };

    assertThatThrownBy(() -> model.generate("prompt", SamplingOptions.builder().build()))
        .isInstanceOf(IllegalStateException.class)
        .hasCauseInstanceOf(IllegalArgumentException.class);
  }

  private static TextGenerationModel modelGenerating(String... tokens) {
    return new TextGenerationModel() {
      @Override
      public String modelName() {
        return "test-model";
      }

      @Override
      public BackendDiagnostics diagnostics() {
        return BackendDiagnostics.unavailable("test");
      }

      @Override
      public void generate(String prompt, SamplingOptions options, TokenStream stream) {
        for (String token : tokens) {
          stream.onToken(token);
        }
        stream.onComplete();
      }
    };
  }
}
