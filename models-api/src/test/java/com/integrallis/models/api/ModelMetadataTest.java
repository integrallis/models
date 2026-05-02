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

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ModelMetadataTest {

  @Nested
  static class Validation {

    @Test
    void nullFamilyRejected() {
      assertThatThrownBy(() -> new ModelMetadata(null, "test", 2048, 32000, 4096, 32, 32, 8))
          .isInstanceOf(NullPointerException.class);
    }

    @Test
    void nullNameRejected() {
      assertThatThrownBy(() -> new ModelMetadata("llama", null, 2048, 32000, 4096, 32, 32, 8))
          .isInstanceOf(NullPointerException.class);
    }

    @Test
    void validMetadataCreated() {
      var meta = new ModelMetadata("llama", "Llama-3.2-1B", 2048, 32000, 2048, 16, 32, 8);

      assertThat(meta.modelFamily()).isEqualTo("llama");
      assertThat(meta.modelName()).isEqualTo("Llama-3.2-1B");
      assertThat(meta.contextLength()).isEqualTo(2048);
      assertThat(meta.vocabSize()).isEqualTo(32000);
      assertThat(meta.embeddingDim()).isEqualTo(2048);
      assertThat(meta.numLayers()).isEqualTo(16);
      assertThat(meta.numHeads()).isEqualTo(32);
      assertThat(meta.numKvHeads()).isEqualTo(8);
    }
  }
}
