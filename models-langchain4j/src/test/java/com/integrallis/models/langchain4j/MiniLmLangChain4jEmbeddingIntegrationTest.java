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

import com.integrallis.models.backend.purejava.GgufEmbeddingBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class MiniLmLangChain4jEmbeddingIntegrationTest {

  @Test
  void embedsThroughTheLangChain4jContract() {
    Path modelPath = Path.of(System.getProperty("models.fixtures.miniLm"));
    var backend =
        GgufEmbeddingBackend.builder(PureJavaBackend.load(modelPath)).normalize(true).build();
    try (var model = new ModelsEmbeddingModel(backend)) {
      float[] transit =
          model.embed("An accessible public transit route to the airport").content().vector();
      float[] related =
          model.embed("How can a wheelchair rider reach the airport by train?").content().vector();
      float[] unrelated =
          model.embed("Java records are shallowly immutable data carriers.").content().vector();

      assertThat(model.dimension()).isEqualTo(384);
      assertThat(transit).hasSize(384);
      assertThat(cosine(transit, related)).isGreaterThan(cosine(transit, unrelated));
    }
  }

  private static double cosine(float[] left, float[] right) {
    double dot = 0.0;
    for (int index = 0; index < left.length; index++) {
      dot += (double) left[index] * right[index];
    }
    return dot;
  }
}
