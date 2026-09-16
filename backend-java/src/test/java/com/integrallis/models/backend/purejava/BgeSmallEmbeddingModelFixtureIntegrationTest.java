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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.io.IOException;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Real-artifact contract checks for BGE Small's CLS-pooled BERT encoder. */
@Tag("integration")
class BgeSmallEmbeddingModelFixtureIntegrationTest {

  private static final ModelFixtureRequirement BGE_SMALL =
      ModelFixtureRequirement.of("hf://CompendiumLabs/bge-small-en-v1.5-gguf")
          .version("[1.5.0,2.0.0)")
          .variant("f16")
          .backend("pure-java")
          .capability("embedding");

  private static ModelFixtureDescriptor fixture() {
    return ModelFixtureRegistry.fromClasspath().resolve(BGE_SMALL).orElseThrow();
  }

  @Test
  void declaresTheExpectedBertAndClsPoolingContract() throws IOException {
    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(fixture().localPath().orElseThrow(), arena);

      assertThat(file.metadata().getString("general.architecture")).contains("bert");
      assertThat(file.metadata().getUint32("bert.pooling_type")).contains(2);
      assertThat(file.metadata().getUint32("bert.embedding_length")).contains(384);
    }
  }

  @Test
  void emitsAUnitLengthSentenceEmbedding() {
    try (PureJavaBackend backend = PureJavaBackend.load(fixture().localPath().orElseThrow());
        GgufEmbeddingBackend embedding =
            GgufEmbeddingBackend.builder(backend).normalize(true).build()) {
      float[] actual = embedding.embed("retrieval augmented generation");

      assertThat(backend.metadata().modelFamily()).isEqualTo("bert");
      assertThat(backend.supportsSequenceEmbedding()).isTrue();
      assertThat(actual).hasSize(384);
      assertThat(l2Norm(actual)).isBetween(0.9999, 1.0001);
    }
  }

  private static double l2Norm(float[] vector) {
    double squared = 0.0;
    for (float value : vector) {
      squared += (double) value * value;
    }
    return Math.sqrt(squared);
  }
}
