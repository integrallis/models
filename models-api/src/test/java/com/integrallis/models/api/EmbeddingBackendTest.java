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
import static org.assertj.core.api.Assertions.assertThatNoException;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class EmbeddingBackendTest {

  @Test
  void defaultBatchEmbeddingPreservesInputOrder() {
    RecordingBackend backend = new RecordingBackend();

    float[][] embeddings = backend.embedAll(List.of("a", "bbb"));

    assertThat(embeddings).hasNumberOfRows(2);
    assertThat(embeddings[0]).containsExactly(1.0f, 0.0f);
    assertThat(embeddings[1]).containsExactly(3.0f, 1.0f);
    assertThat(backend.inputs).containsExactly("a", "bbb");
  }

  @Test
  void listViewContainsTheRowsReturnedByBatchEmbedding() {
    RecordingBackend backend = new RecordingBackend();

    List<float[]> embeddings = backend.embedAllAsList(List.of("ab", "cdef"));

    assertThat(embeddings).hasSize(2);
    assertThat(embeddings.get(0)).containsExactly(2.0f, 0.0f);
    assertThat(embeddings.get(1)).containsExactly(4.0f, 1.0f);
  }

  @Test
  void defaultBatchEmbeddingRejectsNullContainersAndElements() {
    RecordingBackend backend = new RecordingBackend();

    assertThatNullPointerException()
        .isThrownBy(() -> backend.embedAll(null))
        .withMessage("texts must not be null");
    assertThatNullPointerException()
        .isThrownBy(() -> backend.embedAll(java.util.Arrays.asList("valid", null)))
        .withMessage("texts must not contain null");
  }

  @Test
  void defaultCloseRequiresNoResources() {
    assertThatNoException().isThrownBy(new RecordingBackend()::close);
  }

  private static final class RecordingBackend implements EmbeddingBackend {
    private final List<String> inputs = new ArrayList<>();

    @Override
    public int dimension() {
      return 2;
    }

    @Override
    public float[] embed(String text) {
      inputs.add(text);
      return new float[] {text.length(), inputs.size() - 1};
    }
  }
}
