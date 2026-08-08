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
package com.integrallis.models.router;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.io.IOException;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TaskIndexBuilderTest {

  private static final String CORPUS =
      """
      train\tcode\tmbpp\twrite a function that reverses a list
      train\tcode\tmbpp\timplement binary search
      train\tmath\tgsm8k\twhat is seventeen percent of three hundred
      train\tmath\tgsm8k\tsolve for x in two x plus five equals nine
      eval\tcode\tmbpp\tsort an array in place
      """;

  /** Deterministic stand-in for a real model: the axes are word presence, not learned features. */
  private static final TaskIndexBuilder.TaskEmbedder EMBEDDER =
      text -> {
        String lower = text.toLowerCase(java.util.Locale.ROOT);
        String[] axes = {"function", "list", "search", "percent", "solve", "array"};
        float[] vector = new float[axes.length];
        for (int index = 0; index < axes.length; index++) {
          vector[index] = lower.contains(axes[index]) ? 1.0f : 0.0f;
        }
        vector[0] += 0.01f; // keeps every vector non-zero so cosine stays defined
        return vector;
      };

  private static TaskExemplars corpus() {
    return TaskExemplars.parse(new StringReader(CORPUS));
  }

  @Test
  void indexesEveryTrainingPromptAndLeavesTheHeldOutSplitOut(@TempDir Path directory)
      throws IOException {
    int indexed = TaskIndexBuilder.build(corpus(), EMBEDDER, "fake-embedder-v1", directory);

    assertThat(indexed).isEqualTo(4);
    try (VectorCollection collection = open(directory, 6)) {
      assertThat(collection.size()).isEqualTo(4);
    }
  }

  @Test
  void recordsTheModelAndDimensionItWasBuiltWith(@TempDir Path directory) throws IOException {
    // An index searched with a different embedding model returns confident nonsense rather than
    // an error, so what produced it has to travel with it.
    TaskIndexBuilder.build(corpus(), EMBEDDER, "fake-embedder-v1", directory);

    Properties manifest = new Properties();
    try (var reader = Files.newBufferedReader(directory.resolve(TaskIndexBuilder.MANIFEST))) {
      manifest.load(reader);
    }

    assertThat(manifest.getProperty("embeddingModelId")).isEqualTo("fake-embedder-v1");
    assertThat(manifest.getProperty("dimension")).isEqualTo("6");
    assertThat(manifest.getProperty("prompts")).isEqualTo("4");
    assertThat(manifest.getProperty("tasks")).contains("code").contains("math");
  }

  @Test
  void carriesTheTaskLabelOnEveryEntry(@TempDir Path directory) throws IOException {
    TaskIndexBuilder.build(corpus(), EMBEDDER, "fake-embedder-v1", directory);

    try (VectorCollection collection = open(directory, 6)) {
      SearchResult result =
          collection.search(
              SearchRequest.builder(EMBEDDER.embed("implement binary search"), 1)
                  .includeMetadata(true)
                  .build());

      assertThat(result.hits()).hasSize(1);
      assertThat(result.hits().get(0).document().metadata().get(TaskIndexBuilder.TASK_FIELD))
          .isEqualTo(new MetadataValue.Str("code"));
    }
  }

  @Test
  void rejectsAnEmbedderThatChangesDimensionPartWay(@TempDir Path directory) {
    // A ragged index cannot be searched at all; failing here names the cause.
    var ragged =
        new TaskIndexBuilder.TaskEmbedder() {
          private int calls;

          @Override
          public float[] embed(String text) {
            return new float[++calls == 1 ? 4 : 8];
          }
        };

    assertThatThrownBy(() -> TaskIndexBuilder.build(corpus(), ragged, "ragged", directory))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dimensions for prompt");
  }

  @Test
  void rejectsAnEmbedderReturningNothing(@TempDir Path directory) {
    assertThatThrownBy(
            () -> TaskIndexBuilder.build(corpus(), text -> new float[0], "empty", directory))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("no vector");
  }

  private static VectorCollection open(Path directory, int dimension) {
    return VectorCollection.builder()
        .dimension(dimension)
        .metric(SimilarityFunction.COSINE)
        .indexType(IndexType.FLAT)
        .storagePath(directory.toAbsolutePath())
        .build();
  }
}
