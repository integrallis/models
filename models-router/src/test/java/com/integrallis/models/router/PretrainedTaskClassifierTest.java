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

import java.io.StringReader;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class PretrainedTaskClassifierTest {

  private static final String CORPUS =
      """
      train\tcode\tmbpp\twrite a function that reverses a list
      train\tcode\tmbpp\timplement binary search over an array
      train\tmath\tgsm8k\twhat is seventeen percent of three hundred
      train\tmath\tgsm8k\tsolve for x when two x plus five equals nine
      """;

  /** Word-presence axes: deterministic, and separable enough to exercise the decision logic. */
  private static final TaskIndexBuilder.TaskEmbedder EMBEDDER =
      text -> {
        String lower = text.toLowerCase(java.util.Locale.ROOT);
        String[] axes = {"function", "list", "search", "array", "percent", "solve", "equals"};
        float[] vector = new float[axes.length + 1];
        for (int index = 0; index < axes.length; index++) {
          vector[index] = lower.contains(axes[index]) ? 1.0f : 0.0f;
        }
        vector[axes.length] = 0.01f; // keeps unrelated text non-zero so cosine stays defined
        return vector;
      };

  private static TaskIndex index(Path directory) {
    TaskIndexBuilder.build(
        TaskExemplars.parse(new StringReader(CORPUS)), EMBEDDER, "fake-embedder-v1", directory);
    return TaskIndex.open(directory);
  }

  @Test
  void classifiesAQueryAsTheTaskOfItsNearestPrompt(@TempDir Path directory) {
    try (TaskIndex index = index(directory)) {
      PretrainedTaskClassifier classifier = PretrainedTaskClassifier.using(index, EMBEDDER, 0.35);

      assertThat(classifier.classify("implement binary search")).isEqualTo("code");
      assertThat(classifier.classify("solve for x")).isEqualTo("math");
    }
  }

  @Test
  void leavesAnUnfamiliarQueryUnclassified(@TempDir Path directory) {
    // Guessing a task for an unrecognised request would silently apply the wrong quality column;
    // null lets the router fall back to cost, latency and reliability.
    try (TaskIndex index = index(directory)) {
      PretrainedTaskClassifier classifier = PretrainedTaskClassifier.using(index, EMBEDDER, 0.35);

      assertThat(classifier.classify("what is the weather in Denver")).isNull();
    }
  }

  @Test
  void treatsBlankInputAsUnclassified(@TempDir Path directory) {
    try (TaskIndex index = index(directory)) {
      PretrainedTaskClassifier classifier = PretrainedTaskClassifier.using(index, EMBEDDER, 0.35);

      assertThat(classifier.classify("   ")).isNull();
      assertThat(classifier.classify(null)).isNull();
    }
  }

  @Test
  void refusesAQueryEmbeddedByADifferentlyShapedModel(@TempDir Path directory) {
    // A width mismatch is the only mechanically detectable form of "wrong embedding model", so it
    // has to be an error rather than a silent miss.
    try (TaskIndex index = index(directory)) {
      assertThatThrownBy(() -> index.nearest(new float[3]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("same embedding model");
    }
  }

  @Test
  void reportsWhatTheIndexWasBuiltWith(@TempDir Path directory) {
    try (TaskIndex index = index(directory)) {
      assertThat(index.embeddingModelId()).isEqualTo("fake-embedder-v1");
      assertThat(index.dimension()).isEqualTo(8);
      assertThat(index.taskNames()).containsExactlyInAnyOrder("code", "math");
    }
  }

  @Test
  void rejectsAThresholdOutsideTheCosineRange(@TempDir Path directory) {
    try (TaskIndex index = index(directory)) {
      assertThatThrownBy(() -> PretrainedTaskClassifier.using(index, EMBEDDER, 1.5))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("within [-1, 1]");
    }
  }
}
