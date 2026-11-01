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

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.api.SchemaClassificationModel.*;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class SchemaClassificationModelTest {
  @Test
  void ownsRequestCollectionsAndPreservesLabelDescriptionsAndOrder() {
    var labels =
        new ArrayList<>(List.of(new Label("billing", "Invoice or payment"), new Label("support")));
    var task = new Task("intent", labels, false, .5);
    var tasks = new ArrayList<>(List.of(task));
    var request = new Request("", tasks);
    labels.clear();
    tasks.clear();
    assertThat(request.text()).isEmpty();
    assertThat(request.tasks()).containsExactly(task);
    assertThat(task.labels()).extracting(Label::name).containsExactly("billing", "support");
    assertThat(task.labels().getFirst().description()).isEqualTo("Invoice or payment");
    assertThat(task.labels().get(1).description()).isEmpty();
    assertThatThrownBy(() -> task.labels().clear())
        .isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> request.tasks().clear())
        .isInstanceOf(UnsupportedOperationException.class);
  }

  @Test
  void rejectsAmbiguousNamesAndInvalidSchemaLimits() {
    assertThatThrownBy(() -> new Label(" ")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new Label("valid", null)).isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> new Task(" ", List.of(new Label("yes")), false, .5))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new Task("intent", List.of(), false, .5))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                new Task(
                    "intent", List.of(new Label("yes"), new Label("yes", "different")), false, .5))
        .isInstanceOf(IllegalArgumentException.class);
    var labels =
        java.util.stream.IntStream.range(0, 257).mapToObj(i -> new Label("label" + i)).toList();
    assertThatThrownBy(() -> new Task("intent", labels, false, .5))
        .isInstanceOf(IllegalArgumentException.class);
    var task = new Task("intent", List.of(new Label("yes")), false, .5);
    assertThatThrownBy(() -> new Request("text", List.of()))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new Request("text", List.of(task, task)))
        .isInstanceOf(IllegalArgumentException.class);
    var tasks =
        java.util.stream.IntStream.range(0, 33)
            .mapToObj(i -> new Task("task" + i, List.of(new Label("yes")), false, .5))
            .toList();
    assertThatThrownBy(() -> new Request("text", tasks))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void validatesThresholdsAndScoresWithoutNormalizingIndependentLabels() {
    for (double invalid : new double[] {-.01, 1.01, Double.NaN, Double.POSITIVE_INFINITY}) {
      assertThatThrownBy(() -> new Task("tags", List.of(new Label("x")), true, invalid))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> new LabelScore("x", invalid, false))
          .isInstanceOf(IllegalArgumentException.class);
    }
    for (double boundary : new double[] {0, 1}) {
      assertThat(new Task("tags", List.of(new Label("x")), true, boundary).threshold())
          .isEqualTo(boundary);
      assertThat(new LabelScore("x", boundary, true).score()).isEqualTo(boundary);
    }
    var labels =
        new ArrayList<>(List.of(new LabelScore("a", .9, true), new LabelScore("b", .8, true)));
    var result = new Result("tags", labels, false);
    labels.clear();
    assertThat(result.labels()).hasSize(2);
    assertThat(result.labels()).allMatch(LabelScore::selected);
    assertThat(result.thresholdFallback()).isFalse();
    assertThat(new Result("tags", result.labels(), true).thresholdFallback()).isTrue();
    assertThatThrownBy(() -> result.labels().clear())
        .isInstanceOf(UnsupportedOperationException.class);
  }
}
