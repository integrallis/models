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
package com.integrallis.models.backend.purejava.gliner;

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.api.SchemaClassificationModel.*;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class GlinerClassificationTest {
  static Task task(boolean multi, double threshold) {
    return new Task("sentiment", List.of(new Label("yes"), new Label("no")), multi, threshold);
  }

  @Test
  void sigmoidIsIndependentAndFallbackIsExplicit() {
    var request = new Request("text", List.of(task(true, .5)));
    var result = GlinerClassificationModel.decode(request, new float[] {-2, -3}).getFirst();
    assertThat(result.thresholdFallback()).isTrue();
    assertThat(result.labels().getFirst().selected()).isTrue();
    assertThat(result.labels().get(1).selected()).isFalse();
    assertThat(result.labels().getFirst().score()).isCloseTo(0.119202922, within(1e-8));
    var boundary = GlinerClassificationModel.decode(request, new float[] {0, 0}).getFirst();
    assertThat(boundary.labels()).allMatch(LabelScore::selected);
    assertThat(boundary.thresholdFallback()).isFalse();
  }

  @Test
  void singleLabelSoftmaxHasStableTieAndRejectsNonFiniteLogits() {
    var request = new Request("text", List.of(task(false, .5)));
    var result = GlinerClassificationModel.decode(request, new float[] {1000, 1000}).getFirst();
    assertThat(result.labels().getFirst().score()).isEqualTo(.5);
    assertThat(result.labels().getFirst().selected()).isTrue();
    assertThat(result.labels().get(1).selected()).isFalse();
    assertThatThrownBy(() -> GlinerClassificationModel.decode(request, new float[] {Float.NaN, 1}))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> GlinerClassificationModel.decode(request, new float[] {1}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void structuralMarkersAreTrackedBySlotAndTextIsNormalized() {
    List<String> pieces = new ArrayList<>();
    var compiler =
        new GlinerSchemaCompiler(
            text -> {
              pieces.add(text);
              return new int[] {1};
            },
            100);
    var request =
        new Request(
            "CAFÉ [L]",
            List.of(
                new Task(
                    "route",
                    List.of(new Label("[L]", "literal [L] in description"), new Label("other")),
                    false,
                    .5)));
    var compiled = compiler.compile(request);
    assertThat(compiled.markerPositions()).containsExactly(4, 6);
    assertThat(pieces)
        .contains(
            "café", "[", "l", "]", ".", "route [DESCRIPTION] [L]: literal [L] in description");
    assertThat(pieces).doesNotContain("[CLS]", "[SEP]");
  }

  @Test
  void overflowAndAmbiguousSchemasAreRejected() {
    var compiler = new GlinerSchemaCompiler(text -> new int[] {1}, 3);
    assertThatThrownBy(() -> compiler.compile(new Request("text", List.of(task(false, .5)))))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("no truncation");
    assertThatThrownBy(
            () -> new Task("x", List.of(new Label("same"), new Label("same")), false, .5))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new Request("x", List.of(task(false, .5), task(true, .5))))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
