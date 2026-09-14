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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import org.junit.jupiter.api.Test;

class SemanticApplicabilityDiagnosticCliTest {

  @Test
  void rendersOnlyUserIntentAndDeclaredCapabilityText() throws Exception {
    ObjectMapper mapper = new ObjectMapper();
    var record =
        mapper.readTree(
            """
            {
              "messages":[
                {"role":"system","content":"untrusted wrapper"},
                {"role":"user","content":"Weather for 88252?"}
              ],
              "tools":[{"type":"function","function":{
                "name":"get_weather",
                "description":"Gets weather by ZIP code.",
                "parameters":{"type":"object","properties":{
                  "zipcode":{"type":"string","description":"US ZIP code"}
                },"required":["zipcode"]}
              }}]
            }
            """);

    assertThat(SemanticApplicabilityDiagnosticCli.userIntent(record))
        .isEqualTo("Weather for 88252?");
    assertThat(SemanticApplicabilityDiagnosticCli.toolCapabilities(record))
        .containsExactly(
            "get weather. Gets weather by ZIP code. zipcode. US ZIP code. Required: zipcode.");
  }

  @Test
  void selectsThresholdsFromCalibrationRowsOnly() {
    List<SemanticApplicabilityDiagnosticCli.Observation> observations =
        List.of(
            observation("c1", true, true, 0.9, 4),
            observation("n1", true, false, 0.1, -2),
            observation("screen-call", false, true, -100, -100),
            observation("screen-none", false, false, 100, 100));

    SemanticApplicabilityDiagnosticCli.Selection selection =
        SemanticApplicabilityDiagnosticCli.select(observations);

    assertThat(selection.semanticThreshold()).isEqualTo(0.5);
    assertThat(selection.marginThreshold()).isEqualTo(1.0);
    assertThat(selection.calibration().balancedAccuracy()).isEqualTo(1.0);
    assertThat(selection.screen().balancedAccuracy()).isZero();
  }

  @Test
  void conjunctionRejectsAHighMarginWhenSemanticRelevanceIsLow() {
    var observation = observation("n1", true, false, 0.2, 8);

    assertThat(SemanticApplicabilityDiagnosticCli.predictsCall(observation, 0.5, 1.0)).isFalse();
  }

  @Test
  void deduplicatesComparisonArmsOnlyWhenTheirInputsMatch() throws Exception {
    ObjectMapper mapper = new ObjectMapper();
    var base =
        mapper.readTree(
            """
            {"id":"case-1","mode":"base","messages":[{"role":"user","content":"Hi"}],
             "tools":[{"type":"function","function":{"name":"hello","parameters":{}}}]}
            """);
    var adapter = base.deepCopy();
    ((com.fasterxml.jackson.databind.node.ObjectNode) adapter).put("mode", "adapter");

    assertThat(SemanticApplicabilityDiagnosticCli.uniqueInputs(List.of(base, adapter)))
        .containsExactly(base);

    ((com.fasterxml.jackson.databind.node.ObjectNode) adapter.path("messages").get(0))
        .put("content", "Changed");
    org.assertj.core.api.Assertions.assertThatThrownBy(
            () -> SemanticApplicabilityDiagnosticCli.uniqueInputs(List.of(base, adapter)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("conflicting duplicate");
  }

  private static SemanticApplicabilityDiagnosticCli.Observation observation(
      String id, boolean calibration, boolean callExpected, double semantic, double margin) {
    return new SemanticApplicabilityDiagnosticCli.Observation(
        id,
        callExpected ? "simple" : "irrelevance",
        calibration ? "calibration" : "screen",
        callExpected,
        semantic,
        margin);
  }
}
