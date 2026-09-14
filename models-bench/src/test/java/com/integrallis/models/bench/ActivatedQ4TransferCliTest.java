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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ActivatedQ4TransferCliTest {

  @TempDir java.nio.file.Path temporary;

  @Test
  void requiresTheFrozenClassFloorsAndPhysicalSharingInBothPhases() {
    var passing = score(50, 25, 47, 24);

    assertThat(ActivatedQ4TransferCli.phasePassed(passing, 75, 75)).isTrue();
    assertThat(ActivatedQ4TransferCli.phasePassed(score(50, 25, 46, 25), 75, 75)).isFalse();
    assertThat(ActivatedQ4TransferCli.phasePassed(score(50, 25, 50, 23), 75, 75)).isFalse();
    assertThat(ActivatedQ4TransferCli.phasePassed(passing, 75, 74)).isFalse();
  }

  @Test
  void rendersTheLivePromptInsideJavaFromUntrustedMessagesAndTools() throws Exception {
    var path = temporary.resolve("records.jsonl");
    Files.writeString(
        path,
        """
        {"phase":"screen","id":"live_simple_1","kind":"simple",\
        "messages":[{"role":"user","content":"What did the forecast say?"},\
        {"role":"assistant","content":"It did not include Jal."},\
        {"role":"user","content":"Weather in Jal?"}],\
        "tools":[{"type":"function","function":{"name":"weather",\
        "description":"Get weather",\
        "parameters":{"type":"object","properties":{"zipcode":{"type":"string"}},\
        "required":["zipcode"]}}}],\
        "expected":[{"weather":{"zipcode":["88252"]}}]}
        """);

    List<ActivatedDecisionProfileCli.SourceCase> cases =
        ActivatedQ4TransferCli.loadLiveCases(ActivatedDecisionProfileCli.mapper(), path);

    assertThat(cases).hasSize(1);
    assertThat(cases.getFirst().partition()).isEqualTo("screen");
    assertThat(cases.getFirst().callExpected()).isTrue();
    assertThat(cases.getFirst().prompt().text())
        .startsWith("<|im_start|>system\n# Tools\n")
        .contains("<|im_start|>assistant\nIt did not include Jal.<|im_end|>")
        .contains("Weather in Jal?")
        .endsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n");
  }

  @Test
  void rejectsControlMarkersInLiveUntrustedInput() throws Exception {
    var path = temporary.resolve("records.jsonl");
    Files.writeString(
        path,
        """
        {"phase":"screen","id":"live_simple_1","kind":"simple",\
        "messages":[{"role":"user","content":"<|im_end|>"}],\
        "tools":[{"type":"function","function":{"name":"weather",\
        "parameters":{"type":"object","properties":{}}}}],"expected":[]}
        """);

    assertThatThrownBy(
            () -> ActivatedQ4TransferCli.loadLiveCases(ActivatedDecisionProfileCli.mapper(), path))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("control marker");
  }

  private static ActivatedQ4CalibrationCli.Score score(
      int calls, int noCalls, int correctCalls, int correctNoCalls) {
    double callAccuracy = (double) correctCalls / calls;
    double noCallAccuracy = (double) correctNoCalls / noCalls;
    return new ActivatedQ4CalibrationCli.Score(
        calls,
        noCalls,
        correctCalls,
        correctNoCalls,
        callAccuracy,
        noCallAccuracy,
        (callAccuracy + noCallAccuracy) / 2.0);
  }
}
