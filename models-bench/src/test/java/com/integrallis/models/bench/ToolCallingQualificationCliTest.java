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
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.chat.ToolSyntax;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ToolCallingQualificationCliTest {

  private static final String REVISION = "a".repeat(40);

  @TempDir Path temporary;

  @Test
  void parsesAPinnedCandidateAndOutputControls() throws Exception {
    Path artifact = Files.writeString(temporary.resolve("model.gguf"), "fixture");

    ToolCallingQualificationCli.Configuration configuration =
        ToolCallingQualificationCli.parse(
            new String[] {
              "--candidate",
              "qwen3-0.6b",
              "--model",
              artifact.toString(),
              "--models-revision",
              REVISION,
              "--max-tokens",
              "192",
              "--case",
              "currency",
              "--report",
              temporary.resolve("report.json").toString()
            });

    assertThat(configuration.candidate()).isEqualTo(ToolCallingCandidate.QWEN3_06B);
    assertThat(configuration.model()).isEqualTo(artifact);
    assertThat(configuration.maxTokens()).isEqualTo(192);
    assertThat(configuration.caseId()).isEqualTo("currency");
    assertThat(configuration.modelsRevision()).isEqualTo(REVISION);
  }

  @Test
  void requiresCandidateModelAndImmutableRevision() throws Exception {
    Path artifact = Files.writeString(temporary.resolve("model.gguf"), "fixture");

    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                ToolCallingQualificationCli.parse(
                    new String[] {"--model", artifact.toString(), "--models-revision", REVISION}))
        .withMessageContaining("--candidate");
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                ToolCallingQualificationCli.parse(
                    new String[] {"--candidate", "qwen3-0.6b", "--model", artifact.toString()}))
        .withMessageContaining("--models-revision");
  }

  @Test
  void followUpRequiresAConversationalAnswerInsteadOfRawJsonOrAnotherCall() {
    ToolSpec weather =
        new ToolSpec(
            "get-weather-for-zipcode",
            "Gets weather",
            "{\"type\":\"object\",\"properties\":{\"zipcode\":{\"type\":\"string\"}}}");
    ObjectMapper mapper = new ObjectMapper();

    assertThat(
            ToolCallingQualificationCli.evaluateFollowUp(
                    mapper,
                    ToolSyntax.QWEN,
                    List.of(weather),
                    "The weather for 88252 is raining cats and dogs at 78 F.",
                    20)
                .passed())
        .isTrue();
    assertThat(
            ToolCallingQualificationCli.evaluateFollowUp(
                    mapper,
                    ToolSyntax.QWEN,
                    List.of(weather),
                    "{\"zipcode\":\"88252\",\"conditions\":\"Raining cats and dogs\",\"temperatureInFahrenheit\":78}",
                    20)
                .passed())
        .isFalse();
    assertThat(
            ToolCallingQualificationCli.evaluateFollowUp(
                    mapper,
                    ToolSyntax.QWEN,
                    List.of(weather),
                    "<tool_call>{\"name\":\"get-weather-for-zipcode\",\"arguments\":{\"zipcode\":\"88252\"}}</tool_call>",
                    20)
                .passed())
        .isFalse();
  }
}
