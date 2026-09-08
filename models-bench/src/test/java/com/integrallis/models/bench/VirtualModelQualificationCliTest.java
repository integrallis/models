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

import com.integrallis.models.api.ToolCall;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class VirtualModelQualificationCliTest {
  private static final String REVISION = "a".repeat(40);

  @TempDir Path temporary;

  @Test
  void parsesOnePinnedFreshProcessArm() throws Exception {
    Path chat = Files.writeString(temporary.resolve("chat.gguf"), "chat");
    Path tools = Files.writeString(temporary.resolve("tools.gguf"), "tools");

    VirtualModelQualificationCli.Configuration configuration =
        VirtualModelQualificationCli.parse(
            new String[] {
              "--arm",
              "hybrid",
              "--chat-model",
              chat.toString(),
              "--tool-model",
              tools.toString(),
              "--models-revision",
              REVISION,
              "--run-id",
              "hybrid-02",
              "--max-tokens",
              "96",
              "--report",
              temporary.resolve("report.json").toString()
            });

    assertThat(configuration.arm()).isEqualTo(VirtualModelQualificationCli.Arm.HYBRID);
    assertThat(configuration.chatModel()).isEqualTo(chat);
    assertThat(configuration.toolModel()).isEqualTo(tools);
    assertThat(configuration.modelsRevision()).isEqualTo(REVISION);
    assertThat(configuration.runId()).isEqualTo("hybrid-02");
    assertThat(configuration.maxTokens()).isEqualTo(96);
  }

  @Test
  void requiresAnExplicitArmRunIdentityAndImmutableRevision() throws Exception {
    Path model = Files.writeString(temporary.resolve("model.gguf"), "fixture");

    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                VirtualModelQualificationCli.parse(
                    new String[] {
                      "--chat-model",
                      model.toString(),
                      "--tool-model",
                      model.toString(),
                      "--models-revision",
                      REVISION,
                      "--run-id",
                      "run-1"
                    }))
        .withMessageContaining("--arm");
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                VirtualModelQualificationCli.parse(
                    new String[] {
                      "--arm",
                      "control",
                      "--chat-model",
                      model.toString(),
                      "--tool-model",
                      model.toString(),
                      "--models-revision",
                      "main",
                      "--run-id",
                      "run-1"
                    }))
        .withMessageContaining("40-character Git SHA");
  }

  @Test
  void assessesProseAndExactToolArgumentsWithoutSubstringShortcuts() {
    assertThat(
            VirtualModelQualificationCli.assessProse(
                    "The memory record is mem-2048.", List.of("mem-2048"), List.of())
                .passed())
        .isTrue();
    assertThat(
            VirtualModelQualificationCli.assessProse(
                    "The memory record is unknown.", List.of("mem-2048"), List.of())
                .passed())
        .isFalse();
    assertThat(
            VirtualModelQualificationCli.assessProse(
                    "{\"name\":\"ready\",\"arguments\":{\"status\":\"READY\"}}",
                    List.of("ready"),
                    List.of())
                .passed())
        .isFalse();

    var correct = ToolCall.of(0, "remember", "{\"fact\":\"I prefer aisle seats.\"}");
    var wrong = ToolCall.of(0, "remember", "{\"fact\":\"I prefer window seats.\"}");
    assertThat(
            VirtualModelQualificationCli.assessTool(
                    List.of(correct), "remember", "{\"fact\":\"I prefer aisle seats.\"}")
                .passed())
        .isTrue();
    assertThat(
            VirtualModelQualificationCli.assessTool(
                    List.of(wrong), "remember", "{\"fact\":\"I prefer aisle seats.\"}")
                .passed())
        .isFalse();
  }
}
