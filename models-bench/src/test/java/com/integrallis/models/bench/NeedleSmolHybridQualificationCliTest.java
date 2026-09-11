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

import com.integrallis.models.runtime.chat.ChatMessage;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NeedleSmolHybridQualificationCliTest {
  private static final String REVISION = "a".repeat(40);

  @TempDir Path temporary;

  @Test
  void parsesThreePinnedMembersForAFreshProcessArm() throws Exception {
    Path control = Files.writeString(temporary.resolve("control.gguf"), "control");
    Path chat = Files.writeString(temporary.resolve("chat.gguf"), "chat");
    Path tools = Files.writeString(temporary.resolve("tools.cact"), "tools");

    var configuration =
        NeedleSmolHybridQualificationCli.parse(
            new String[] {
              "--arm",
              "hybrid",
              "--control-model",
              control.toString(),
              "--chat-model",
              chat.toString(),
              "--chat-profile",
              "smollm2-360m",
              "--tool-model",
              tools.toString(),
              "--tool-profile",
              "needle2",
              "--models-revision",
              REVISION,
              "--run-id",
              "hybrid-01",
              "--max-tokens",
              "96",
              "--report",
              temporary.resolve("report.json").toString()
            });

    assertThat(configuration.arm()).isEqualTo(NeedleSmolHybridQualificationCli.Arm.HYBRID);
    assertThat(configuration.controlModel()).isEqualTo(control);
    assertThat(configuration.chatModel()).isEqualTo(chat);
    assertThat(configuration.chatProfile())
        .isEqualTo(NeedleSmolHybridQualificationCli.ChatProfile.SMOLLM2_360M);
    assertThat(configuration.toolModel()).isEqualTo(tools);
    assertThat(configuration.toolProfile())
        .isEqualTo(NeedleSmolHybridQualificationCli.ToolProfile.NEEDLE2);
    assertThat(configuration.modelsRevision()).isEqualTo(REVISION);
    assertThat(configuration.runId()).isEqualTo("hybrid-01");
    assertThat(configuration.maxTokens()).isEqualTo(96);
  }

  @Test
  void requiresTheCompletePinnedTopology() throws Exception {
    Path model = Files.writeString(temporary.resolve("model.gguf"), "fixture");

    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                NeedleSmolHybridQualificationCli.parse(
                    new String[] {
                      "--arm",
                      "hybrid",
                      "--control-model",
                      model.toString(),
                      "--chat-model",
                      model.toString(),
                      "--chat-profile",
                      "smollm2-360m",
                      "--tool-profile",
                      "needle2",
                      "--models-revision",
                      REVISION,
                      "--run-id",
                      "run-1"
                    }))
        .withMessageContaining("--tool-model");
  }

  @Test
  void assignsOnlySelectionTurnsToNeedle() {
    assertThat(
            NeedleSmolHybridQualificationCli.expectedMember(
                NeedleSmolHybridQualificationCli.Arm.HYBRID,
                "tool-use",
                ChatMessage.user("Call remember")))
        .isEqualTo("needle-2");
    assertThat(
            NeedleSmolHybridQualificationCli.expectedMember(
                NeedleSmolHybridQualificationCli.Arm.HYBRID,
                "chat",
                ChatMessage.tool("remember", "{\"stored\":true}")))
        .isEqualTo("smollm2-360m");
    assertThat(
            NeedleSmolHybridQualificationCli.expectedMember(
                NeedleSmolHybridQualificationCli.Arm.CONTROL,
                "tool-use",
                ChatMessage.user("Call remember")))
        .isEqualTo("qwen-1.7b");
  }

  @Test
  void supportsTheQualifiedQwenToolMember() {
    assertThat(NeedleSmolHybridQualificationCli.ToolProfile.parse("qwen3-1.7b").memberId())
        .isEqualTo("qwen-1.7b-tools");
    assertThat(
            NeedleSmolHybridQualificationCli.expectedMember(
                NeedleSmolHybridQualificationCli.Arm.HYBRID,
                NeedleSmolHybridQualificationCli.ToolProfile.QWEN3_1_7B,
                "tool-use",
                ChatMessage.user("Call remember")))
        .isEqualTo("qwen-1.7b-tools");
  }

  @Test
  void supportsTheQualifiedQwenChatMember() {
    assertThat(NeedleSmolHybridQualificationCli.ChatProfile.parse("qwen3-0.6b").memberId())
        .isEqualTo("qwen-0.6b-chat");
    assertThat(
            NeedleSmolHybridQualificationCli.expectedMember(
                NeedleSmolHybridQualificationCli.Arm.HYBRID,
                NeedleSmolHybridQualificationCli.ChatProfile.QWEN3_0_6B,
                NeedleSmolHybridQualificationCli.ToolProfile.QWEN3_1_7B,
                "chat",
                ChatMessage.user("hello")))
        .isEqualTo("qwen-0.6b-chat");
  }

  @Test
  void keepsSchemasVisibleToTheControlButOnlyToolTurnsInTheHybrid() {
    assertThat(
            NeedleSmolHybridQualificationCli.declaresTools(
                NeedleSmolHybridQualificationCli.Arm.CONTROL, false))
        .isTrue();
    assertThat(
            NeedleSmolHybridQualificationCli.declaresTools(
                NeedleSmolHybridQualificationCli.Arm.HYBRID, true))
        .isTrue();
    assertThat(
            NeedleSmolHybridQualificationCli.declaresTools(
                NeedleSmolHybridQualificationCli.Arm.HYBRID, false))
        .isFalse();
  }
}
