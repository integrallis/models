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

import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.runtime.GenerationMetrics;
import com.integrallis.models.runtime.PromptCacheMetrics;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.VirtualChatModel;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class VirtualChatRouterTest {

  @Test
  void delegatesUnlabelledTurnsToTheRoutersTaskClassifier() {
    ModelRouter router =
        ModelRouter.builder()
            .candidates(List.of(candidate("chat", "chat"), candidate("tools", "tool-use")))
            .classifier(query -> query.contains("weather") ? "tool-use" : "chat")
            .build();

    VirtualChatModel.Selection selected =
        new VirtualChatRouter(router)
            .select(turn("auto-session", "", ChatMessage.user("look up the weather"), "", 0, 0));

    assertThat(selected.memberId()).isEqualTo("tools");
    assertThat(selected.taskType()).isEqualTo("tool-use");
  }

  @Test
  void routesVirtualTurnsWithModelSpecificCacheEvidenceAndReleasesAffinity() {
    ModelRouter router =
        ModelRouter.builder()
            .candidates(List.of(candidate("chat", "chat"), candidate("tools", "tool-use")))
            .policy(RoutingPolicy.BALANCED)
            .build();
    VirtualChatRouter selector = new VirtualChatRouter(router);

    VirtualChatModel.Selection chat =
        selector.select(turn("session-7", "chat", ChatMessage.user("hello"), "", 120, 0));
    VirtualChatModel.Selection tools =
        selector.select(
            turn(
                "session-7", "tool-use", ChatMessage.user("look up the weather"), "chat", 160, 80));

    assertThat(chat.memberId()).isEqualTo("chat");
    assertThat(tools.memberId()).isEqualTo("tools");
    assertThat(tools.reason()).contains("tool-use");
    assertThat(router.activeSessionCount()).isOne();

    VirtualChatModel.Turn toolTurn =
        turn("session-7", "tool-use", ChatMessage.user("look up the weather"), "tools", 160, 80);
    selector.recordSuccess(
        toolTurn,
        new VirtualChatModel.Response(
            "tools",
            "fixture",
            VirtualChatModel.Boundary.FIRST_SWITCH,
            "done",
            "done",
            List.of(),
            metrics()));

    assertThat(router.status("tools").orElseThrow().timeToFirstTokenMillis()).isGreaterThan(10);

    selector.recordFailure(toolTurn, tools, new IllegalStateException("fixture failure"));

    assertThat(router.status("tools").orElseThrow().consecutiveFailures()).isOne();

    selector.closeSession("session-7");

    assertThat(router.activeSessionCount()).isZero();
  }

  private static GenerationMetrics metrics() {
    return new GenerationMetrics(
        true,
        true,
        Duration.ofMillis(1),
        Duration.ofMillis(1),
        Duration.ofMillis(100),
        Optional.of(Duration.ofMillis(123)),
        Duration.ofMillis(50),
        Duration.ofMillis(175),
        new GenerationUsage(100, 10),
        new PromptCacheMetrics(true, 100, 80, 20));
  }

  private static VirtualChatModel.Turn turn(
      String sessionId,
      String task,
      ChatMessage input,
      String previous,
      int chatCache,
      int toolCache) {
    return new VirtualChatModel.Turn(
        sessionId,
        task,
        input,
        List.of(input),
        List.of(),
        previous,
        Map.of(
            "chat", state("chat", "chat", chatCache),
            "tools", state("tools", "tool-use", toolCache)));
  }

  private static VirtualChatModel.MemberState state(
      String id, String capability, int cachedTokens) {
    return new VirtualChatModel.MemberState(
        id,
        Set.of(capability),
        cachedTokens > 0,
        false,
        cachedTokens,
        new PromptCacheMetrics(true, cachedTokens, cachedTokens, 0),
        "");
  }

  private static ModelCandidate candidate(String id, String capability) {
    return ModelCandidate.builder(id)
        .local(true)
        .tags(Set.of(capability))
        .timeToFirstTokenMillis(10)
        .tokensPerSecond(10)
        .quality(Map.of(capability, 1.0))
        .build();
  }
}
