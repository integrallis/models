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
package com.integrallis.models.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.runtime.GenerationMetrics;
import com.integrallis.models.runtime.SharedToolTurn;
import com.integrallis.models.runtime.TokenConstraint;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class ActivatedTurnRegistryTest {

  @Test
  void rejectsAnEmptyCallSetInsteadOfRetainingAnUnreachableBranch() {
    ActivatedTurnRegistry registry = new ActivatedTurnRegistry();
    RecordingTurn turn = new RecordingTurn();

    assertThatThrownBy(() -> registry.retain(turn, List.of()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("at least one tool call");
    assertThat(turn.closeCount).isZero();
  }

  @Test
  void requiresEveryResultFromTheSameTurnBeforeTransferringOwnership() {
    AtomicLong clock = new AtomicLong();
    ActivatedTurnRegistry registry = new ActivatedTurnRegistry(4, 100, clock::get);
    RecordingTurn turn = new RecordingTurn();
    List<ToolExecutionRequest> requests = registry.retain(turn, List.of(call("one"), call("two")));

    assertThatThrownBy(() -> registry.take(results(requests.getFirst().id())))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("incomplete");
    assertThat(turn.closeCount).isZero();

    ActivatedTurnRegistry.PendingTurn pending =
        registry.take(results(requests.getFirst().id(), requests.getLast().id()));

    assertThat(pending.turn).isSameAs(turn);
    assertThat(turn.closeCount).isZero();
    pending.turn.close();
    assertThat(turn.closeCount).isOne();
  }

  @Test
  void rejectsResultsThatCrossRequestScopedTurns() {
    ActivatedTurnRegistry registry = new ActivatedTurnRegistry();
    RecordingTurn first = new RecordingTurn();
    RecordingTurn second = new RecordingTurn();
    List<ToolExecutionRequest> firstRequests = registry.retain(first, List.of(call("first")));
    List<ToolExecutionRequest> secondRequests = registry.retain(second, List.of(call("second")));

    assertThatThrownBy(
            () ->
                registry.take(
                    results(firstRequests.getFirst().id(), secondRequests.getFirst().id())))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("different activated turns");

    registry.close();
    assertThat(first.closeCount).isOne();
    assertThat(second.closeCount).isOne();
  }

  @Test
  void closesExpiredTurnsAndNeverReturnsTheirBranches() {
    AtomicLong clock = new AtomicLong();
    ActivatedTurnRegistry registry = new ActivatedTurnRegistry(4, 10, clock::get);
    RecordingTurn expired = new RecordingTurn();
    List<ToolExecutionRequest> requests = registry.retain(expired, List.of(call("expired")));

    clock.set(10);

    assertThatThrownBy(() -> registry.take(results(requests.getFirst().id())))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("no live activated tool turn");
    assertThat(expired.closeCount).isOne();
  }

  @Test
  void evictsAndClosesTheOldestTurnAtTheConfiguredBound() {
    AtomicLong clock = new AtomicLong();
    ActivatedTurnRegistry registry = new ActivatedTurnRegistry(2, 100, clock::get);
    RecordingTurn first = new RecordingTurn();
    RecordingTurn second = new RecordingTurn();
    RecordingTurn third = new RecordingTurn();
    registry.retain(first, List.of(call("first")));
    clock.incrementAndGet();
    registry.retain(second, List.of(call("second")));
    clock.incrementAndGet();

    registry.retain(third, List.of(call("third")));

    assertThat(first.closeCount).isOne();
    assertThat(second.closeCount).isZero();
    assertThat(third.closeCount).isZero();
    registry.close();
    assertThat(second.closeCount).isOne();
    assertThat(third.closeCount).isOne();
  }

  @Test
  void ignoresCompletedResultsFromEarlierConversationTurns() {
    ActivatedTurnRegistry registry = new ActivatedTurnRegistry();

    ChatRequest laterUserTurn =
        ChatRequest.builder()
            .messages(
                List.of(
                    ToolExecutionResultMessage.from("completed", "tool", "{}"),
                    UserMessage.from("What should I do next?")))
            .build();

    assertThat(registry.take(laterUserTurn)).isNull();
  }

  @Test
  void assignsDistinctCorrelationIdsWhenConcurrentModelOutputsReuseSyntheticIds() {
    ActivatedTurnRegistry registry = new ActivatedTurnRegistry();
    RecordingTurn first = new RecordingTurn();
    RecordingTurn second = new RecordingTurn();

    List<ToolExecutionRequest> firstRequests = registry.retain(first, List.of(call("000000000")));
    List<ToolExecutionRequest> secondRequests = registry.retain(second, List.of(call("000000000")));

    assertThat(firstRequests.getFirst().id()).isNotEqualTo(secondRequests.getFirst().id());
    assertThat(registry.take(results(firstRequests.getFirst().id())).turn).isSameAs(first);
    assertThat(registry.take(results(secondRequests.getFirst().id())).turn).isSameAs(second);
  }

  private static ToolExecutionRequest call(String id) {
    return ToolExecutionRequest.builder().id(id).name("tool").arguments("{}").build();
  }

  private static ChatRequest results(String... ids) {
    return ChatRequest.builder()
        .messages(
            java.util.Arrays.stream(ids)
                .<ChatMessage>map(id -> ToolExecutionResultMessage.from(id, "tool", "{}"))
                .toList())
        .build();
  }

  private static final class RecordingTurn implements SharedToolTurn {
    private int closeCount;

    @Override
    public String generateToolCall(SamplingOptions options, TokenConstraint constraint) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void generateToolCall(
        SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
      throw new UnsupportedOperationException();
    }

    @Override
    public String generateBaseResponse(ModelPrompt prompt, SamplingOptions options) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void generateBaseResponse(
        ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
      throw new UnsupportedOperationException();
    }

    @Override
    public int sharedPrefixTokens() {
      return 0;
    }

    @Override
    public long sharedPrefixBytes() {
      return 0;
    }

    @Override
    public boolean physicallySharesPrefix() {
      return true;
    }

    @Override
    public GenerationMetrics toolMetrics() {
      return GenerationMetrics.unavailable();
    }

    @Override
    public GenerationMetrics responseMetrics() {
      return GenerationMetrics.unavailable();
    }

    @Override
    public void close() {
      closeCount++;
    }
  }
}
