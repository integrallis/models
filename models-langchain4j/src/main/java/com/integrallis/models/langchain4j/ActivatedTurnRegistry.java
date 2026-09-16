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

import com.integrallis.models.runtime.SharedToolTurn;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import java.time.Duration;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.LongSupplier;

/** Bounds request-scoped branches across LangChain4j's separate call/result callbacks. */
final class ActivatedTurnRegistry implements AutoCloseable {
  private static final int MAX_PENDING_TURNS = 64;
  private static final long TURN_TTL_NANOS = Duration.ofMinutes(5).toNanos();

  private final int maxPendingTurns;
  private final long turnTtlNanos;
  private final LongSupplier nanoTime;
  private final LinkedHashMap<String, PendingTurn> byCallId = new LinkedHashMap<>();
  private final LinkedHashSet<PendingTurn> turns = new LinkedHashSet<>();
  private long nextTurnId;

  ActivatedTurnRegistry() {
    this(MAX_PENDING_TURNS, TURN_TTL_NANOS, System::nanoTime);
  }

  ActivatedTurnRegistry(int maxPendingTurns, long turnTtlNanos, LongSupplier nanoTime) {
    if (maxPendingTurns <= 0) {
      throw new IllegalArgumentException("maxPendingTurns must be > 0");
    }
    if (turnTtlNanos <= 0) {
      throw new IllegalArgumentException("turnTtlNanos must be > 0");
    }
    this.maxPendingTurns = maxPendingTurns;
    this.turnTtlNanos = turnTtlNanos;
    this.nanoTime = Objects.requireNonNull(nanoTime, "nanoTime");
  }

  synchronized List<ToolExecutionRequest> retain(
      SharedToolTurn turn, List<ToolExecutionRequest> requests) {
    Objects.requireNonNull(turn, "turn");
    Objects.requireNonNull(requests, "requests");
    if (requests.isEmpty()) {
      throw new IllegalArgumentException("activated turn must contain at least one tool call");
    }
    long turnId = nextTurnId++;
    List<ToolExecutionRequest> correlated = correlate(turnId, requests);
    Set<String> callIds = new HashSet<>();
    for (ToolExecutionRequest request : correlated) {
      if (request.id() == null || request.id().isBlank() || !callIds.add(request.id())) {
        throw new IllegalStateException("activated tool calls require distinct nonblank IDs");
      }
    }
    long now = nanoTime.getAsLong();
    removeExpired(now);
    while (turns.size() >= maxPendingTurns) {
      closePending(turns.getFirst());
    }
    for (String callId : callIds) {
      if (byCallId.containsKey(callId)) {
        throw new IllegalStateException("duplicate pending tool-call ID: " + callId);
      }
    }
    PendingTurn pending = new PendingTurn(turn, Set.copyOf(callIds), now);
    turns.add(pending);
    for (String callId : callIds) {
      byCallId.put(callId, pending);
    }
    return correlated;
  }

  synchronized PendingTurn take(ChatRequest request) {
    Objects.requireNonNull(request, "request");
    Set<String> resultIds = new HashSet<>();
    List<dev.langchain4j.data.message.ChatMessage> messages = request.messages();
    for (int index = messages.size() - 1; index >= 0; index--) {
      if (!(messages.get(index) instanceof ToolExecutionResultMessage result)) break;
      resultIds.add(result.id());
    }
    removeExpired(nanoTime.getAsLong());
    if (resultIds.isEmpty()) {
      return null;
    }

    PendingTurn matched = null;
    for (String resultId : resultIds) {
      PendingTurn candidate = byCallId.get(resultId);
      if (candidate == null) {
        throw new IllegalStateException(
            "no live activated tool turn matches result ID " + resultId);
      }
      if (matched != null && matched != candidate) {
        throw new IllegalStateException("tool results combine different activated turns");
      }
      matched = candidate;
    }
    if (!resultIds.containsAll(matched.callIds)) {
      throw new IllegalStateException(
          "tool results are incomplete for activated turn; expected " + matched.callIds);
    }
    remove(matched);
    return matched;
  }

  private static List<ToolExecutionRequest> correlate(
      long turnId, List<ToolExecutionRequest> requests) {
    java.util.ArrayList<ToolExecutionRequest> correlated =
        new java.util.ArrayList<>(requests.size());
    String turn = Long.toUnsignedString(turnId, Character.MAX_RADIX);
    for (int index = 0; index < requests.size(); index++) {
      ToolExecutionRequest request = Objects.requireNonNull(requests.get(index), "request");
      correlated.add(
          ToolExecutionRequest.builder()
              .id("alora-" + turn + "-" + Integer.toString(index, Character.MAX_RADIX))
              .name(request.name())
              .arguments(request.arguments())
              .build());
    }
    return List.copyOf(correlated);
  }

  @Override
  public synchronized void close() {
    for (PendingTurn pending : List.copyOf(turns)) {
      closePending(pending);
    }
  }

  private void removeExpired(long now) {
    for (PendingTurn pending : List.copyOf(turns)) {
      if (now - pending.createdNanos >= turnTtlNanos) {
        closePending(pending);
      }
    }
  }

  private void closePending(PendingTurn pending) {
    remove(pending);
    pending.turn.close();
  }

  private void remove(PendingTurn pending) {
    turns.remove(pending);
    for (String callId : pending.callIds) {
      byCallId.remove(callId, pending);
    }
  }

  static final class PendingTurn {
    final SharedToolTurn turn;
    private final Set<String> callIds;
    private final long createdNanos;

    private PendingTurn(SharedToolTurn turn, Set<String> callIds, long createdNanos) {
      this.turn = turn;
      this.callIds = callIds;
      this.createdNanos = createdNanos;
    }
  }
}
