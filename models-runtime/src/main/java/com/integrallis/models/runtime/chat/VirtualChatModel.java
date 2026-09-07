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
package com.integrallis.models.runtime.chat;

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.GenerationMetrics;
import com.integrallis.models.runtime.PromptCacheMetrics;
import com.integrallis.models.runtime.TextGenerationSession;
import com.integrallis.models.runtime.TokenConstraint;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.function.Supplier;

/**
 * Presents several independently stateful generation models as one semantic chat model.
 *
 * <p>A virtual session owns one canonical role-aware conversation and one physical generation
 * session per member. Conversation messages are shared, then rendered through the selected member's
 * own chat template. KV state is never copied between members: each member reuses only the exact
 * prompt prefix retained by its own {@link TextGenerationSession}.
 */
public final class VirtualChatModel {
  private final Map<String, Member> members;
  private final Selector selector;
  private final Executor backgroundPrefillExecutor;

  private VirtualChatModel(Builder builder) {
    if (builder.members.isEmpty()) {
      throw new IllegalStateException("at least one virtual model member is required");
    }
    this.members = Collections.unmodifiableMap(new LinkedHashMap<>(builder.members));
    this.selector = Objects.requireNonNull(builder.selector, "selector");
    this.backgroundPrefillExecutor = builder.backgroundPrefillExecutor;
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Opens independent KV state for every member behind one semantic conversation. */
  public Session openSession() {
    return openSession(UUID.randomUUID().toString());
  }

  /** Opens a named session so router telemetry can retain a stable conversation identity. */
  public Session openSession(String sessionId) {
    String id = requireText(sessionId, "sessionId");
    Map<String, ActiveMember> active = new LinkedHashMap<>();
    try {
      for (Member member : members.values()) {
        TextGenerationSession session =
            Objects.requireNonNull(member.sessionFactory().get(), "generation session");
        active.put(member.id(), new ActiveMember(member, session));
      }
      return new Session(id, active, selector, backgroundPrefillExecutor);
    } catch (RuntimeException | Error failure) {
      try {
        closeMembers(active.values());
      } catch (RuntimeException closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  /** Selects one physical member for a semantic turn. */
  @FunctionalInterface
  public interface Selector {
    Selection select(Turn turn);

    /** Records a completed routed turn; selectors without adaptive state may ignore it. */
    default void recordSuccess(Turn turn, Response response) {}

    /** Records a failed routed turn; selectors without adaptive state may ignore it. */
    default void recordFailure(Turn turn, Selection selection, Throwable failure) {}

    /** Releases any selector-owned affinity when a virtual conversation closes. */
    default void closeSession(String sessionId) {}

    /** Selects the prior capable member, or the first member declaring the requested task. */
    static Selector byCapability() {
      return turn -> {
        MemberState previous = turn.members().get(turn.previousMemberId());
        if (previous != null && previous.capabilities().contains(turn.taskType())) {
          return new Selection(previous.id(), "retained capable member", turn.taskType());
        }
        return turn.members().values().stream()
            .filter(member -> member.capabilities().contains(turn.taskType()))
            .findFirst()
            .map(
                member ->
                    new Selection(member.id(), "capability=" + turn.taskType(), turn.taskType()))
            .orElseThrow(
                () ->
                    new IllegalArgumentException(
                        "no virtual model member declares capability " + turn.taskType()));
      };
    }
  }

  /** Creates a fresh member-specific decoding constraint for one selected turn. */
  @FunctionalInterface
  public interface ConstraintFactory {
    Optional<TokenConstraint> create(TextGenerationSession session, Turn turn);

    static ConstraintFactory none() {
      return (session, turn) -> Optional.empty();
    }
  }

  /** Selected member and a human-readable reason retained in response telemetry. */
  public record Selection(String memberId, String reason, String taskType) {
    public Selection {
      memberId = requireText(memberId, "memberId");
      reason = requireText(reason, "reason");
      taskType = taskType == null ? "" : taskType;
    }

    public Selection(String memberId, String reason) {
      this(memberId, reason, "");
    }
  }

  /** Immutable routing input, including model-specific cache affinity. */
  public record Turn(
      String sessionId,
      String taskType,
      ChatMessage input,
      List<ChatMessage> history,
      List<ToolSpec> tools,
      String previousMemberId,
      Map<String, MemberState> members) {
    public Turn {
      sessionId = requireText(sessionId, "sessionId");
      taskType = taskType == null ? "" : taskType;
      input = Objects.requireNonNull(input, "input");
      history = List.copyOf(Objects.requireNonNull(history, "history"));
      tools = List.copyOf(Objects.requireNonNull(tools, "tools"));
      previousMemberId = previousMemberId == null ? "" : previousMemberId;
      members =
          Collections.unmodifiableMap(
              new LinkedHashMap<>(Objects.requireNonNull(members, "members")));
    }
  }

  /** The latest cache evidence for one physical member. */
  public record MemberState(
      String id,
      Set<String> capabilities,
      boolean invoked,
      boolean prefilled,
      int cachedPromptTokens,
      PromptCacheMetrics lastPromptCache,
      String prefillFailure) {
    public MemberState {
      id = requireText(id, "id");
      capabilities = Set.copyOf(Objects.requireNonNull(capabilities, "capabilities"));
      if (cachedPromptTokens < 0) {
        throw new IllegalArgumentException("cachedPromptTokens must not be negative");
      }
      lastPromptCache = Objects.requireNonNull(lastPromptCache, "lastPromptCache");
      prefillFailure = prefillFailure == null ? "" : prefillFailure;
    }
  }

  /** Where the selected member sits relative to the prior successful turn. */
  public enum Boundary {
    COLD,
    SAME_MODEL,
    FIRST_SWITCH,
    SWITCH_BACK
  }

  /** One generated turn plus the physical route and cache measurements that produced it. */
  public record Response(
      String memberId,
      String routeReason,
      String taskType,
      Boundary boundary,
      String output,
      String content,
      List<ToolCall> toolCalls,
      GenerationMetrics metrics) {
    public Response {
      memberId = requireText(memberId, "memberId");
      routeReason = requireText(routeReason, "routeReason");
      taskType = taskType == null ? "" : taskType;
      boundary = Objects.requireNonNull(boundary, "boundary");
      output = Objects.requireNonNull(output, "output");
      content = Objects.requireNonNull(content, "content");
      toolCalls = List.copyOf(Objects.requireNonNull(toolCalls, "toolCalls"));
      metrics = Objects.requireNonNull(metrics, "metrics");
    }

    public Response(
        String memberId,
        String routeReason,
        Boundary boundary,
        String output,
        String content,
        List<ToolCall> toolCalls,
        GenerationMetrics metrics) {
      this(memberId, routeReason, "", boundary, output, content, toolCalls, metrics);
    }

    public boolean hasToolCalls() {
      return !toolCalls.isEmpty();
    }
  }

  /** One conversation and its isolated physical model sessions. */
  public static final class Session implements AutoCloseable {
    private final Map<String, ActiveMember> members;
    private final Selector selector;
    private final String sessionId;
    private final Executor backgroundPrefillExecutor;
    private final Map<String, CompletableFuture<Void>> pendingPrefills = new LinkedHashMap<>();
    private final List<ChatMessage> history = new ArrayList<>();
    private final Set<String> seenMembers = new LinkedHashSet<>();
    private String previousMemberId = "";
    private boolean closed;

    private Session(
        String sessionId,
        Map<String, ActiveMember> members,
        Selector selector,
        Executor backgroundPrefillExecutor) {
      this.sessionId = sessionId;
      this.members = members;
      this.selector = selector;
      this.backgroundPrefillExecutor = backgroundPrefillExecutor;
    }

    /**
     * Adds one semantic input and generates the next assistant turn with the selected member.
     *
     * <p>The input remains in the canonical history if generation fails, so a caller may retry or
     * inspect the failed conversation without silently losing user or tool data.
     */
    public synchronized Response generate(
        String taskType, ChatMessage input, List<ToolSpec> tools, SamplingOptions options) {
      requireOpen();
      Objects.requireNonNull(input, "input");
      List<ToolSpec> declaredTools = tools == null ? List.of() : List.copyOf(tools);
      Objects.requireNonNull(options, "options");
      history.add(input);
      Turn turn =
          new Turn(
              sessionId,
              taskType,
              input,
              List.copyOf(history),
              declaredTools,
              previousMemberId,
              memberStates());
      Selection selection = Objects.requireNonNull(selector.select(turn), "selection");
      ActiveMember selected = members.get(selection.memberId());
      if (selected == null) {
        throw new IllegalArgumentException("selector chose unknown member " + selection.memberId());
      }

      Response response;
      try {
        awaitPrefill(selection.memberId());
        scheduleInactivePrefills(selection.memberId(), turn.history(), declaredTools);

        var prompt =
            declaredTools.isEmpty()
                ? selected.member.template().render(turn.history())
                : selected.member.template().render(turn.history(), declaredTools);
        Optional<TokenConstraint> constraint =
            Objects.requireNonNull(
                selected.member.constraintFactory().create(selected.session, turn),
                "token constraint");
        String output =
            constraint.isPresent()
                ? selected.session.generate(prompt, options, constraint.orElseThrow())
                : selected.session.generate(prompt, options);
        ToolCallScanner.Result scan =
            ToolCallScanner.scan(output, selected.member.template().toolSyntax(), declaredTools);
        if (scan.content().isBlank() && scan.toolCalls().isEmpty()) {
          throw new IllegalStateException(
              "member " + selected.member.id() + " produced no response");
        }
        if (scan.toolCalls().isEmpty()) {
          history.add(ChatMessage.assistant(scan.content()));
        } else {
          history.add(ChatMessage.assistantToolCalls(scan.content(), scan.toolCalls()));
        }

        Boundary boundary = boundary(selection.memberId());
        seenMembers.add(selection.memberId());
        previousMemberId = selection.memberId();
        GenerationMetrics metrics = selected.session.lastGenerationMetrics();
        selected.lastMetrics = metrics;
        selected.lastPromptCache = metrics.promptCache();
        selected.prefilled = false;
        response =
            new Response(
                selection.memberId(),
                selection.reason(),
                selection.taskType().isBlank() ? turn.taskType() : selection.taskType(),
                boundary,
                output,
                scan.content(),
                scan.toolCalls(),
                metrics);
      } catch (RuntimeException | Error failure) {
        try {
          selector.recordFailure(turn, selection, failure);
        } catch (RuntimeException | Error feedbackFailure) {
          failure.addSuppressed(feedbackFailure);
        }
        throw failure;
      }
      selector.recordSuccess(turn, response);
      return response;
    }

    /** Generates a turn whose task is classified by the configured selector. */
    public synchronized Response generate(
        ChatMessage input, List<ToolSpec> tools, SamplingOptions options) {
      return generate("", input, tools, options);
    }

    /** Returns the canonical, model-independent conversation accumulated so far. */
    public synchronized List<ChatMessage> history() {
      requireOpen();
      return List.copyOf(history);
    }

    /** Returns current per-member cache evidence for routing and diagnostics. */
    public synchronized Map<String, MemberState> memberStates() {
      requireOpen();
      Map<String, MemberState> states = new LinkedHashMap<>();
      members.forEach(
          (id, active) -> {
            GenerationMetrics metrics = active.lastMetrics;
            PromptCacheMetrics cache = active.lastPromptCache;
            states.put(
                id,
                new MemberState(
                    id,
                    active.member.capabilities(),
                    metrics.available(),
                    active.prefilled,
                    cache.inputTokens(),
                    cache,
                    active.prefillFailure));
          });
      return Collections.unmodifiableMap(states);
    }

    @Override
    public synchronized void close() {
      if (!closed) {
        closed = true;
        RuntimeException failure = null;
        pendingPrefills.values().forEach(future -> future.handle((ignored, error) -> null).join());
        try {
          closeMembers(members.values());
        } catch (RuntimeException closeFailure) {
          failure = closeFailure;
        }
        try {
          selector.closeSession(sessionId);
        } catch (RuntimeException selectorFailure) {
          if (failure == null) {
            failure = selectorFailure;
          } else {
            failure.addSuppressed(selectorFailure);
          }
        }
        if (failure != null) {
          throw failure;
        }
      }
    }

    private void awaitPrefill(String memberId) {
      CompletableFuture<Void> pending = pendingPrefills.get(memberId);
      if (pending != null) {
        pending.handle((ignored, failure) -> null).join();
      }
    }

    private void scheduleInactivePrefills(
        String selectedMemberId, List<ChatMessage> promptHistory, List<ToolSpec> tools) {
      if (backgroundPrefillExecutor == null) {
        return;
      }
      members.forEach(
          (id, active) -> {
            if (id.equals(selectedMemberId)) {
              return;
            }
            CompletableFuture<Void> previous =
                pendingPrefills.getOrDefault(id, CompletableFuture.completedFuture(null));
            CompletableFuture<Void> next =
                previous
                    .handle((ignored, failure) -> null)
                    .thenRunAsync(
                        () -> {
                          try {
                            var prompt =
                                tools.isEmpty()
                                    ? active.member.template().render(promptHistory)
                                    : active.member.template().render(promptHistory, tools);
                            var prefill = active.session.prefillPrompt(prompt);
                            active.lastPromptCache = prefill.promptCache();
                            active.prefilled = true;
                            active.prefillFailure = "";
                          } catch (RuntimeException failure) {
                            active.prefillFailure =
                                failure.getClass().getSimpleName() + ": " + failure.getMessage();
                            active.prefilled = false;
                          }
                        },
                        backgroundPrefillExecutor);
            pendingPrefills.put(id, next);
          });
    }

    private Boundary boundary(String selected) {
      if (previousMemberId.isEmpty()) {
        return Boundary.COLD;
      }
      if (previousMemberId.equals(selected)) {
        return Boundary.SAME_MODEL;
      }
      return seenMembers.contains(selected) ? Boundary.SWITCH_BACK : Boundary.FIRST_SWITCH;
    }

    private void requireOpen() {
      if (closed) {
        throw new IllegalStateException("virtual chat session is closed");
      }
    }
  }

  public static final class Builder {
    private final Map<String, Member> members = new LinkedHashMap<>();
    private Selector selector = Selector.byCapability();
    private Executor backgroundPrefillExecutor;

    private Builder() {}

    public Builder member(
        String id,
        Set<String> capabilities,
        ChatTemplate template,
        Supplier<TextGenerationSession> sessionFactory) {
      return member(id, capabilities, template, sessionFactory, ConstraintFactory.none());
    }

    public Builder member(
        String id,
        Set<String> capabilities,
        ChatTemplate template,
        Supplier<TextGenerationSession> sessionFactory,
        ConstraintFactory constraintFactory) {
      String memberId = requireText(id, "id");
      Member member =
          new Member(
              memberId,
              Set.copyOf(Objects.requireNonNull(capabilities, "capabilities")),
              Objects.requireNonNull(template, "template"),
              Objects.requireNonNull(sessionFactory, "sessionFactory"),
              Objects.requireNonNull(constraintFactory, "constraintFactory"));
      if (members.putIfAbsent(memberId, member) != null) {
        throw new IllegalArgumentException("duplicate virtual model member " + memberId);
      }
      return this;
    }

    public Builder selector(Selector value) {
      this.selector = Objects.requireNonNull(value, "selector");
      return this;
    }

    /**
     * Prefills inactive members while the selected member generates.
     *
     * <p>The caller owns the executor and chooses the resource policy. Without this option, members
     * catch up only when selected.
     */
    public Builder backgroundPrefill(Executor executor) {
      this.backgroundPrefillExecutor = Objects.requireNonNull(executor, "executor");
      return this;
    }

    public VirtualChatModel build() {
      return new VirtualChatModel(this);
    }
  }

  private record Member(
      String id,
      Set<String> capabilities,
      ChatTemplate template,
      Supplier<TextGenerationSession> sessionFactory,
      ConstraintFactory constraintFactory) {}

  private static final class ActiveMember {
    private final Member member;
    private final TextGenerationSession session;
    private GenerationMetrics lastMetrics = GenerationMetrics.unavailable();
    private volatile PromptCacheMetrics lastPromptCache = new PromptCacheMetrics(true, 0, 0, 0);
    private volatile boolean prefilled;
    private volatile String prefillFailure = "";

    private ActiveMember(Member member, TextGenerationSession session) {
      this.member = member;
      this.session = session;
    }
  }

  private static void closeMembers(Iterable<ActiveMember> members) {
    RuntimeException failure = null;
    for (ActiveMember member : members) {
      try {
        member.session.close();
      } catch (RuntimeException closeFailure) {
        if (failure == null) {
          failure = closeFailure;
        } else {
          failure.addSuppressed(closeFailure);
        }
      }
    }
    if (failure != null) {
      throw failure;
    }
  }

  private static String requireText(String value, String name) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value;
  }
}
