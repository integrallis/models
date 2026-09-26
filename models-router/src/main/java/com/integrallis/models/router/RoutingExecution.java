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

import java.math.BigDecimal;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.FutureTask;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

/**
 * One end-to-end invocation. Close on every terminal path. Cancellation stops admission and settles
 * active reservations conservatively; stopping an already running external provider is best effort.
 */
public final class RoutingExecution implements AutoCloseable {
  private static final ScheduledExecutorService TIMER =
      Executors.newSingleThreadScheduledExecutor(
          Thread.ofPlatform().daemon().name("models-routing-deadlines").factory());
  private final RoutingExecutionOptions options;
  private final RoutingBudget requestBudget;
  private final RoutingCancellationToken stop = new RoutingCancellationToken();
  private final LongSupplier clock;
  private final long started;
  private final ScheduledFuture<?> deadline;
  private final AutoCloseable subscription;
  private Attempt active;
  private boolean closed;
  private boolean cancelled;

  public RoutingExecution(RoutingExecutionOptions options) {
    this(options, System::nanoTime);
  }

  RoutingExecution(RoutingExecutionOptions options, LongSupplier clock) {
    this.options = Objects.requireNonNull(options, "options");
    this.clock = Objects.requireNonNull(clock, "clock");
    started = clock.getAsLong();
    requestBudget =
        options.requestBudget() == null ? null : new RoutingBudget(options.requestBudget());
    subscription =
        options.cancellation() == null ? () -> {} : options.cancellation().onCancel(this::cancel);
    deadline =
        options.timeout() == null
            ? null
            : TIMER.schedule(
                () -> Thread.ofVirtual().name("models-routing-timeout").start(this::cancel),
                options.timeout().toNanos(),
                TimeUnit.NANOSECONDS);
  }

  public RoutingExecutionOptions options() {
    return options;
  }

  public RoutingRequest account(RoutingRequest request) {
    checkActive();
    if (options.tokenBounds() == null) return request;
    return new RoutingRequest(
        request.query(),
        Math.max(request.estimatedTokens(), options.tokenBounds().contextTokens()),
        request.taskTypeOverride(),
        request.sessionId());
  }

  public void checkActive() {
    if (Thread.currentThread().isInterrupted()
        || (options.timeout() != null
            && clock.getAsLong() - started >= options.timeout().toNanos())) {
      cancel();
    }
    synchronized (this) {
      if (closed || cancelled || stop.isCancelled())
        throw new CancellationException("routed execution cancelled or deadline exceeded");
    }
  }

  /** Registers best-effort provider/subscription cancellation. */
  public AutoCloseable onCancel(Runnable action) {
    return stop.onCancel(action);
  }

  public void cancel() {
    synchronized (this) {
      cancelled = true;
      if (active != null) active.settle(null);
    }
    stop.cancel();
  }

  public Attempt beginAttempt(ModelCandidate candidate) {
    Objects.requireNonNull(candidate, "candidate");
    checkActive();
    synchronized (this) {
      if (closed || cancelled) throw new CancellationException("routed execution cancelled");
      if (active != null) throw new IllegalStateException("an attempt is already active");
      BigDecimal amount =
          options.tokenBounds() == null
              ? BigDecimal.ZERO
              : RoutingBudget.cost(
                  candidate,
                  options.tokenBounds().inputTokens(),
                  options.tokenBounds().outputTokens());
      RoutingBudget.Reservation first =
          requestBudget == null ? null : requestBudget.reserve(amount);
      RoutingBudget.Reservation second;
      try {
        second = options.sharedBudget() == null ? null : options.sharedBudget().reserve(amount);
      } catch (RuntimeException denied) {
        if (first != null) first.release();
        throw denied;
      }
      active = new Attempt(candidate, first, second);
      return active;
    }
  }

  /**
   * Runs cancellable blocking work on a virtual thread when a deadline/token is supplied. The
   * caller returns promptly on cancellation even if a provider ignores interruption. Such a
   * provider can still incur charges, which is why the active reservation remains charged.
   */
  @SuppressWarnings("try")
  public <R> R call(Callable<R> work) throws Exception {
    checkActive();
    if (options.timeout() == null && options.cancellation() == null) return work.call();
    FutureTask<R> task = new FutureTask<>(work);
    try (AutoCloseable registration = onCancel(() -> task.cancel(true))) {
      Thread.ofVirtual().name("models-routed-call").start(task);
      try {
        return task.get();
      } catch (InterruptedException interrupted) {
        cancel();
        Thread.currentThread().interrupt();
        throw new CancellationException("routed caller interrupted");
      } catch (ExecutionException failure) {
        if (failure.getCause() instanceof Exception exception) throw exception;
        if (failure.getCause() instanceof Error error) throw error;
        throw new IllegalStateException(failure.getCause());
      }
    }
  }

  @Override
  public void close() {
    synchronized (this) {
      if (closed) return;
      closed = true;
      if (active != null) active.settle(null);
    }
    if (deadline != null) deadline.cancel(false);
    try {
      subscription.close();
    } catch (Exception ignored) {
      /* removal only */
    }
  }

  public final class Attempt implements AutoCloseable {
    private final ModelCandidate candidate;
    private final RoutingBudget.Reservation first;
    private final RoutingBudget.Reservation second;
    private boolean settled;

    private Attempt(
        ModelCandidate candidate,
        RoutingBudget.Reservation first,
        RoutingBudget.Reservation second) {
      this.candidate = candidate;
      this.first = first;
      this.second = second;
    }

    /**
     * Settle known usage; null charges the bound. Duplicate/late terminal events have no effect.
     */
    public void settle(RoutingUsage usage) {
      synchronized (RoutingExecution.this) {
        if (settled) return;
        settled = true;
        active = null;
        BigDecimal cost =
            usage == null
                ? null
                : RoutingBudget.cost(candidate, usage.inputTokens(), usage.outputTokens());
        if (first != null) first.settle(cost);
        if (second != null) second.settle(cost);
        if (usage != null
            && options.tokenBounds() != null
            && (usage.inputTokens() > options.tokenBounds().inputTokens()
                || usage.outputTokens() > options.tokenBounds().outputTokens())) {
          cancelled = true;
          throw new RoutingBudgetExceededException(
              "provider usage exceeded declared token bounds; actual cost recorded");
        }
      }
    }

    @Override
    public void close() {
      settle(null);
    }
  }
}
