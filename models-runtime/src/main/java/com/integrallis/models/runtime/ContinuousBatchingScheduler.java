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
package com.integrallis.models.runtime;

import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

/** One model-owned worker that continuously batches independent generation-session steps. */
final class ContinuousBatchingScheduler implements AutoCloseable {
  private final BatchInferenceBackend backend;
  private final Object executionLock;
  private final int maximumBatchSize;
  private final int maximumPrefillChunkTokens;
  private final long batchFormationDelayNanos;
  private final ArrayBlockingQueue<Request> waiting;
  private final Thread worker;
  private final AtomicLong completedRequests = new AtomicLong();
  private final AtomicLong failedRequests = new AtomicLong();
  private final AtomicLong rejectedRequests = new AtomicLong();
  private final AtomicLong batchInvocations = new AtomicLong();
  private final AtomicLong sessionSteps = new AtomicLong();
  private final AtomicInteger largestBatch = new AtomicInteger();
  private final AtomicInteger activeRequests = new AtomicInteger();
  private int prefillCursor;
  private volatile boolean closed;

  ContinuousBatchingScheduler(
      BatchInferenceBackend backend, Object executionLock, ContinuousBatchingOptions options) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.executionLock = Objects.requireNonNull(executionLock, "executionLock");
    Objects.requireNonNull(options, "options");
    int backendMaximum = backend.maxBatchSize();
    if (backendMaximum <= 0) {
      throw new IllegalArgumentException("backend maxBatchSize must be > 0");
    }
    maximumBatchSize = options.maximumBatchSize();
    if (maximumBatchSize > backendMaximum) {
      throw new IllegalArgumentException(
          "maximumBatchSize " + maximumBatchSize + " exceeds backend capacity " + backendMaximum);
    }
    maximumPrefillChunkTokens = options.maximumPrefillChunkTokens();
    batchFormationDelayNanos = options.batchFormationDelay().toNanos();
    waiting = new ArrayBlockingQueue<>(options.maximumQueuedRequests());
    worker =
        Thread.ofPlatform()
            .daemon(true)
            .name("models-continuous-batching-" + backend.metadata().modelName())
            .start(this::run);
  }

  SessionState session(InferenceSession session) {
    return new SessionState(Objects.requireNonNull(session, "session"));
  }

  GenerationMetrics generate(
      SessionState state,
      ModelPrompt prompt,
      SamplingOptions options,
      TokenStream stream,
      TokenConstraint constraint) {
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(stream, "stream");
    Objects.requireNonNull(constraint, "constraint");
    long requestStarted = System.nanoTime();
    TokenizedPrompt tokenized = tokenize(prompt);
    Request request =
        Request.generation(state, tokenized, options, stream, constraint, requestStarted);
    if (!submit(request)) {
      failedRequests.incrementAndGet();
      rejectedRequests.incrementAndGet();
      IllegalStateException failure =
          new IllegalStateException(
              closed
                  ? "continuous batching scheduler is closed"
                  : "continuous batching queue is full");
      state.lastGenerationMetrics = GenerationMetrics.unavailable();
      signalError(stream, failure);
      return GenerationMetrics.unavailable();
    }
    return request.generationResult.join();
  }

  PromptPrefillMetrics prefill(SessionState state, ModelPrompt prompt) {
    long requestStarted = System.nanoTime();
    TokenizedPrompt tokenized = tokenize(prompt);
    Request request = Request.prefill(state, tokenized, requestStarted);
    if (!submit(request)) {
      failedRequests.incrementAndGet();
      rejectedRequests.incrementAndGet();
      throw new IllegalStateException(
          closed ? "continuous batching scheduler is closed" : "continuous batching queue is full");
    }
    try {
      return request.prefillResult.join();
    } catch (CompletionException failure) {
      if (failure.getCause() instanceof RuntimeException runtimeFailure) {
        throw runtimeFailure;
      }
      if (failure.getCause() instanceof Error error) {
        throw error;
      }
      throw failure;
    }
  }

  ContinuousBatchingMetrics metrics() {
    return new ContinuousBatchingMetrics(
        completedRequests.get(),
        failedRequests.get(),
        rejectedRequests.get(),
        batchInvocations.get(),
        sessionSteps.get(),
        largestBatch.get(),
        activeRequests.get(),
        waiting.size());
  }

  @Override
  public void close() {
    if (!closed) {
      closed = true;
      worker.interrupt();
      try {
        worker.join();
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException(
            "interrupted while stopping continuous batching", interrupted);
      }
    }
  }

  private TokenizedPrompt tokenize(ModelPrompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    if (prompt.isEmpty()) {
      throw new IllegalArgumentException("prompt must not be empty");
    }
    long started = System.nanoTime();
    int[] tokens = backend.tokenizer().encode(prompt);
    Duration elapsed = Duration.ofNanos(elapsed(started, System.nanoTime()));
    if (tokens.length == 0) {
      throw new IllegalArgumentException("prompt produced no tokens");
    }
    return new TokenizedPrompt(tokens, elapsed);
  }

  private boolean submit(Request request) {
    return !closed && waiting.offer(request);
  }

  private void run() {
    List<Request> active = new ArrayList<>(maximumBatchSize);
    try {
      while (!closed) {
        if (active.isEmpty()) {
          active.add(waiting.take());
          activeRequests.set(active.size());
          if (batchFormationDelayNanos > 0) {
            LockSupport.parkNanos(batchFormationDelayNanos);
            if (Thread.interrupted() && closed) {
              break;
            }
          }
        }
        admit(active);
        prepare(active);
        removeFailed(active);
        advancePrefill(active);
        removeFailed(active);
        if (!active.isEmpty()) {
          advance(active);
          active.removeIf(Request::isDone);
          activeRequests.set(active.size());
        }
      }
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
    } finally {
      IllegalStateException failure =
          new IllegalStateException("continuous batching scheduler closed");
      active.forEach(request -> fail(request, failure));
      activeRequests.set(0);
      Request request;
      while ((request = waiting.poll()) != null) {
        fail(request, failure);
      }
    }
  }

  private void admit(List<Request> active) {
    while (active.size() < maximumBatchSize) {
      Request request = waiting.poll();
      if (request == null) {
        return;
      }
      active.add(request);
      activeRequests.set(active.size());
    }
  }

  private void prepare(List<Request> active) {
    for (Request request : active) {
      if (request.prepared || request.isDone()) {
        continue;
      }
      try {
        long started = System.nanoTime();
        if (request.promptTokens.length > backend.contextCapacity()) {
          throw new IllegalArgumentException(
              "prompt token count "
                  + request.promptTokens.length
                  + " exceeds context capacity "
                  + backend.contextCapacity());
        }
        synchronized (executionLock) {
          int reusable =
              reusablePrefixLength(request.state.cachedPromptTokens, request.promptTokens);
          if (reusable == 0 || request.state.session.checkpoint() < reusable) {
            backend.reset(request.state.session);
            reusable = 0;
          } else {
            backend.rewind(request.state.session, reusable);
          }
          request.promptCache =
              new PromptCacheMetrics(
                  true,
                  request.promptTokens.length,
                  reusable,
                  request.promptTokens.length - reusable);
          request.promptIndex = reusable;
        }
        request.promptPreparation = Duration.ofNanos(elapsed(started, System.nanoTime()));
        request.prepared = true;
      } catch (RuntimeException | Error failure) {
        fail(request, failure);
      }
    }
  }

  private void advancePrefill(List<Request> active) {
    if (active.isEmpty()) {
      return;
    }
    boolean decoding = active.stream().anyMatch(Request::readyToDecode);
    int chunks = decoding ? 1 : active.size();
    int scanned = 0;
    int index = Math.floorMod(prefillCursor, active.size());
    while (scanned < active.size() && chunks > 0) {
      Request request = active.get(index);
      if (!request.prepared || !request.inPrompt() || request.isDone()) {
        index = (index + 1) % active.size();
        scanned++;
        continue;
      }
      advancePrefill(request);
      chunks--;
      index = (index + 1) % active.size();
      scanned++;
    }
    prefillCursor = index;
  }

  private void advancePrefill(Request request) {
    try {
      int start = request.promptIndex;
      int end = Math.min(request.promptTokens.length, start + maximumPrefillChunkTokens);
      long started = System.nanoTime();
      float[] logits;
      synchronized (executionLock) {
        logits =
            backend.prefill(
                request.state.session, Arrays.copyOfRange(request.promptTokens, start, end), start);
      }
      request.prefill = request.prefill.plusNanos(elapsed(started, System.nanoTime()));
      request.promptIndex = end;
      if (!request.inPrompt()) {
        if (request.prefillOnly) {
          completePrefill(request);
        } else {
          acceptInitialLogits(request, logits);
        }
      }
    } catch (RuntimeException | Error failure) {
      fail(request, failure);
    }
  }

  private void removeFailed(List<Request> active) {
    active.removeIf(Request::isDone);
    activeRequests.set(active.size());
  }

  private void advance(List<Request> active) {
    for (Request request : active) {
      if (request.readyToDecode()
          && request.state.session.checkpoint() >= backend.contextCapacity()) {
        fail(request, new IllegalStateException("generation exceeded the model context window"));
      }
    }
    List<Request> runnable =
        active.stream()
            .filter(Request::readyToDecode)
            .filter(request -> request.state.session.checkpoint() < backend.contextCapacity())
            .toList();
    if (runnable.isEmpty()) {
      return;
    }

    InferenceSession[] sessions = new InferenceSession[runnable.size()];
    int[] tokens = new int[runnable.size()];
    for (int row = 0; row < runnable.size(); row++) {
      Request request = runnable.get(row);
      sessions[row] = request.state.session;
      tokens[row] = request.pendingToken;
    }

    try {
      LogitBatch logits;
      synchronized (executionLock) {
        logits = backend.forwardBatchTransient(sessions, tokens);
      }
      batchInvocations.incrementAndGet();
      sessionSteps.addAndGet(runnable.size());
      largestBatch.accumulateAndGet(runnable.size(), Math::max);
      for (int row = 0; row < runnable.size(); row++) {
        acceptLogits(runnable.get(row), logits, row);
      }
    } catch (RuntimeException | Error failure) {
      runnable.forEach(request -> fail(request, failure));
    }
  }

  private void acceptLogits(Request request, LogitBatch logits, int row) {
    try {
      int next = request.sampler.sample(logits, row, request.allTokens, request.constraint::allows);
      acceptToken(request, next);
    } catch (RuntimeException | Error failure) {
      fail(request, failure);
    }
  }

  private void acceptInitialLogits(Request request, float[] logits) {
    int next = request.sampler.sample(logits, request.allTokens, request.constraint::allows);
    acceptToken(request, next);
  }

  private void acceptToken(Request request, int next) {
    if (backend.tokenizer().isEndOfGeneration(next)) {
      completeGeneration(request);
      return;
    }
    request.generatedTokens++;
    if (request.firstTokenAt < 0) {
      request.firstTokenAt = System.nanoTime();
    }
    boolean stopped = request.emitter.emit(backend.tokenizer().decode(next));
    request.allTokens.add(next);
    request.constraint.accept(next);
    if (stopped
        || request.constraint.isComplete()
        || request.generatedTokens == request.options.maxTokens()) {
      completeGeneration(request);
    } else {
      request.pendingToken = next;
    }
  }

  private void completeGeneration(Request request) {
    request.emitter.finish();
    request.state.cachedPromptTokens = request.promptTokens.clone();
    long completedAt = System.nanoTime();
    GenerationUsage usage =
        new GenerationUsage(request.promptTokens.length, request.generatedTokens);
    GenerationMetrics metrics = request.generationMetrics(completedAt, usage);
    request.state.lastGenerationMetrics = metrics;
    try {
      request.stream.onComplete(usage);
      completedRequests.incrementAndGet();
      request.generationResult.complete(metrics);
    } catch (RuntimeException | Error failure) {
      fail(request, failure);
    }
  }

  private void completePrefill(Request request) {
    request.state.cachedPromptTokens = request.promptTokens.clone();
    long completedAt = System.nanoTime();
    PromptPrefillMetrics metrics =
        new PromptPrefillMetrics(
            request.tokenization,
            request.promptPreparation,
            request.prefill,
            Duration.ofNanos(elapsed(request.submittedAt, completedAt)),
            request.promptCache);
    completedRequests.incrementAndGet();
    request.prefillResult.complete(metrics);
  }

  private void fail(Request request, Throwable failure) {
    if (request.isDone()) {
      return;
    }
    request.state.cachedPromptTokens = null;
    try {
      synchronized (executionLock) {
        backend.reset(request.state.session);
      }
    } catch (RuntimeException | Error resetFailure) {
      failure.addSuppressed(resetFailure);
    }
    failedRequests.incrementAndGet();
    if (request.prefillOnly) {
      request.prefillResult.completeExceptionally(failure);
    } else {
      request.state.lastGenerationMetrics = request.generationMetrics(System.nanoTime(), null);
      signalError(request.stream, failure);
      request.generationResult.complete(request.state.lastGenerationMetrics);
    }
  }

  private static void signalError(TokenStream stream, Throwable failure) {
    try {
      stream.onError(failure);
    } catch (RuntimeException | Error ignored) {
      // The inference failure remains the terminal signal; callback failures cannot be rethrown.
    }
  }

  private static int reusablePrefixLength(int[] cached, int[] prompt) {
    if (cached == null) {
      return 0;
    }
    int shared = 0;
    int limit = Math.min(cached.length, prompt.length);
    while (shared < limit && cached[shared] == prompt[shared]) {
      shared++;
    }
    return Math.min(shared, prompt.length - 1);
  }

  private static long elapsed(long started, long completed) {
    return Math.max(0, completed - started);
  }

  static final class SessionState {
    private final InferenceSession session;
    private volatile int[] cachedPromptTokens;
    private volatile GenerationMetrics lastGenerationMetrics = GenerationMetrics.unavailable();

    private SessionState(InferenceSession session) {
      this.session = session;
    }

    GenerationMetrics lastGenerationMetrics() {
      return lastGenerationMetrics;
    }

    void invalidatePromptCache() {
      cachedPromptTokens = null;
    }
  }

  private record TokenizedPrompt(int[] tokens, Duration tokenization) {
    private TokenizedPrompt {
      tokens = tokens.clone();
    }
  }

  private static final class Request {
    private final SessionState state;
    private final int[] promptTokens;
    private final Duration tokenization;
    private final SamplingOptions options;
    private final TokenStream stream;
    private final TokenConstraint constraint;
    private final Sampler sampler;
    private final StopSequenceEmitter emitter;
    private final boolean prefillOnly;
    private final long submittedAt;
    private final CompletableFuture<GenerationMetrics> generationResult = new CompletableFuture<>();
    private final CompletableFuture<PromptPrefillMetrics> prefillResult = new CompletableFuture<>();
    private final List<Integer> allTokens = new ArrayList<>();
    private Duration promptPreparation = Duration.ZERO;
    private Duration prefill = Duration.ZERO;
    private PromptCacheMetrics promptCache = new PromptCacheMetrics(true, 0, 0, 0);
    private long firstTokenAt = -1;
    private int promptIndex;
    private int pendingToken = -1;
    private int generatedTokens;
    private boolean prepared;

    private Request(
        SessionState state,
        TokenizedPrompt prompt,
        SamplingOptions options,
        TokenStream stream,
        TokenConstraint constraint,
        boolean prefillOnly,
        long submittedAt) {
      this.state = Objects.requireNonNull(state, "state");
      promptTokens = prompt.tokens();
      tokenization = prompt.tokenization();
      this.options = options;
      this.stream = stream;
      this.constraint = constraint;
      sampler = options == null ? null : new Sampler(options);
      emitter =
          options == null || stream == null
              ? null
              : new StopSequenceEmitter(stream, options.stopSequences());
      this.prefillOnly = prefillOnly;
      this.submittedAt = submittedAt;
      Arrays.stream(promptTokens).forEach(allTokens::add);
    }

    static Request generation(
        SessionState state,
        TokenizedPrompt prompt,
        SamplingOptions options,
        TokenStream stream,
        TokenConstraint constraint,
        long submittedAt) {
      return new Request(state, prompt, options, stream, constraint, false, submittedAt);
    }

    static Request prefill(SessionState state, TokenizedPrompt prompt, long submittedAt) {
      return new Request(state, prompt, null, null, null, true, submittedAt);
    }

    boolean inPrompt() {
      return promptIndex < promptTokens.length;
    }

    boolean readyToDecode() {
      return prepared && !prefillOnly && !inPrompt() && pendingToken >= 0 && !isDone();
    }

    boolean isDone() {
      return prefillOnly ? prefillResult.isDone() : generationResult.isDone();
    }

    GenerationMetrics generationMetrics(long completedAt, GenerationUsage usage) {
      GenerationUsage measuredUsage =
          usage == null ? new GenerationUsage(promptTokens.length, generatedTokens) : usage;
      return new GenerationMetrics(
          true,
          usage != null,
          tokenization,
          promptPreparation,
          prefill,
          firstTokenAt < 0
              ? Optional.empty()
              : Optional.of(Duration.ofNanos(elapsed(submittedAt, firstTokenAt))),
          firstTokenAt < 0 ? Duration.ZERO : Duration.ofNanos(elapsed(firstTokenAt, completedAt)),
          Duration.ofNanos(elapsed(submittedAt, completedAt)),
          measuredUsage,
          promptCache);
    }
  }
}
