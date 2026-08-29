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

import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.LongSupplier;

/**
 * Autoregressive generation loop that drives an inference backend to generate text.
 *
 * <p>Calls that share the same backend instance are serialized because stateful backends may own a
 * single key-value cache. Rewindable backends retain the longest exact token prefix between
 * sequential requests and prefill only the changed suffix.
 */
public final class GenerationLoop {

  private final InferenceBackend backend;
  private final SpeculativeGenerationOptions speculativeOptions;
  private final LongSupplier nanoTime;
  private volatile SpeculativeGenerationMetrics lastSpeculativeMetrics =
      SpeculativeGenerationMetrics.inactive();
  private volatile PromptCacheMetrics lastPromptCacheMetrics = PromptCacheMetrics.unavailable();
  private volatile GenerationMetrics lastGenerationMetrics = GenerationMetrics.unavailable();
  private int[] cachedPromptTokens;

  public GenerationLoop(InferenceBackend backend) {
    this(backend, SpeculativeGenerationOptions.disabled(), System::nanoTime);
  }

  public GenerationLoop(InferenceBackend backend, SpeculativeGenerationOptions speculativeOptions) {
    this(backend, speculativeOptions, System::nanoTime);
  }

  GenerationLoop(
      InferenceBackend backend,
      SpeculativeGenerationOptions speculativeOptions,
      LongSupplier nanoTime) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.speculativeOptions = Objects.requireNonNull(speculativeOptions, "speculativeOptions");
    this.nanoTime = Objects.requireNonNull(nanoTime, "nanoTime");
  }

  /** Returns the measurements captured by the most recently completed request. */
  public SpeculativeGenerationMetrics lastSpeculativeMetrics() {
    return lastSpeculativeMetrics;
  }

  /** Returns prompt-prefix cache measurements for the most recently completed request. */
  public PromptCacheMetrics lastPromptCacheMetrics() {
    return lastPromptCacheMetrics;
  }

  /** Returns phase timings and token usage for the most recently completed request. */
  public GenerationMetrics lastGenerationMetrics() {
    return lastGenerationMetrics;
  }

  /**
   * Forgets the prompt-token view retained by the high-level generation path.
   *
   * <p>Package-level pipeline operations invoke this before changing backend context directly so a
   * later generation never reuses a prefix that no longer describes the backend state.
   */
  void invalidatePromptCache() {
    cachedPromptTokens = null;
  }

  /** Generates text from a prompt, returning the complete generated string. */
  public String generate(String prompt, SamplingOptions options) {
    return generate(ModelPrompt.text(Objects.requireNonNull(prompt, "prompt")), options);
  }

  /** Generates text from a prompt while enforcing a token-level constraint. */
  public String generate(String prompt, SamplingOptions options, TokenConstraint constraint) {
    return generate(
        ModelPrompt.text(Objects.requireNonNull(prompt, "prompt")), options, constraint);
  }

  /** Generates text from a segmented prompt, returning the complete generated string. */
  public String generate(ModelPrompt prompt, SamplingOptions options) {
    return generate(prompt, options, TokenConstraint.unrestricted());
  }

  /** Generates text from a segmented prompt while enforcing a token-level constraint. */
  public String generate(ModelPrompt prompt, SamplingOptions options, TokenConstraint constraint) {
    requirePrompt(prompt);
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(constraint, "constraint");

    StringBuilder result = new StringBuilder();
    generate(
        prompt,
        options,
        new TokenStream() {
          @Override
          public void onToken(String token) {
            result.append(token);
          }

          @Override
          public void onComplete() {}

          @Override
          public void onError(Throwable t) {
            throw new RuntimeException("Generation error", t);
          }
        },
        constraint);
    return result.toString();
  }

  /** Generates text with streaming output via a TokenStream callback. */
  public void generate(String prompt, SamplingOptions options, TokenStream stream) {
    generate(ModelPrompt.text(Objects.requireNonNull(prompt, "prompt")), options, stream);
  }

  /** Generates text with streaming output while enforcing a token-level constraint. */
  public void generate(
      String prompt, SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
    generate(
        ModelPrompt.text(Objects.requireNonNull(prompt, "prompt")), options, stream, constraint);
  }

  /** Generates text from a segmented prompt with streaming output via a callback. */
  public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
    generate(prompt, options, stream, TokenConstraint.unrestricted());
  }

  /** Generates text from a segmented prompt with streaming output and a token-level constraint. */
  public void generate(
      ModelPrompt prompt, SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
    requirePrompt(prompt);
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(stream, "stream");
    Objects.requireNonNull(constraint, "constraint");

    synchronized (backend) {
      long requestStarted = nanoTime.getAsLong();
      long phaseStarted = requestStarted;
      Tokenizer tokenizer = backend.tokenizer();
      int[] promptTokens = tokenizer.encode(prompt);
      long phaseCompleted = nanoTime.getAsLong();
      long tokenizationNanos = elapsed(phaseStarted, phaseCompleted);
      if (promptTokens.length == 0) {
        throw new IllegalArgumentException("prompt produced no tokens");
      }
      phaseStarted = phaseCompleted;
      PromptPrefill promptPrefill = preparePrompt(promptTokens);
      phaseCompleted = nanoTime.getAsLong();
      long promptPreparationNanos = elapsed(phaseStarted, phaseCompleted);

      Sampler sampler = new Sampler(options);
      StopSequenceEmitter emitter = new StopSequenceEmitter(stream, options.stopSequences());
      List<Integer> allTokens = new ArrayList<>();
      boolean speculativeActive =
          constraint == TokenConstraint.unrestricted()
              && speculativeOptions.enabled()
              && backend instanceof SpeculativeInferenceBackend;
      MutableSpeculativeMetrics speculativeMetrics =
          new MutableSpeculativeMetrics(speculativeActive, speculativeOptions.maximumDraftTokens());
      MutableGenerationMetrics generationMetrics =
          new MutableGenerationMetrics(
              requestStarted, tokenizationNanos, promptPreparationNanos, nanoTime);
      boolean successful = false;

      try {
        long prefillStarted = nanoTime.getAsLong();
        float[] logits;
        try {
          logits = backend.prefill(promptPrefill.tokensToEvaluate(), promptPrefill.startPosition());
        } finally {
          generationMetrics.prefillNanos = elapsed(prefillStarted, nanoTime.getAsLong());
        }
        for (int token : promptTokens) {
          allTokens.add(token);
        }
        int position = promptTokens.length;

        if (speculativeActive) {
          generateSpeculatively(
              (SpeculativeInferenceBackend) backend,
              tokenizer,
              sampler,
              emitter,
              allTokens,
              logits,
              position,
              options.maxTokens(),
              speculativeMetrics,
              generationMetrics);
        } else {
          generateSequentially(
              tokenizer,
              sampler,
              emitter,
              allTokens,
              logits,
              position,
              options.maxTokens(),
              constraint,
              generationMetrics);
        }

        emitter.finish();
        cachedPromptTokens =
            backend instanceof RewindableInferenceBackend ? promptTokens.clone() : null;
        successful = true;
        stream.onComplete(
            new GenerationUsage(promptTokens.length, allTokens.size() - promptTokens.length));
      } catch (Exception e) {
        cachedPromptTokens = null;
        stream.onError(e);
      } finally {
        lastSpeculativeMetrics = speculativeMetrics.snapshot();
        lastPromptCacheMetrics = promptPrefill.metrics();
        lastGenerationMetrics =
            generationMetrics.snapshot(
                successful,
                new GenerationUsage(
                    promptTokens.length, Math.max(0, allTokens.size() - promptTokens.length)),
                promptPrefill.metrics(),
                nanoTime.getAsLong());
      }
    }
  }

  private static void requirePrompt(ModelPrompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    if (prompt.isEmpty()) {
      throw new IllegalArgumentException("prompt must not be empty");
    }
  }

  private PromptPrefill preparePrompt(int[] promptTokens) {
    if (!(backend instanceof RewindableInferenceBackend rewindableBackend)) {
      backend.reset();
      return new PromptPrefill(
          0, promptTokens, new PromptCacheMetrics(false, promptTokens.length, 0, 0));
    }

    int reusableTokens = reusablePrefixLength(promptTokens);
    if (reusableTokens == 0 || rewindableBackend.checkpoint() < reusableTokens) {
      backend.reset();
      reusableTokens = 0;
    } else {
      rewindableBackend.rewind(reusableTokens);
    }
    return new PromptPrefill(
        reusableTokens,
        Arrays.copyOfRange(promptTokens, reusableTokens, promptTokens.length),
        new PromptCacheMetrics(
            true, promptTokens.length, reusableTokens, promptTokens.length - reusableTokens));
  }

  private int reusablePrefixLength(int[] promptTokens) {
    if (cachedPromptTokens == null) {
      return 0;
    }
    int shared = 0;
    int limit = Math.min(cachedPromptTokens.length, promptTokens.length);
    while (shared < limit && cachedPromptTokens[shared] == promptTokens[shared]) {
      shared++;
    }
    return Math.min(shared, promptTokens.length - 1);
  }

  private void generateSequentially(
      Tokenizer tokenizer,
      Sampler sampler,
      StopSequenceEmitter emitter,
      List<Integer> allTokens,
      float[] initialLogits,
      int initialPosition,
      int maxTokens,
      TokenConstraint constraint,
      MutableGenerationMetrics metrics) {
    float[] logits = initialLogits;
    int position = initialPosition;
    for (int generated = 0; generated < maxTokens; generated++) {
      int nextToken = sampler.sample(logits, allTokens, constraint::allows);
      if (tokenizer.isEndOfGeneration(nextToken)) {
        return;
      }
      if (emit(tokenizer, emitter, allTokens, nextToken, metrics)) {
        return;
      }
      constraint.accept(nextToken);
      if (constraint.isComplete()) {
        return;
      }
      if (generated + 1 == maxTokens) {
        return;
      }
      logits = backend.forwardTransient(nextToken, position);
      position++;
    }
  }

  private void generateSpeculatively(
      SpeculativeInferenceBackend speculativeBackend,
      Tokenizer tokenizer,
      Sampler sampler,
      StopSequenceEmitter emitter,
      List<Integer> allTokens,
      float[] initialLogits,
      int initialPosition,
      int maxTokens,
      MutableSpeculativeMetrics metrics,
      MutableGenerationMetrics generationMetrics) {
    NgramDraftStrategy strategy = new NgramDraftStrategy(speculativeOptions);
    float[] logits = initialLogits;
    LogitBatch carriedLogits = null;
    int carriedLogitRow = -1;
    Integer carriedToken = null;
    int position = initialPosition;
    int generated = 0;

    while (generated < maxTokens) {
      int nextToken;
      if (carriedToken != null) {
        nextToken = carriedToken;
      } else if (carriedLogits != null) {
        nextToken = sampler.sample(carriedLogits, carriedLogitRow, allTokens);
      } else {
        nextToken = sampler.sample(logits, allTokens);
      }
      carriedToken = null;
      carriedLogits = null;
      carriedLogitRow = -1;
      if (tokenizer.isEndOfGeneration(nextToken)) {
        return;
      }

      int remainingAfterPending = maxTokens - generated - 1;
      int[] draft;
      if (strategy.isSuppressed(generated)) {
        metrics.suppressedSteps++;
        draft = new int[0];
      } else {
        long searchStart = System.nanoTime();
        draft = strategy.propose(allTokens, nextToken, remainingAfterPending);
        metrics.draftSearchNanos += System.nanoTime() - searchStart;
      }

      if (emit(tokenizer, emitter, allTokens, nextToken, generationMetrics)) {
        return;
      }
      generated++;
      if (generated == maxTokens) {
        return;
      }

      if (draft.length == 0) {
        long forwardStart = System.nanoTime();
        logits = speculativeBackend.forwardTransient(nextToken, position);
        metrics.ordinaryForwardNanos += System.nanoTime() - forwardStart;
        metrics.ordinaryForwardCalls++;
        position++;
        continue;
      }

      int checkpoint = speculativeBackend.checkpoint();
      if (checkpoint != position) {
        throw new IllegalStateException(
            "speculative checkpoint does not match generation position: "
                + checkpoint
                + " != "
                + position);
      }
      int[] verificationTokens = new int[draft.length + 1];
      verificationTokens[0] = nextToken;
      System.arraycopy(draft, 0, verificationTokens, 1, draft.length);

      long verificationStart = System.nanoTime();
      LogitBatch verification = speculativeBackend.verifyTransient(verificationTokens, checkpoint);
      metrics.verificationNanos += System.nanoTime() - verificationStart;
      metrics.draftAttempts++;
      metrics.proposedTokens += draft.length;
      metrics.verificationBatchHistogram[verificationTokens.length]++;
      for (int draftPosition = 0; draftPosition < draft.length; draftPosition++) {
        metrics.proposedByPosition[draftPosition]++;
      }

      int accepted = 0;
      boolean reachedEos = false;
      boolean reachedStopSequence = false;
      while (accepted < draft.length && generated < maxTokens) {
        int targetToken = sampler.sample(verification, accepted, allTokens);
        if (tokenizer.isEndOfGeneration(targetToken)) {
          reachedEos = true;
          break;
        }
        if (targetToken != draft[accepted]) {
          carriedToken = targetToken;
          break;
        }
        reachedStopSequence = emit(tokenizer, emitter, allTokens, targetToken, generationMetrics);
        accepted++;
        generated++;
        if (reachedStopSequence) {
          break;
        }
      }
      metrics.acceptedTokens += accepted;
      for (int acceptedPosition = 0; acceptedPosition < accepted; acceptedPosition++) {
        metrics.acceptedByPosition[acceptedPosition]++;
      }
      strategy.recordVerification(draft.length, accepted, generated);

      int retainedCheckpoint = checkpoint + 1 + accepted;
      if (accepted < draft.length) {
        speculativeBackend.rewind(retainedCheckpoint);
        metrics.rollbacks++;
      }
      position = retainedCheckpoint;

      if (reachedEos || reachedStopSequence || generated == maxTokens) {
        return;
      }
      if (accepted == draft.length) {
        carriedLogits = verification;
        carriedLogitRow = draft.length;
      }
    }
  }

  private static boolean emit(
      Tokenizer tokenizer,
      StopSequenceEmitter emitter,
      List<Integer> allTokens,
      int token,
      MutableGenerationMetrics metrics) {
    metrics.tokenGenerated();
    boolean stopped = emitter.emit(tokenizer.decode(token));
    allTokens.add(token);
    return stopped;
  }

  private record PromptPrefill(
      int startPosition, int[] tokensToEvaluate, PromptCacheMetrics metrics) {}

  private static long elapsed(long started, long completed) {
    return Math.max(0, completed - started);
  }

  private static final class MutableGenerationMetrics {
    private final long requestStarted;
    private final long tokenizationNanos;
    private final long promptPreparationNanos;
    private final LongSupplier nanoTime;
    private long prefillNanos;
    private long firstTokenAt = -1;

    private MutableGenerationMetrics(
        long requestStarted,
        long tokenizationNanos,
        long promptPreparationNanos,
        LongSupplier nanoTime) {
      this.requestStarted = requestStarted;
      this.tokenizationNanos = tokenizationNanos;
      this.promptPreparationNanos = promptPreparationNanos;
      this.nanoTime = nanoTime;
    }

    private void tokenGenerated() {
      if (firstTokenAt < 0) {
        firstTokenAt = nanoTime.getAsLong();
      }
    }

    private GenerationMetrics snapshot(
        boolean successful,
        GenerationUsage usage,
        PromptCacheMetrics promptCache,
        long completedAt) {
      long totalNanos = elapsed(requestStarted, completedAt);
      Optional<java.time.Duration> timeToFirstToken =
          firstTokenAt < 0
              ? Optional.empty()
              : Optional.of(java.time.Duration.ofNanos(elapsed(requestStarted, firstTokenAt)));
      long decodeNanos = firstTokenAt < 0 ? 0 : elapsed(firstTokenAt, completedAt);
      return new GenerationMetrics(
          true,
          successful,
          java.time.Duration.ofNanos(tokenizationNanos),
          java.time.Duration.ofNanos(promptPreparationNanos),
          java.time.Duration.ofNanos(prefillNanos),
          timeToFirstToken,
          java.time.Duration.ofNanos(decodeNanos),
          java.time.Duration.ofNanos(totalNanos),
          usage,
          promptCache);
    }
  }

  private static final class StopSequenceEmitter {
    private final TokenStream stream;
    private final List<String> stopSequences;
    private final StringBuilder pending = new StringBuilder();

    private StopSequenceEmitter(TokenStream stream, List<String> stopSequences) {
      this.stream = stream;
      this.stopSequences = stopSequences;
    }

    private boolean emit(String token) {
      if (stopSequences.isEmpty()) {
        stream.onToken(token);
        return false;
      }

      pending.append(token);
      int stopIndex = earliestStopIndex();
      if (stopIndex >= 0) {
        flush(stopIndex);
        pending.setLength(0);
        return true;
      }

      int retainedSuffix = longestPotentialStopPrefix();
      flush(pending.length() - retainedSuffix);
      return false;
    }

    private void finish() {
      flush(pending.length());
    }

    private int earliestStopIndex() {
      int earliest = -1;
      for (String stopSequence : stopSequences) {
        int index = pending.indexOf(stopSequence);
        if (index >= 0 && (earliest < 0 || index < earliest)) {
          earliest = index;
        }
      }
      return earliest;
    }

    private int longestPotentialStopPrefix() {
      int retained = 0;
      for (String stopSequence : stopSequences) {
        int limit = Math.min(pending.length(), stopSequence.length() - 1);
        for (int length = limit; length > retained; length--) {
          if (pending
              .substring(pending.length() - length)
              .equals(stopSequence.substring(0, length))) {
            retained = length;
            break;
          }
        }
      }
      return retained;
    }

    private void flush(int length) {
      if (length <= 0) {
        return;
      }
      stream.onToken(pending.substring(0, length));
      pending.delete(0, length);
    }
  }

  private static final class MutableSpeculativeMetrics {
    private final boolean active;
    private int draftAttempts;
    private int proposedTokens;
    private int acceptedTokens;
    private int rollbacks;
    private int suppressedSteps;
    private int ordinaryForwardCalls;
    private final int[] proposedByPosition;
    private final int[] acceptedByPosition;
    private final int[] verificationBatchHistogram;
    private long draftSearchNanos;
    private long verificationNanos;
    private long ordinaryForwardNanos;

    private MutableSpeculativeMetrics(boolean active, int maximumDraftTokens) {
      this.active = active;
      proposedByPosition = new int[maximumDraftTokens];
      acceptedByPosition = new int[maximumDraftTokens];
      verificationBatchHistogram = new int[maximumDraftTokens + 2];
    }

    private SpeculativeGenerationMetrics snapshot() {
      return new SpeculativeGenerationMetrics(
          active,
          draftAttempts,
          proposedTokens,
          acceptedTokens,
          rollbacks,
          suppressedSteps,
          ordinaryForwardCalls,
          toList(proposedByPosition),
          toList(acceptedByPosition),
          toList(verificationBatchHistogram),
          draftSearchNanos,
          verificationNanos,
          ordinaryForwardNanos);
    }

    private static List<Integer> toList(int[] values) {
      List<Integer> result = new ArrayList<>(values.length);
      for (int value : values) {
        result.add(value);
      }
      return List.copyOf(result);
    }
  }
}
