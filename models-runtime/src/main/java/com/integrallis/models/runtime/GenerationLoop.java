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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

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
  private volatile SpeculativeGenerationMetrics lastSpeculativeMetrics =
      SpeculativeGenerationMetrics.inactive();
  private volatile PromptCacheMetrics lastPromptCacheMetrics = PromptCacheMetrics.unavailable();
  private int[] cachedPromptTokens;

  public GenerationLoop(InferenceBackend backend) {
    this(backend, SpeculativeGenerationOptions.disabled());
  }

  public GenerationLoop(InferenceBackend backend, SpeculativeGenerationOptions speculativeOptions) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.speculativeOptions = Objects.requireNonNull(speculativeOptions, "speculativeOptions");
  }

  /** Returns the measurements captured by the most recently completed request. */
  public SpeculativeGenerationMetrics lastSpeculativeMetrics() {
    return lastSpeculativeMetrics;
  }

  /** Returns prompt-prefix cache measurements for the most recently completed request. */
  public PromptCacheMetrics lastPromptCacheMetrics() {
    return lastPromptCacheMetrics;
  }

  /** Generates text from a prompt, returning the complete generated string. */
  public String generate(String prompt, SamplingOptions options) {
    Objects.requireNonNull(prompt, "prompt");
    if (prompt.isEmpty()) {
      throw new IllegalArgumentException("prompt must not be empty");
    }
    Objects.requireNonNull(options, "options");

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
        });
    return result.toString();
  }

  /** Generates text with streaming output via a TokenStream callback. */
  public void generate(String prompt, SamplingOptions options, TokenStream stream) {
    Objects.requireNonNull(prompt, "prompt");
    if (prompt.isEmpty()) {
      throw new IllegalArgumentException("prompt must not be empty");
    }
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(stream, "stream");

    synchronized (backend) {
      Tokenizer tokenizer = backend.tokenizer();
      int[] promptTokens = tokenizer.encode(prompt);
      if (promptTokens.length == 0) {
        throw new IllegalArgumentException("prompt produced no tokens");
      }
      PromptPrefill promptPrefill = preparePrompt(promptTokens);

      Sampler sampler = new Sampler(options);
      List<Integer> allTokens = new ArrayList<>();
      boolean speculativeActive =
          speculativeOptions.enabled() && backend instanceof SpeculativeInferenceBackend;
      MutableSpeculativeMetrics speculativeMetrics =
          new MutableSpeculativeMetrics(speculativeActive, speculativeOptions.maximumDraftTokens());

      try {
        float[] logits =
            backend.prefill(promptPrefill.tokensToEvaluate(), promptPrefill.startPosition());
        for (int token : promptTokens) {
          allTokens.add(token);
        }
        int position = promptTokens.length;

        if (speculativeActive) {
          generateSpeculatively(
              (SpeculativeInferenceBackend) backend,
              tokenizer,
              sampler,
              stream,
              allTokens,
              logits,
              position,
              options.maxTokens(),
              speculativeMetrics);
        } else {
          generateSequentially(
              tokenizer, sampler, stream, allTokens, logits, position, options.maxTokens());
        }

        cachedPromptTokens =
            backend instanceof RewindableInferenceBackend ? promptTokens.clone() : null;
        stream.onComplete();
      } catch (Exception e) {
        cachedPromptTokens = null;
        stream.onError(e);
      } finally {
        lastSpeculativeMetrics = speculativeMetrics.snapshot();
        lastPromptCacheMetrics = promptPrefill.metrics();
      }
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
      TokenStream stream,
      List<Integer> allTokens,
      float[] initialLogits,
      int initialPosition,
      int maxTokens) {
    float[] logits = initialLogits;
    int position = initialPosition;
    for (int generated = 0; generated < maxTokens; generated++) {
      int nextToken = sampler.sample(logits, allTokens);
      if (tokenizer.isEndOfGeneration(nextToken)) {
        return;
      }
      emit(tokenizer, stream, allTokens, nextToken);
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
      TokenStream stream,
      List<Integer> allTokens,
      float[] initialLogits,
      int initialPosition,
      int maxTokens,
      MutableSpeculativeMetrics metrics) {
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

      emit(tokenizer, stream, allTokens, nextToken);
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
        emit(tokenizer, stream, allTokens, targetToken);
        accepted++;
        generated++;
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

      if (reachedEos || generated == maxTokens) {
        return;
      }
      if (accepted == draft.length) {
        carriedLogits = verification;
        carriedLogitRow = draft.length;
      }
    }
  }

  private static void emit(
      Tokenizer tokenizer, TokenStream stream, List<Integer> allTokens, int token) {
    stream.onToken(tokenizer.decode(token));
    allTokens.add(token);
  }

  private record PromptPrefill(
      int startPosition, int[] tokensToEvaluate, PromptCacheMetrics metrics) {}

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
