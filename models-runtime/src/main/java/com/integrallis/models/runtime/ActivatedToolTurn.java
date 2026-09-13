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
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.TokenStream;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicBoolean;

/** Request-scoped activated tool-selection and exact base-response branches. */
public final class ActivatedToolTurn implements SharedToolTurn {
  private final InferencePipeline pipeline;
  private final int[] invocationTokens;
  private final ModelPrompt renderedToolPrompt;
  private final int activationBoundaryTokens;
  private final int sharedPrefixTokens;
  private final long sharedPrefixBytes;
  private final TextGenerationSession base;
  private final TextGenerationSession tool;
  private final boolean physicallyShared;
  private final ActivatedToolCallingModel.PrefixStrategy prefixStrategy;
  private final int minimumSharedPrefixTokens;
  private final AtomicBoolean closed = new AtomicBoolean();
  private boolean toolGenerated;
  private boolean responseGenerated;
  private boolean advanced;
  private boolean failed;

  ActivatedToolTurn(
      InferencePipeline pipeline,
      int[] invocationTokens,
      ModelPrompt renderedToolPrompt,
      int activationBoundaryTokens,
      int sharedPrefixTokens,
      long sharedPrefixBytes,
      TextGenerationSession base,
      TextGenerationSession tool,
      boolean physicallyShared,
      ActivatedToolCallingModel.PrefixStrategy prefixStrategy,
      int minimumSharedPrefixTokens) {
    this.pipeline = Objects.requireNonNull(pipeline, "pipeline");
    this.invocationTokens = Objects.requireNonNull(invocationTokens, "invocationTokens").clone();
    this.renderedToolPrompt = renderedToolPrompt;
    this.activationBoundaryTokens = activationBoundaryTokens;
    this.sharedPrefixTokens = sharedPrefixTokens;
    this.sharedPrefixBytes = sharedPrefixBytes;
    this.base = base;
    this.tool = tool;
    this.physicallyShared = physicallyShared;
    this.prefixStrategy = Objects.requireNonNull(prefixStrategy, "prefixStrategy");
    this.minimumSharedPrefixTokens = minimumSharedPrefixTokens;
  }

  /** Generates the structured tool selection on the activated branch. */
  public synchronized String generateToolCall(SamplingOptions options, TokenConstraint constraint) {
    requireToolGenerationAvailable();
    try {
      String result = tool.generate(renderedToolPrompt, options, constraint);
      toolGenerated = true;
      return result;
    } catch (RuntimeException | Error failure) {
      failed = true;
      throw failure;
    }
  }

  /** Streams the structured tool selection on the activated branch. */
  public synchronized void generateToolCall(
      SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
    requireToolGenerationAvailable();
    TerminalTrackingStream tracking = new TerminalTrackingStream(stream);
    try {
      tool.generate(renderedToolPrompt, options, tracking, constraint);
      if (tracking.failed()) {
        failed = true;
        return;
      }
      tracking.requireCompleted();
      toolGenerated = true;
    } catch (RuntimeException | Error failure) {
      failed = true;
      throw failure;
    }
  }

  @Override
  public synchronized ActivatedToolTurn continueToolSelection(ModelPrompt nextToolPrompt) {
    requireContinuationAvailable();
    Objects.requireNonNull(nextToolPrompt, "nextToolPrompt");
    int[] promptTokens = pipeline.tokenize(nextToolPrompt);
    int prefixLength = ActivatedToolCallingModel.lastIndexOf(promptTokens, invocationTokens);
    if (prefixLength <= activationBoundaryTokens) {
      throw new IllegalArgumentException(
          "continued tool prompt must extend the shared prefix and contain the activated "
              + "adapter invocation sequence");
    }
    int[] sharedTokens = java.util.Arrays.copyOf(promptTokens, prefixLength);

    TextGenerationSession nextBase = null;
    TextGenerationSession nextTool = null;
    try {
      if (ActivatedToolCallingModel.shouldShare(
          prefixLength, prefixStrategy, minimumSharedPrefixTokens)) {
        SharedPromptPrefix nextPrefix =
            responseGenerated
                ? pipeline.reconcileSharedTokenPrefix(base, sharedTokens)
                : pipeline.extendSharedTokenPrefix(base, sharedTokens);
        tool.close();
        advanced = true;
        nextBase =
            pipeline.openGenerationSession(nextPrefix, SharedPrefixInferenceBackend.Branch.BASE);
        nextTool =
            pipeline.openGenerationSession(
                nextPrefix, SharedPrefixInferenceBackend.Branch.ACTIVATED_ADAPTER);
        boolean nextPhysicallyShared = nextBase.sharesPrefixStorageWith(nextTool);
        if (!nextPhysicallyShared) {
          throw new IllegalStateException(
              "backend violated the shared-prefix contract while extending a tool turn");
        }
        return new ActivatedToolTurn(
            pipeline,
            invocationTokens,
            nextToolPrompt,
            nextPrefix.tokenCount(),
            nextPrefix.tokenCount(),
            nextPrefix.sharedBytes(),
            nextBase,
            nextTool,
            true,
            prefixStrategy,
            minimumSharedPrefixTokens);
      }

      base.prefillTokenPrefix(sharedTokens);
      nextBase = base;
      nextTool =
          pipeline.openGenerationSessionAfterBasePrefix(
              sharedTokens, SharedPrefixInferenceBackend.Branch.ACTIVATED_ADAPTER);
      if (nextBase.sharesPrefixStorageWith(nextTool)) {
        throw new IllegalStateException(
            "recomputed branches unexpectedly share physical KV storage");
      }
      tool.close();
      advanced = true;
      return new ActivatedToolTurn(
          pipeline,
          invocationTokens,
          nextToolPrompt,
          prefixLength,
          0,
          0,
          nextBase,
          nextTool,
          false,
          prefixStrategy,
          minimumSharedPrefixTokens);
    } catch (RuntimeException | Error failure) {
      failed = true;
      ActivatedToolCallingModel.closeAfterFailure(failure, nextTool, nextBase);
      throw failure;
    }
  }

  /** Generates conversational text on the exact base branch after tool execution. */
  public synchronized String generateBaseResponse(ModelPrompt prompt, SamplingOptions options) {
    requireResponseGenerationAvailable();
    try {
      String result = base.generate(Objects.requireNonNull(prompt, "prompt"), options);
      responseGenerated = true;
      return result;
    } catch (RuntimeException | Error failure) {
      failed = true;
      throw failure;
    }
  }

  /** Streams conversational text on the exact base branch after tool execution. */
  public synchronized void generateBaseResponse(
      ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
    requireResponseGenerationAvailable();
    TerminalTrackingStream tracking = new TerminalTrackingStream(stream);
    try {
      base.generate(Objects.requireNonNull(prompt, "prompt"), options, tracking);
      if (tracking.failed()) {
        failed = true;
        return;
      }
      tracking.requireCompleted();
      responseGenerated = true;
    } catch (RuntimeException | Error failure) {
      failed = true;
      throw failure;
    }
  }

  /** Returns the number of prompt tokens evaluated once and shared by both branches. */
  public int sharedPrefixTokens() {
    return sharedPrefixTokens;
  }

  /** Returns the physical KV-cache bytes referenced by both branches. */
  public long sharedPrefixBytes() {
    return sharedPrefixBytes;
  }

  /** Returns the runtime-verified physical-sharing result. */
  public boolean physicallySharesPrefix() {
    return physicallyShared;
  }

  /** Returns all inference-state bytes reachable by the exact base branch, when measurable. */
  public synchronized OptionalLong baseInferenceStateBytes() {
    requireOpen();
    return base.allocatedInferenceStateBytes();
  }

  /** Returns all inference-state bytes reachable by the activated branch, when measurable. */
  public synchronized OptionalLong toolInferenceStateBytes() {
    requireOpen();
    return tool.allocatedInferenceStateBytes();
  }

  /** Returns the unique bytes held by both branches, counting their physical shared prefix once. */
  public synchronized OptionalLong uniqueInferenceStateBytes() {
    requireOpen();
    OptionalLong baseBytes = base.allocatedInferenceStateBytes();
    OptionalLong toolBytes = tool.allocatedInferenceStateBytes();
    if (baseBytes.isEmpty() || toolBytes.isEmpty()) {
      return OptionalLong.empty();
    }
    long total = Math.addExact(baseBytes.orElseThrow(), toolBytes.orElseThrow());
    return OptionalLong.of(physicallyShared ? Math.subtractExact(total, sharedPrefixBytes) : total);
  }

  /** Returns metrics for the completed activated tool-selection generation. */
  public synchronized GenerationMetrics toolMetrics() {
    requireOpen();
    return tool.lastGenerationMetrics();
  }

  /** Returns metrics for the completed exact base-response generation. */
  public synchronized GenerationMetrics responseMetrics() {
    requireOpen();
    return base.lastGenerationMetrics();
  }

  synchronized TextGenerationSession consumeBaseBranch() {
    requireUsable();
    if (!responseGenerated) {
      throw new IllegalStateException(
          "generate the base response before resuming the conversation");
    }
    if (advanced) {
      throw new IllegalStateException("tool turn has already advanced");
    }
    tool.close();
    advanced = true;
    return base;
  }

  synchronized boolean failed() {
    requireOpen();
    return failed;
  }

  @Override
  public synchronized void close() {
    if (closed.compareAndSet(false, true)) {
      if (advanced) {
        return;
      }
      Throwable failure = null;
      try {
        tool.close();
      } catch (RuntimeException | Error closeFailure) {
        failure = closeFailure;
      }
      try {
        base.close();
      } catch (RuntimeException | Error closeFailure) {
        if (failure == null) {
          failure = closeFailure;
        } else {
          failure.addSuppressed(closeFailure);
        }
      }
      if (failure instanceof RuntimeException runtimeFailure) {
        throw runtimeFailure;
      }
      if (failure instanceof Error error) {
        throw error;
      }
    }
  }

  private void requireToolGenerationAvailable() {
    requireUsable();
    if (toolGenerated) {
      throw new IllegalStateException("tool selection has already been generated for this turn");
    }
  }

  private void requireResponseGenerationAvailable() {
    requireUsable();
    if (!toolGenerated) {
      throw new IllegalStateException("generate the tool selection before the base response");
    }
    if (responseGenerated) {
      throw new IllegalStateException("base response has already been generated for this turn");
    }
  }

  private void requireContinuationAvailable() {
    requireUsable();
    if (!toolGenerated) {
      throw new IllegalStateException("generate the tool selection before continuing the turn");
    }
    if (advanced) {
      throw new IllegalStateException("tool turn has already advanced to another selection");
    }
  }

  private void requireOpen() {
    if (closed.get()) {
      throw new IllegalStateException("activated tool turn is closed");
    }
  }

  private void requireUsable() {
    requireOpen();
    if (failed) {
      throw new IllegalStateException("activated tool turn failed and cannot be reused");
    }
  }

  private static final class TerminalTrackingStream implements TokenStream {
    private final TokenStream delegate;
    private boolean completed;
    private boolean failed;

    private TerminalTrackingStream(TokenStream delegate) {
      this.delegate = Objects.requireNonNull(delegate, "stream");
    }

    @Override
    public void onToken(String token) {
      delegate.onToken(token);
    }

    @Override
    public void onComplete() {
      completed = true;
      delegate.onComplete();
    }

    @Override
    public void onComplete(GenerationUsage usage) {
      completed = true;
      delegate.onComplete(usage);
    }

    @Override
    public void onError(Throwable failure) {
      failed = true;
      delegate.onError(failure);
    }

    private boolean failed() {
      return failed;
    }

    private void requireCompleted() {
      if (!completed) {
        throw new IllegalStateException(
            "generation returned without a terminal TokenStream callback");
      }
    }
  }
}
