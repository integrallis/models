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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One stateful conversation over an exact base model and its activated tool specialist.
 *
 * <p>Ordinary responses retain one base-model cache lineage. A tool selection consumes that base
 * lineage into an immutable prefix, forks physically shared base and activated branches, and
 * returns the exact base branch to this conversation after the tool result is synthesized. A later
 * tool selection therefore extends the retained base branch instead of rebuilding the conversation
 * from token zero.
 */
public final class ActivatedToolConversation implements AutoCloseable {
  private final ActivatedToolCallingModel model;
  private final InferencePipeline pipeline;
  private final AtomicBoolean closed = new AtomicBoolean();
  private TextGenerationSession baseSession;
  private ActivatedToolTurn activeTurn;
  private GenerationMetrics lastToolMetrics = GenerationMetrics.unavailable();
  private GenerationMetrics lastBaseMetrics = GenerationMetrics.unavailable();
  private int sharedPrefixTokens;
  private long sharedPrefixBytes;
  private boolean physicallySharedPrefix;

  ActivatedToolConversation(ActivatedToolCallingModel model, InferencePipeline pipeline) {
    this.model = Objects.requireNonNull(model, "model");
    this.pipeline = Objects.requireNonNull(pipeline, "pipeline");
  }

  /** Generates an ordinary exact-base response while retaining its prompt cache. */
  public synchronized String generateBase(ModelPrompt prompt, SamplingOptions options) {
    requireBaseAvailable();
    TextGenerationSession base = baseSession();
    String output = base.generate(Objects.requireNonNull(prompt, "prompt"), options);
    lastBaseMetrics = base.lastGenerationMetrics();
    return output;
  }

  /** Streams an ordinary exact-base response while retaining its prompt cache. */
  public synchronized void generateBase(
      ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
    requireBaseAvailable();
    TextGenerationSession base = baseSession();
    base.generate(Objects.requireNonNull(prompt, "prompt"), options, stream);
    lastBaseMetrics = base.lastGenerationMetrics();
  }

  /**
   * Selects a tool on the activated branch.
   *
   * <p>Calling this again before {@link #completeToolResult} extends the retained base branch into
   * a consecutive dependent tool selection.
   */
  public synchronized String selectTool(
      ModelPrompt renderedToolPrompt, SamplingOptions options, TokenConstraint constraint) {
    ActivatedToolTurn turn = toolTurn(renderedToolPrompt);
    String output = turn.generateToolCall(options, constraint);
    captureToolMetrics(turn);
    return output;
  }

  /** Streams a tool selection on the activated branch. */
  public synchronized void selectTool(
      ModelPrompt renderedToolPrompt,
      SamplingOptions options,
      TokenStream stream,
      TokenConstraint constraint) {
    ActivatedToolTurn turn = toolTurn(renderedToolPrompt);
    turn.generateToolCall(options, stream, constraint);
    captureToolMetrics(turn);
  }

  /** Synthesizes a tool result on the exact base branch and resumes the base cache lineage. */
  public synchronized String completeToolResult(ModelPrompt prompt, SamplingOptions options) {
    ActivatedToolTurn turn = requireActiveTurn();
    String output = turn.generateBaseResponse(Objects.requireNonNull(prompt, "prompt"), options);
    lastBaseMetrics = turn.responseMetrics();
    resumeBase(turn);
    return output;
  }

  /** Streams tool-result synthesis and resumes the exact base cache lineage. */
  public synchronized void completeToolResult(
      ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
    ActivatedToolTurn turn = requireActiveTurn();
    turn.generateBaseResponse(Objects.requireNonNull(prompt, "prompt"), options, stream);
    lastBaseMetrics = turn.responseMetrics();
    if (!turn.failed()) {
      resumeBase(turn);
    }
  }

  /** Returns metrics for the most recently completed activated tool selection. */
  public synchronized GenerationMetrics lastToolMetrics() {
    requireOpen();
    return lastToolMetrics;
  }

  /** Returns metrics for the most recently completed exact-base response. */
  public synchronized GenerationMetrics lastBaseMetrics() {
    requireOpen();
    return lastBaseMetrics;
  }

  /** Returns the immutable prefix-token count at the latest base/adapter fork. */
  public synchronized int sharedPrefixTokens() {
    requireOpen();
    return sharedPrefixTokens;
  }

  /** Returns the physical KV bytes shared at the latest base/adapter fork. */
  public synchronized long sharedPrefixBytes() {
    requireOpen();
    return sharedPrefixBytes;
  }

  /** Returns the runtime-verified physical-sharing result for the latest base/adapter fork. */
  public synchronized boolean physicallySharesPrefix() {
    requireOpen();
    return physicallySharedPrefix;
  }

  /** Returns whether a tool selection is awaiting another selection or a tool result. */
  public synchronized boolean hasActiveToolTurn() {
    requireOpen();
    return activeTurn != null;
  }

  @Override
  public synchronized void close() {
    if (!closed.compareAndSet(false, true)) {
      return;
    }
    if (activeTurn != null) {
      activeTurn.close();
      activeTurn = null;
    }
    if (baseSession != null) {
      baseSession.close();
      baseSession = null;
    }
  }

  private TextGenerationSession baseSession() {
    if (baseSession == null) {
      baseSession = pipeline.openGenerationSession();
    }
    return baseSession;
  }

  private ActivatedToolTurn toolTurn(ModelPrompt prompt) {
    requireOpen();
    Objects.requireNonNull(prompt, "renderedToolPrompt");
    if (activeTurn != null) {
      activeTurn = activeTurn.continueToolSelection(prompt);
      captureSharedPrefix(activeTurn);
      return activeTurn;
    }

    TextGenerationSession retained = baseSession;
    try {
      activeTurn = model.openAutomaticToolTurn(prompt, retained);
      baseSession = null;
      captureSharedPrefix(activeTurn);
      return activeTurn;
    } catch (RuntimeException | Error failure) {
      if (retained != null) {
        retained.close();
        baseSession = null;
      }
      throw failure;
    }
  }

  private void resumeBase(ActivatedToolTurn turn) {
    baseSession = turn.consumeBaseBranch();
    activeTurn = null;
    turn.close();
  }

  private void captureToolMetrics(ActivatedToolTurn turn) {
    lastToolMetrics = turn.toolMetrics();
    captureSharedPrefix(turn);
  }

  private void captureSharedPrefix(ActivatedToolTurn turn) {
    sharedPrefixTokens = turn.sharedPrefixTokens();
    sharedPrefixBytes = turn.sharedPrefixBytes();
    physicallySharedPrefix = turn.physicallySharesPrefix();
  }

  private ActivatedToolTurn requireActiveTurn() {
    requireOpen();
    if (activeTurn == null) {
      throw new IllegalStateException("select a tool before completing its result");
    }
    return activeTurn;
  }

  private void requireBaseAvailable() {
    requireOpen();
    if (activeTurn != null) {
      throw new IllegalStateException("complete the active tool turn before base generation");
    }
  }

  private void requireOpen() {
    if (closed.get()) {
      throw new IllegalStateException("activated tool conversation is closed");
    }
  }
}
