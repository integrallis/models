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

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.HiddenStateInferenceBackend;
import com.integrallis.models.api.InferenceContextWindow;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One explicit high-level generation lineage over a loaded model.
 *
 * <p>Each session owns independent backend state and its own exact prompt-prefix history. Multiple
 * sessions share the loaded model weights. The ordinary pipeline serializes their execution;
 * pipelines configured with {@link ContinuousBatchingOptions} may advance compatible sessions in
 * one physical-model call. Closing a session releases only its request-specific state; closing the
 * owning pipeline closes every remaining session and the model.
 */
public final class TextGenerationSession implements ConstrainedTextGenerationModel, AutoCloseable {
  private final Object executionLock;
  private final Object operationLock = new Object();
  private final SessionBackend backend;
  private final GenerationLoop generationLoop;
  private final ContinuousBatchingScheduler continuousBatching;
  private final ContinuousBatchingScheduler.SessionState continuousBatchingState;
  private final int[] preparedPrefixTokens;
  private final Runnable closeListener;
  private final AtomicBoolean closed = new AtomicBoolean();

  TextGenerationSession(
      BatchInferenceBackend backend,
      InferenceSession session,
      Object executionLock,
      Runnable closeListener) {
    this(backend, session, executionLock, closeListener, null);
  }

  TextGenerationSession(
      BatchInferenceBackend backend,
      InferenceSession session,
      Object executionLock,
      Runnable closeListener,
      ContinuousBatchingScheduler continuousBatching) {
    this(backend, session, executionLock, closeListener, continuousBatching, null);
  }

  TextGenerationSession(
      BatchInferenceBackend backend,
      InferenceSession session,
      Object executionLock,
      Runnable closeListener,
      ContinuousBatchingScheduler continuousBatching,
      int[] preparedPromptTokens) {
    this.executionLock = Objects.requireNonNull(executionLock, "executionLock");
    this.backend = new SessionBackend(backend, session);
    this.generationLoop =
        new GenerationLoop(
            this.backend, SpeculativeGenerationOptions.disabled(), System::nanoTime, executionLock);
    this.closeListener = Objects.requireNonNull(closeListener, "closeListener");
    this.continuousBatching = continuousBatching;
    this.continuousBatchingState =
        continuousBatching == null ? null : continuousBatching.session(session);
    this.preparedPrefixTokens = preparedPromptTokens == null ? null : preparedPromptTokens.clone();
    if (preparedPromptTokens != null) {
      generationLoop.restorePromptCache(preparedPromptTokens);
      if (continuousBatchingState != null) {
        continuousBatchingState.restorePromptCache(preparedPromptTokens);
      }
    }
  }

  /** Returns the active context capacity and next token position for this session. */
  public InferenceContextWindow contextWindow() {
    synchronized (operationLock) {
      requireOpen();
      synchronized (executionLock) {
        return new InferenceContextWindow(
            backend.contextCapacity(), java.util.OptionalInt.of(backend.checkpoint()));
      }
    }
  }

  /** Returns phase, token-usage, and prompt-cache measurements for this session's latest call. */
  public GenerationMetrics lastGenerationMetrics() {
    synchronized (operationLock) {
      requireOpen();
      return continuousBatching == null
          ? generationLoop.lastGenerationMetrics()
          : continuousBatchingState.lastGenerationMetrics();
    }
  }

  /** Returns the physical bytes allocated for this session's backend state, when measurable. */
  public OptionalLong allocatedInferenceStateBytes() {
    synchronized (operationLock) {
      requireOpen();
      synchronized (executionLock) {
        return backend.allocatedStateBytes();
      }
    }
  }

  /**
   * Prefills this session's own prompt/KV lineage without decoding output tokens.
   *
   * <p>Use this to catch up an inactive routed model before a possible handoff. A prepared prefix
   * belongs only to this model and session; it cannot be transferred to another model.
   */
  public PromptPrefillMetrics prefillPrompt(ModelPrompt prompt) {
    synchronized (operationLock) {
      requireOpen();
      if (continuousBatching != null) {
        return continuousBatching.prefill(continuousBatchingState, prompt);
      }
      synchronized (executionLock) {
        return generationLoop.prefillPrompt(prompt);
      }
    }
  }

  /** Prefills trusted token IDs while preserving this session's exact cache lineage. */
  PromptPrefillMetrics prefillTokenPrefix(int[] promptTokens) {
    Objects.requireNonNull(promptTokens, "promptTokens");
    synchronized (operationLock) {
      requireOpen();
      if (continuousBatching != null) {
        throw new UnsupportedOperationException(
            "prefilling trusted token IDs is not supported by continuous batching");
      }
      synchronized (executionLock) {
        return generationLoop.prefillTokens(promptTokens);
      }
    }
  }

  float[] nextTokenLogits(ModelPrompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    synchronized (operationLock) {
      requireOpen();
      if (continuousBatching != null) {
        throw new UnsupportedOperationException(
            "next-token scoring is not supported by continuous batching");
      }
      synchronized (executionLock) {
        return generationLoop.nextTokenLogits(prompt);
      }
    }
  }

  float[] nextTokenHiddenState(ModelPrompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    synchronized (operationLock) {
      requireOpen();
      if (continuousBatching != null) {
        throw new UnsupportedOperationException(
            "next-token hidden-state scoring is not supported by continuous batching");
      }
      synchronized (executionLock) {
        return generationLoop.nextTokenHiddenState(prompt);
      }
    }
  }

  /**
   * Clears this session's mutable context and exact prompt history.
   *
   * <p>A session forked from an immutable physical prefix returns to that prepared baseline; an
   * ordinary session returns to an empty context.
   */
  public void resetContext() {
    synchronized (operationLock) {
      requireOpen();
      synchronized (executionLock) {
        generationLoop.invalidatePromptCache();
        if (continuousBatchingState != null) {
          continuousBatchingState.invalidatePromptCache();
        }
        if (preparedPrefixTokens == null) {
          backend.reset();
        } else {
          backend.rewind(preparedPrefixTokens.length);
        }
        if (preparedPrefixTokens != null) {
          generationLoop.restorePromptCache(preparedPrefixTokens);
          if (continuousBatchingState != null) {
            continuousBatchingState.restorePromptCache(preparedPrefixTokens);
          }
        }
      }
    }
  }

  /** Returns whether this session has released its backend-owned state. */
  public boolean isClosed() {
    return closed.get();
  }

  /**
   * Returns whether this session and another reference the same physical immutable KV prefix.
   *
   * <p>False means no shared prefix; it never means that equivalent KV state was copied.
   */
  public boolean sharesPrefixStorageWith(TextGenerationSession other) {
    Objects.requireNonNull(other, "other");
    requireOpen();
    other.requireOpen();
    if (executionLock != other.executionLock) {
      return false;
    }
    synchronized (executionLock) {
      return backend.sharesPrefixStorageWith(other.backend);
    }
  }

  SharedInferencePrefix extendAndFreezeSharedPrefix(
      SharedPrefixInferenceBackend owner, int[] promptTokens) {
    Objects.requireNonNull(owner, "owner");
    Objects.requireNonNull(promptTokens, "promptTokens");
    synchronized (operationLock) {
      requireOpen();
      if (continuousBatching != null) {
        throw new UnsupportedOperationException(
            "extending a physical shared prefix is not supported by continuous batching");
      }
      synchronized (executionLock) {
        generationLoop.requireStrictPromptExtension(promptTokens);
        generationLoop.prefillTokens(promptTokens);
        SharedInferencePrefix prefix = backend.freezeSharedPrefix(owner);
        if (!closed.compareAndSet(false, true)) {
          throw new IllegalStateException("text generation session closed while freezing prefix");
        }
        generationLoop.invalidatePromptCache();
        closeListener.run();
        return prefix;
      }
    }
  }

  SharedInferencePrefix reconcileAndFreezeSharedPrefix(
      SharedPrefixInferenceBackend owner, int[] promptTokens) {
    Objects.requireNonNull(owner, "owner");
    Objects.requireNonNull(promptTokens, "promptTokens");
    synchronized (operationLock) {
      requireOpen();
      if (continuousBatching != null) {
        throw new UnsupportedOperationException(
            "reconciling a physical shared prefix is not supported by continuous batching");
      }
      if (preparedPrefixTokens == null) {
        throw new IllegalStateException(
            "only a session forked from an immutable prefix can reconcile its prompt");
      }
      requireRetainedPreparedPrefix(promptTokens);
      synchronized (executionLock) {
        generationLoop.prefillTokens(promptTokens);
        SharedInferencePrefix prefix = backend.freezeSharedPrefix(owner);
        if (!closed.compareAndSet(false, true)) {
          throw new IllegalStateException("text generation session closed while freezing prefix");
        }
        generationLoop.invalidatePromptCache();
        closeListener.run();
        return prefix;
      }
    }
  }

  @Override
  public String modelName() {
    synchronized (operationLock) {
      requireOpen();
      synchronized (executionLock) {
        return backend.metadata().modelName();
      }
    }
  }

  @Override
  public BackendDiagnostics diagnostics() {
    synchronized (operationLock) {
      requireOpen();
      synchronized (executionLock) {
        return backend.diagnostics();
      }
    }
  }

  @Override
  public Tokenizer tokenizer() {
    synchronized (operationLock) {
      requireOpen();
      return backend.tokenizer();
    }
  }

  @Override
  public String generate(String prompt, SamplingOptions options) {
    synchronized (operationLock) {
      requireOpen();
      return continuousBatching == null
          ? generateSequentially(
              ModelPrompt.text(Objects.requireNonNull(prompt, "prompt")), options)
          : ConstrainedTextGenerationModel.super.generate(
              ModelPrompt.text(Objects.requireNonNull(prompt, "prompt")),
              options,
              TokenConstraint.unrestricted());
    }
  }

  @Override
  public String generate(ModelPrompt prompt, SamplingOptions options) {
    synchronized (operationLock) {
      requireOpen();
      return continuousBatching == null
          ? generateSequentially(prompt, options)
          : ConstrainedTextGenerationModel.super.generate(
              prompt, options, TokenConstraint.unrestricted());
    }
  }

  @Override
  public void generate(String prompt, SamplingOptions options, TokenStream stream) {
    generate(ModelPrompt.text(Objects.requireNonNull(prompt, "prompt")), options, stream);
  }

  @Override
  public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
    synchronized (operationLock) {
      requireOpen();
      generateScheduledOrSequential(prompt, options, stream, TokenConstraint.unrestricted());
    }
  }

  @Override
  public void generate(
      ModelPrompt prompt, SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
    synchronized (operationLock) {
      requireOpen();
      generateScheduledOrSequential(prompt, options, stream, constraint);
    }
  }

  @Override
  public void close() {
    synchronized (operationLock) {
      synchronized (executionLock) {
        if (closed.compareAndSet(false, true)) {
          generationLoop.invalidatePromptCache();
          if (continuousBatchingState != null) {
            continuousBatchingState.invalidatePromptCache();
          }
          backend.close();
          closeListener.run();
        }
      }
    }
  }

  private String generateSequentially(ModelPrompt prompt, SamplingOptions options) {
    synchronized (executionLock) {
      return generationLoop.generate(prompt, options);
    }
  }

  private void generateScheduledOrSequential(
      ModelPrompt prompt, SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
    if (continuousBatching == null) {
      synchronized (executionLock) {
        generationLoop.generate(prompt, options, stream, constraint);
      }
      return;
    }
    continuousBatching.generate(continuousBatchingState, prompt, options, stream, constraint);
  }

  private void requireOpen() {
    if (closed.get()) {
      throw new IllegalStateException("text generation session is closed");
    }
  }

  private void requireRetainedPreparedPrefix(int[] promptTokens) {
    if (promptTokens.length <= preparedPrefixTokens.length) {
      throw new IllegalArgumentException(
          "reconciled prompt must retain the immutable shared prefix and add a suffix");
    }
    for (int index = 0; index < preparedPrefixTokens.length; index++) {
      if (promptTokens[index] != preparedPrefixTokens[index]) {
        throw new IllegalArgumentException(
            "reconciled prompt must retain the immutable shared prefix and add a suffix");
      }
    }
  }

  private static final class SessionBackend
      implements RewindableInferenceBackend, HiddenStateInferenceBackend {
    private final BatchInferenceBackend backend;
    private final InferenceSession session;

    private SessionBackend(BatchInferenceBackend backend, InferenceSession session) {
      this.backend = Objects.requireNonNull(backend, "backend");
      this.session = Objects.requireNonNull(session, "session");
    }

    @Override
    public String name() {
      return backend.name();
    }

    @Override
    public ModelMetadata metadata() {
      return backend.metadata();
    }

    @Override
    public int contextCapacity() {
      return backend.contextCapacity();
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return backend.diagnostics();
    }

    @Override
    public Tokenizer tokenizer() {
      return backend.tokenizer();
    }

    @Override
    public float[] forward(int token, int position) {
      return backend.forward(session, token, position);
    }

    @Override
    public float[] forwardTransient(int token, int position) {
      return backend.forwardTransient(session, token, position);
    }

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      return backend.prefill(session, tokens, startPosition);
    }

    @Override
    public boolean supportsHiddenState() {
      return backend.supportsHiddenState();
    }

    @Override
    public float[] forwardHiddenState(int token, int position) {
      return backend.forwardHiddenState(session, token, position);
    }

    @Override
    public float[] forwardHiddenStateTransient(int token, int position) {
      return backend.forwardHiddenStateTransient(session, token, position);
    }

    @Override
    public float[] prefillHiddenState(int[] tokens, int startPosition) {
      return backend.prefillHiddenState(session, tokens, startPosition);
    }

    @Override
    public void reset() {
      backend.reset(session);
    }

    @Override
    public int checkpoint() {
      return session.checkpoint();
    }

    @Override
    public void rewind(int checkpoint) {
      backend.rewind(session, checkpoint);
    }

    @Override
    public void close() {
      session.close();
    }

    private boolean sharesPrefixStorageWith(SessionBackend other) {
      if (backend != other.backend || !(backend instanceof SharedPrefixInferenceBackend sharing)) {
        return false;
      }
      return sharing.sharesPrefixStorage(session, other.session);
    }

    private SharedInferencePrefix freezeSharedPrefix(SharedPrefixInferenceBackend owner) {
      if (backend != owner) {
        throw new IllegalArgumentException("shared-prefix backend does not own this session");
      }
      return owner.freezePrefix(session);
    }

    private OptionalLong allocatedStateBytes() {
      return session.allocatedStateBytes();
    }
  }
}
