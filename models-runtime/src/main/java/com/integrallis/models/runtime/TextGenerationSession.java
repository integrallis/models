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
import com.integrallis.models.api.InferenceContextWindow;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One explicit high-level generation lineage over a loaded model.
 *
 * <p>Each session owns independent backend state and its own exact prompt-prefix history. Multiple
 * sessions share the loaded model weights, while execution is serialized on the owning pipeline so
 * transient logits and backend scratch cannot race. Closing a session releases only its
 * request-specific state; closing the owning pipeline closes every remaining session and the model.
 */
public final class TextGenerationSession implements ConstrainedTextGenerationModel, AutoCloseable {
  private final Object executionLock;
  private final SessionBackend backend;
  private final GenerationLoop generationLoop;
  private final Runnable closeListener;
  private final AtomicBoolean closed = new AtomicBoolean();

  TextGenerationSession(
      BatchInferenceBackend backend,
      InferenceSession session,
      Object executionLock,
      Runnable closeListener) {
    this.executionLock = Objects.requireNonNull(executionLock, "executionLock");
    this.backend = new SessionBackend(backend, session);
    this.generationLoop =
        new GenerationLoop(
            this.backend, SpeculativeGenerationOptions.disabled(), System::nanoTime, executionLock);
    this.closeListener = Objects.requireNonNull(closeListener, "closeListener");
  }

  /** Returns the active context capacity and next token position for this session. */
  public InferenceContextWindow contextWindow() {
    synchronized (executionLock) {
      requireOpen();
      return new InferenceContextWindow(
          backend.contextCapacity(), java.util.OptionalInt.of(backend.checkpoint()));
    }
  }

  /** Returns phase, token-usage, and prompt-cache measurements for this session's latest call. */
  public GenerationMetrics lastGenerationMetrics() {
    synchronized (executionLock) {
      requireOpen();
      return generationLoop.lastGenerationMetrics();
    }
  }

  /** Clears this session's context and exact prompt-prefix history. */
  public void resetContext() {
    synchronized (executionLock) {
      requireOpen();
      generationLoop.invalidatePromptCache();
      backend.reset();
    }
  }

  /** Returns whether this session has released its backend-owned state. */
  public boolean isClosed() {
    return closed.get();
  }

  @Override
  public String modelName() {
    synchronized (executionLock) {
      requireOpen();
      return backend.metadata().modelName();
    }
  }

  @Override
  public BackendDiagnostics diagnostics() {
    synchronized (executionLock) {
      requireOpen();
      return backend.diagnostics();
    }
  }

  @Override
  public Tokenizer tokenizer() {
    synchronized (executionLock) {
      requireOpen();
      return backend.tokenizer();
    }
  }

  @Override
  public String generate(String prompt, SamplingOptions options) {
    synchronized (executionLock) {
      requireOpen();
      return generationLoop.generate(prompt, options);
    }
  }

  @Override
  public String generate(ModelPrompt prompt, SamplingOptions options) {
    synchronized (executionLock) {
      requireOpen();
      return generationLoop.generate(prompt, options);
    }
  }

  @Override
  public void generate(String prompt, SamplingOptions options, TokenStream stream) {
    synchronized (executionLock) {
      requireOpen();
      generationLoop.generate(prompt, options, stream);
    }
  }

  @Override
  public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
    synchronized (executionLock) {
      requireOpen();
      generationLoop.generate(prompt, options, stream);
    }
  }

  @Override
  public void generate(
      ModelPrompt prompt, SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
    synchronized (executionLock) {
      requireOpen();
      generationLoop.generate(prompt, options, stream, constraint);
    }
  }

  @Override
  public void close() {
    synchronized (executionLock) {
      if (closed.compareAndSet(false, true)) {
        generationLoop.invalidatePromptCache();
        backend.close();
        closeListener.run();
      }
    }
  }

  private void requireOpen() {
    if (closed.get()) {
      throw new IllegalStateException("text generation session is closed");
    }
  }

  private static final class SessionBackend implements RewindableInferenceBackend {
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
  }
}
