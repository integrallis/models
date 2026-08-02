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
package com.integrallis.models.backend.purejava.gemma4;

import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.io.IOException;
import java.util.Objects;

/** Owns one loaded Gemma 4 decoder and its zero-copy mapped expert views. */
public final class Gemma4Decoder implements AutoCloseable {

  /** Independent sequence state that shares this decoder's immutable weights. */
  public static final class Session {
    private final Gemma4Decoder owner;
    private final Gemma4ForwardPass.Session delegate;

    private Session(Gemma4Decoder owner, Gemma4ForwardPass.Session delegate) {
      this.owner = owner;
      this.delegate = delegate;
    }

    /** Returns this sequence's next absolute token position. */
    public int checkpoint() {
      owner.checkOpen();
      return delegate.checkpoint();
    }
  }

  private final Gemma4Config config;
  private final Gemma4Experts experts;
  private final Gemma4ForwardPass forwardPass;
  private Gemma4ForwardPass.Session[] sessionBatch = new Gemma4ForwardPass.Session[0];
  private boolean closed;

  private Gemma4Decoder(Gemma4Config config, Gemma4Experts experts, Gemma4ForwardPass forwardPass) {
    this.config = config;
    this.experts = experts;
    this.forwardPass = forwardPass;
  }

  /** Opens an exact text-only Gemma 4 GGUF decoder. */
  public static Gemma4Decoder load(
      GgufFile file, int runtimeContextLength, GgufBatchedMatrixKernel batchedMatrixKernel)
      throws IOException {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel");
    Gemma4Config config = Gemma4Config.fromMetadata(file.metadata());
    Gemma4Weights weights = Gemma4Weights.fromGgufFile(file, config);
    Gemma4TensorLayout layout = weights.expertLayout();
    Gemma4Experts experts =
        new Gemma4MappedExperts(
            file.fileSegment(),
            config.numLayers(),
            config.numExperts(),
            (layer, expert) -> layout.layer(layer).expert(expert));
    try {
      Gemma4ForwardPass forwardPass =
          new Gemma4ForwardPass(
              config,
              weights,
              Gemma4KvCache.create(config, runtimeContextLength, 1),
              experts,
              batchedMatrixKernel);
      return new Gemma4Decoder(config, experts, forwardPass);
    } catch (RuntimeException | Error failure) {
      closeAfterFailure(experts, failure);
      throw failure;
    }
  }

  /** Returns the parsed immutable model shape. */
  public Gemma4Config config() {
    checkOpen();
    return config;
  }

  /** Returns the maximum independent-session batch accepted by this decoder. */
  public int maxBatchSize() {
    checkOpen();
    return forwardPass.maxSessionBatchSize();
  }

  /** Returns the prompt batch size selected for the loaded tensor topology and kernel. */
  public int prefillBatchSize() {
    checkOpen();
    return forwardPass.prefillBatchSize();
  }

  /** Executes one default-sequence token and returns stable logits. */
  public float[] forward(int token, int position) {
    checkOpen();
    return forwardPass.forward(token, position);
  }

  /** Executes one default-sequence token using reusable logits storage. */
  public float[] forwardTransient(int token, int position) {
    checkOpen();
    return forwardPass.forwardTransient(token, position);
  }

  /** Evaluates a contiguous prompt on the default sequence. */
  public float[] prefill(int[] tokens, int startPosition) {
    checkOpen();
    return forwardPass.prefill(tokens, startPosition);
  }

  /** Opens independent KV state that shares loaded weights and routed experts. */
  public Session openSession() {
    checkOpen();
    return new Session(this, forwardPass.openSession());
  }

  /** Executes one independent-session token and returns stable logits. */
  public float[] forward(Session session, int token, int position) {
    return forwardPass.forward(requireSession(session), token, position);
  }

  /** Executes one independent-session token using reusable logits storage. */
  public float[] forwardTransient(Session session, int token, int position) {
    return forwardPass.forwardTransient(requireSession(session), token, position);
  }

  /** Evaluates a contiguous prompt on one independent sequence. */
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    return forwardPass.prefill(requireSession(session), tokens, startPosition);
  }

  /** Executes one token per independent session and returns stable logits. */
  public LogitBatch forwardBatch(Session[] sessions, int[] tokens) {
    return forwardPass.forwardBatch(unwrapSessions(sessions), tokens);
  }

  /** Executes one token per independent session using reusable batch storage. */
  public LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens) {
    return forwardPass.forwardBatchTransient(unwrapSessions(sessions), tokens);
  }

  /** Discards one session's state at and after a prior checkpoint. */
  public void rewind(Session session, int checkpoint) {
    forwardPass.rewind(requireSession(session), checkpoint);
  }

  /** Clears one independent sequence. */
  public void reset(Session session) {
    forwardPass.reset(requireSession(session));
  }

  /** Returns the default sequence's next absolute token position. */
  public int checkpoint() {
    checkOpen();
    return forwardPass.checkpoint();
  }

  /** Verifies a speculative continuation and returns stable logits for every token. */
  public LogitBatch verify(int[] tokens, int startPosition) {
    checkOpen();
    return forwardPass.verify(tokens, startPosition);
  }

  /** Verifies a speculative continuation using reusable batch storage. */
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    checkOpen();
    return forwardPass.verifyTransient(tokens, startPosition);
  }

  /** Discards default-sequence state at and after a prior checkpoint. */
  public void rewind(int checkpoint) {
    checkOpen();
    forwardPass.rewind(checkpoint);
  }

  /** Clears the default sequence while retaining allocated model resources. */
  public void reset() {
    checkOpen();
    forwardPass.reset();
  }

  @Override
  public void close() throws IOException {
    if (!closed) {
      closed = true;
      experts.close();
    }
  }

  private Gemma4ForwardPass.Session[] unwrapSessions(Session[] sessions) {
    checkOpen();
    Objects.requireNonNull(sessions, "sessions");
    if (sessionBatch.length != sessions.length) {
      sessionBatch = new Gemma4ForwardPass.Session[sessions.length];
    }
    for (int index = 0; index < sessions.length; index++) {
      sessionBatch[index] = requireSession(sessions[index]);
    }
    return sessionBatch;
  }

  private Gemma4ForwardPass.Session requireSession(Session session) {
    checkOpen();
    Objects.requireNonNull(session, "session");
    if (session.owner != this) {
      throw new IllegalArgumentException("session belongs to a different Gemma 4 decoder");
    }
    return session.delegate;
  }

  private void checkOpen() {
    if (closed) {
      throw new IllegalStateException("Gemma 4 decoder is closed");
    }
  }

  private static void closeAfterFailure(Gemma4Experts experts, Throwable failure) {
    try {
      experts.close();
    } catch (IOException | RuntimeException closeFailure) {
      failure.addSuppressed(closeFailure);
    }
  }
}
