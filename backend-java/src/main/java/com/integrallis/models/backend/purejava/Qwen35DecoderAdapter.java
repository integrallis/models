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
package com.integrallis.models.backend.purejava;

import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.backend.purejava.qwen35.Qwen35ForwardPass;
import java.util.Objects;

/** Adapts Qwen3.5 hybrid recurrent/attention state to the backend contract. */
final class Qwen35DecoderAdapter implements PureJavaDecoder {

  private final class Qwen35Session implements Session {
    private final Qwen35ForwardPass.Session delegate;

    private Qwen35Session() {
      this(forwardPass.openSession(capacity));
    }

    private Qwen35Session(Qwen35ForwardPass.Session delegate) {
      this.delegate = delegate;
    }

    @Override
    public int checkpoint() {
      return delegate.checkpoint();
    }
  }

  private record Qwen35Prefix(Qwen35ForwardPass.SessionPrefix delegate) implements SharedPrefix {
    @Override
    public int checkpoint() {
      return delegate.checkpoint();
    }

    @Override
    public long sharedBytes() {
      return delegate.sharedBytes();
    }
  }

  private final Qwen35ForwardPass forwardPass;
  private final int capacity;
  private final Qwen35Session defaultSession;

  Qwen35DecoderAdapter(Qwen35ForwardPass forwardPass, int capacity) {
    this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
    this.capacity = capacity;
    this.defaultSession = new Qwen35Session();
  }

  @Override
  public int maxBatchSize() {
    return 1;
  }

  @Override
  public float[] forward(int token, int position) {
    return forwardPass.forward(defaultSession.delegate, token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    return forward(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return forwardPass.prefill(defaultSession.delegate, tokens, startPosition);
  }

  @Override
  public Session openSession() {
    return new Qwen35Session();
  }

  @Override
  public boolean supportsSharedPrefixes() {
    return forwardPass.supportsSharedPrefixes();
  }

  @Override
  public com.integrallis.models.api.PortablePrefixCapabilities.Truncation
      portablePrefixTruncation() {
    return supportsSharedPrefixes()
        ? com.integrallis.models.api.PortablePrefixCapabilities.Truncation.RETAINED_CHECKPOINTS
        : com.integrallis.models.api.PortablePrefixCapabilities.Truncation.UNSUPPORTED;
  }

  @Override
  public String portablePrefixConfiguration() {
    return forwardPass.portablePrefixConfiguration();
  }

  @Override
  public Class<?>[] portablePrefixImplementations() {
    return Qwen35ForwardPass.portablePrefixImplementations();
  }

  @Override
  public long portablePrefixBytes(int count) {
    if (count > capacity) throw new IllegalArgumentException("prefix exceeds session capacity");
    return forwardPass.portablePrefixBytes(count);
  }

  @Override
  public SharedPrefix freezePrefix(Session source) {
    return new Qwen35Prefix(forwardPass.freezePrefix(requireSession(source).delegate));
  }

  @Override
  public SharedPrefix truncatePrefix(SharedPrefix prefix, int count) {
    return new Qwen35Prefix(requirePrefix(prefix).truncate(count));
  }

  @Override
  public void validatePrefixTokens(SharedPrefix prefix, int[] tokens) throws java.io.IOException {
    requirePrefix(prefix).validateTokens(tokens);
  }

  @Override
  public void writePrefix(SharedPrefix prefix, int count, java.io.DataOutput output)
      throws java.io.IOException {
    requirePrefix(prefix).writeTo(output, count);
  }

  @Override
  public SharedPrefix readPrefix(java.io.DataInput input, int count, long maxBytes)
      throws java.io.IOException {
    return new Qwen35Prefix(forwardPass.readPrefix(input, count, maxBytes, capacity));
  }

  @Override
  public Session fork(SharedPrefix prefix, boolean activated) {
    if (activated) throw new UnsupportedOperationException("hybrid activated state is unavailable");
    return new Qwen35Session(forwardPass.fork(requirePrefix(prefix)));
  }

  @Override
  public boolean sharesPrefixStorage(Session first, Session second) {
    return forwardPass.sharesPrefixStorage(
        requireSession(first).delegate, requireSession(second).delegate);
  }

  private Qwen35ForwardPass.SessionPrefix requirePrefix(SharedPrefix prefix) {
    if (!(prefix instanceof Qwen35Prefix qwen))
      throw new IllegalArgumentException("prefix belongs to a different decoder");
    return qwen.delegate();
  }

  @Override
  public float[] forward(Session session, int token, int position) {
    return forwardPass.forward(requireSession(session).delegate, token, position);
  }

  @Override
  public float[] forwardTransient(Session session, int token, int position) {
    return forward(session, token, position);
  }

  @Override
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    return forwardPass.prefill(requireSession(session).delegate, tokens, startPosition);
  }

  @Override
  public LogitBatch forwardBatch(Session[] sessions, int[] tokens) {
    return forwardBatchTransient(sessions, tokens).snapshot();
  }

  @Override
  public LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens) {
    Objects.requireNonNull(sessions, "sessions");
    Objects.requireNonNull(tokens, "tokens");
    if (sessions.length != 1 || tokens.length != 1) {
      throw new IllegalArgumentException(
          "Qwen3.5 compatibility execution currently supports one session per batch");
    }
    Qwen35Session session = requireSession(sessions[0]);
    float[] logits =
        forwardPass.forward(session.delegate, tokens[0], session.delegate.checkpoint());
    return new LogitBatch(1, forwardPass.config().vocabSize(), logits);
  }

  @Override
  public void rewind(Session session, int checkpoint) {
    forwardPass.rewind(requireSession(session).delegate, checkpoint);
  }

  @Override
  public void reset(Session session) {
    forwardPass.reset(requireSession(session).delegate);
  }

  @Override
  public int checkpoint() {
    return defaultSession.checkpoint();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    return verifyTransient(tokens, startPosition).snapshot();
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition != checkpoint()) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + checkpoint() + ", got " + startPosition);
    }
    int vocabularySize = forwardPass.config().vocabSize();
    float[] rows = new float[Math.multiplyExact(tokens.length, vocabularySize)];
    for (int index = 0; index < tokens.length; index++) {
      float[] logits = forward(tokens[index], startPosition + index);
      System.arraycopy(logits, 0, rows, index * vocabularySize, vocabularySize);
    }
    return new LogitBatch(tokens.length, vocabularySize, rows);
  }

  @Override
  public void rewind(int checkpoint) {
    forwardPass.rewind(defaultSession.delegate, checkpoint);
  }

  @Override
  public void reset() {
    forwardPass.reset(defaultSession.delegate);
  }

  @Override
  public boolean supportsResumption() {
    return true;
  }

  @Override
  public boolean supportsGroupedDecisions() {
    return true;
  }

  @Override
  public int maximumGroupSize() {
    return forwardPass.prefillBatchSize();
  }

  @Override
  public float[][] decideGrouped(int[][] suffixes) {
    Objects.requireNonNull(suffixes, "suffixes");
    return forwardPass.decideGrouped(defaultSession.delegate, suffixes);
  }

  @Override
  public Object captureResumption() {
    return forwardPass.captureLinearState(defaultSession.delegate);
  }

  @Override
  public void resume(Object point) {
    Objects.requireNonNull(point, "point");
    if (!(point instanceof Qwen35ForwardPass.LinearStateSnapshot snapshot)) {
      throw new IllegalArgumentException("resumption was not issued by this decoder");
    }
    forwardPass.restoreLinearState(defaultSession.delegate, snapshot);
  }

  @Override
  public void close() {}

  private Qwen35Session requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (!(session instanceof Qwen35DecoderAdapter.Qwen35Session qwen35Session)) {
      throw new IllegalArgumentException("session belongs to a different decoder");
    }
    return qwen35Session;
  }
}
