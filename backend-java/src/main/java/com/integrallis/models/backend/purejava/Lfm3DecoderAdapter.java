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
import com.integrallis.models.backend.purejava.lfm3.Lfm3ForwardPass;
import java.util.Arrays;
import java.util.Objects;

/** Owns one LFM3 session; checkpoints can be reconstructed by replaying retained tokens. */
final class Lfm3DecoderAdapter implements PureJavaDecoder {

  private final class Lfm3Session implements Session {
    @Override
    public int checkpoint() {
      return forwardPass.nextPosition();
    }
  }

  private final Lfm3ForwardPass forwardPass;
  private final Lfm3Session defaultSession;
  private boolean sessionHandedOut;

  /**
   * Every token this decoder has been advanced through, in position order.
   *
   * <p>Positions are strictly sequential -- the forward pass refuses anything else -- so index
   * {@code i} is the token at position {@code i} and the first {@code n} entries reconstruct the
   * state at checkpoint {@code n} exactly. Entries at or beyond the current position are stale and
   * unreachable: {@link #rewind} never replays past it.
   *
   * <p>Written by {@link #forward(int, int)}, which every path that advances the graph goes
   * through. A path that calls the forward pass directly would advance the state without recording
   * it, and the next rewind would then rebuild a state missing those tokens.
   */
  private int[] tokenHistory = new int[0];

  Lfm3DecoderAdapter(Lfm3ForwardPass forwardPass) {
    this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
    this.defaultSession = new Lfm3Session();
  }

  @Override
  public int maxBatchSize() {
    return 1;
  }

  @Override
  public float[] hiddenState(int token, int position) {
    float[] result = forwardPass.hiddenState(token, position);
    record(token, position);
    return result;
  }

  @Override
  public float[] prefillHiddenState(int[] tokens, int startPosition) {
    float[] result = forwardPass.prefillHiddenState(tokens, startPosition);
    for (int i = 0; i < tokens.length; i++) record(tokens[i], startPosition + i);
    return result;
  }

  @Override
  public float[] hiddenState(Session session, int token, int position) {
    requireDefault(session);
    return hiddenState(token, position);
  }

  @Override
  public float[] prefillHiddenState(Session session, int[] tokens, int startPosition) {
    requireDefault(session);
    return prefillHiddenState(tokens, startPosition);
  }

  @Override
  public boolean supportsHiddenState() {
    return true;
  }

  @Override
  public float[] forward(int token, int position) {
    float[] logits = forwardPass.forward(token, position);
    // After the call, so a rejected position does not leave a token recorded for it.
    record(token, position);
    return logits;
  }

  private void record(int token, int position) {
    if (position >= tokenHistory.length) {
      // Grow geometrically: a prompt is replayed token by token, so this is hit once per token.
      tokenHistory =
          Arrays.copyOf(
              tokenHistory, Math.max(16, Math.max(position + 1, 2 * tokenHistory.length)));
    }
    tokenHistory[position] = token;
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    return forward(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    float[] logits = forwardPass.prefill(tokens, startPosition);
    for (int i = 0; i < tokens.length; i++) record(tokens[i], startPosition + i);
    return logits;
  }

  @Override
  public Session openSession() {
    if (sessionHandedOut) {
      throw new UnsupportedOperationException(
          "LFM3 holds one recurrent state per decoder, so it supports a single session");
    }
    sessionHandedOut = true;
    return defaultSession;
  }

  @Override
  public float[] forward(Session session, int token, int position) {
    requireDefault(session);
    return forward(token, position);
  }

  @Override
  public float[] forwardTransient(Session session, int token, int position) {
    requireDefault(session);
    return forward(token, position);
  }

  @Override
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    requireDefault(session);
    return prefill(tokens, startPosition);
  }

  @Override
  public LogitBatch forwardBatch(Session[] sessions, int[] tokens) {
    if (sessions.length != 1 || tokens.length != 1) {
      throw new UnsupportedOperationException(
          "LFM3 decodes one sequence at a time; batched sessions are not supported");
    }
    requireDefault(sessions[0]);
    float[] logits = forward(tokens[0], forwardPass.nextPosition());
    return new LogitBatch(1, logits.length, logits.clone());
  }

  @Override
  public LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens) {
    return forwardBatch(sessions, tokens);
  }

  @Override
  public void rewind(Session session, int checkpoint) {
    requireDefault(session);
    int position = forwardPass.nextPosition();
    if (checkpoint == position) {
      return;
    }
    if (checkpoint < 0 || checkpoint > position) {
      // Forward is not a rewind: there are no tokens recorded past the current position, and
      // inventing them is the failure this whole mechanism exists to avoid.
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + position + ": " + checkpoint);
    }
    forwardPass.reset();
    if (checkpoint > 0) forwardPass.prefill(Arrays.copyOf(tokenHistory, checkpoint), 0);
  }

  @Override
  public void reset(Session session) {
    requireDefault(session);
    reset();
  }

  @Override
  public int checkpoint() {
    return forwardPass.nextPosition();
  }

  /**
   * Refused: speculative verification needs to discard the tokens it tried.
   *
   * <p>{@link #rewind} can now undo them by replay, but replaying the whole prefix after every
   * speculative batch costs more than the speculation saves, so this stays a refusal rather than
   * becoming a silent pessimisation. Speculation is an optimisation a caller opts into; rewinding
   * is not.
   */
  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    throw new UnsupportedOperationException(
        "LFM3 speculative verification is unsupported: rewind requires prefix replay");
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    return verify(tokens, startPosition);
  }

  @Override
  public void reset() {
    forwardPass.reset();
    // Hygiene, not correctness: a rewind can only ask for a checkpoint at or below the current
    // position, which reset puts at zero, so no stale entry is reachable even if this were left.
    // Dropping the array also lets a long finished sequence be collected.
    tokenHistory = new int[0];
  }

  @Override
  public void rewind(int checkpoint) {
    rewind(defaultSession, checkpoint);
  }

  @Override
  public void close() {
    // Nothing mapped and nothing pooled: the state is plain heap arrays owned by the forward pass.
  }

  private void requireDefault(Session session) {
    if (session != defaultSession) {
      throw new IllegalArgumentException("session was not opened by this LFM3 decoder");
    }
  }
}
