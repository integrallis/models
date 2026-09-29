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
import com.integrallis.models.backend.purejava.gemma3n.Gemma3nForwardPass;
import java.util.Arrays;
import java.util.Objects;

/**
 * Adapts the Gemma 3n decoder to the backend contract.
 *
 * <p>Single-sequence: AltUp's router produces its mixing coefficients per token, so the state is
 * per decoder rather than per session and a second concurrent session is refused instead of
 * silently sharing one.
 *
 * <p><b>Rewinding replays.</b> Gemma 3n's key-value cache could be truncated, but its sliding
 * layers use ring buffers whose evicted positions are gone, so a rewind past the window cannot be
 * served from the cache. Rather than refuse -- which took LFM2 out of the RAG harness entirely
 * until 2026-09-29, because reusing a shared prompt prefix begins by rewinding to it -- this
 * retains the tokens and rebuilds the state by running them again. Replay is not an approximation
 * of the state: it is the definition of it.
 */
final class Gemma3nDecoderAdapter implements PureJavaDecoder {

  private final class Gemma3nSession implements Session {
    @Override
    public int checkpoint() {
      return forwardPass.nextPosition();
    }
  }

  private final Gemma3nForwardPass forwardPass;
  private final Gemma3nSession defaultSession;
  private boolean sessionHandedOut;

  /**
   * Every token this decoder has been advanced through, in position order.
   *
   * <p>Written by {@link #forward(int, int)}, which every path that advances the graph goes
   * through. A path that called the graph directly would advance the state without recording it,
   * and the next rewind would rebuild a state missing those tokens -- which is exactly the defect
   * the LFM2 adapter shipped with before its own tests caught it.
   */
  private int[] tokenHistory = new int[0];

  Gemma3nDecoderAdapter(Gemma3nForwardPass forwardPass) {
    this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
    this.defaultSession = new Gemma3nSession();
  }

  @Override
  public int maxBatchSize() {
    return 1;
  }

  @Override
  public float[] forward(int token, int position) {
    float[] logits = forwardPass.forward(token, position);
    // After the call, so a rejected position leaves no token recorded for it.
    record(token, position);
    return logits;
  }

  private void record(int token, int position) {
    if (position >= tokenHistory.length) {
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
    // Serial: AltUp's coefficients are per token, so a batched prefill would still advance one
    // token
    // at a time. Through forward(int, int) so these tokens are recorded for a later rewind.
    float[] logits = null;
    for (int index = 0; index < tokens.length; index++) {
      logits = forward(tokens[index], startPosition + index);
    }
    return logits;
  }

  @Override
  public Session openSession() {
    if (sessionHandedOut) {
      throw new UnsupportedOperationException(
          "Gemma 3n holds one AltUp state per decoder, so it supports a single session");
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
          "Gemma 3n decodes one sequence at a time; batched sessions are not supported");
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
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + position + ": " + checkpoint);
    }
    forwardPass.reset();
    for (int replay = 0; replay < checkpoint; replay++) {
      forwardPass.forward(tokenHistory[replay], replay);
    }
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
   * Refused: speculative verification has to discard the tokens it tried.
   *
   * <p>{@link #rewind} can undo them by replay, but replaying the whole prefix after every
   * speculative batch costs more than the speculation saves, so this stays a refusal rather than
   * becoming a silent pessimisation.
   */
  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    throw new UnsupportedOperationException(
        "Gemma 3n cannot verify speculatively: undoing a batch means replaying the prefix");
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
    tokenHistory = new int[0];
  }

  @Override
  public void rewind(int checkpoint) {
    rewind(defaultSession, checkpoint);
  }

  @Override
  public void close() {
    // Nothing mapped and nothing pooled beyond the weights, which the arena owns.
  }

  private void requireDefault(Session session) {
    if (session != defaultSession) {
      throw new IllegalArgumentException("session was not opened by this Gemma 3n decoder");
    }
  }
}
