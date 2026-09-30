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
import com.integrallis.models.backend.purejava.lfm2.Lfm2ForwardPass;
import java.util.Arrays;
import java.util.Objects;

/**
 * Adapts the LFM2 hybrid decoder to the backend contract.
 *
 * <p>Single-sequence: the recurrent state is per decoder, not per session, so a second concurrent
 * session is refused rather than silently sharing one shift register.
 *
 * <p><b>Rewinding replays.</b> The convolutional layers carry a shift register, and rolling one
 * back would need a snapshot per position that this decoder does not keep. Rather than refuse --
 * which took LFM2 out of the RAG harness entirely, since reusing a shared prompt prefix begins by
 * rewinding to it -- the decoder retains the tokens it was given and rebuilds the state by running
 * them again from empty. Replay is not an approximation of the state: it is the definition of it.
 * The cost is recomputing the retained prefix, which is the same trade the Qwen3.5 decoder makes
 * for its Gated DeltaNet recurrence, and still far cheaper than the alternative of not running at
 * all.
 */
final class Lfm2DecoderAdapter implements PureJavaDecoder {

  private final class Lfm2Session implements Session {
    @Override
    public int checkpoint() {
      return forwardPass.nextPosition();
    }
  }

  private final Lfm2ForwardPass forwardPass;
  private final Lfm2Session defaultSession;
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

  Lfm2DecoderAdapter(Lfm2ForwardPass forwardPass) {
    this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
    this.defaultSession = new Lfm2Session();
  }

  @Override
  public int maxBatchSize() {
    return 1;
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
    // Serial: the convolution recurrence is sequential, so a batched prefill would still have to
    // advance it one token at a time.
    //
    // Through forward(int, int) rather than the graph directly, so these tokens are recorded. Going
    // straight to the forward pass left a prefilled prompt absent from the replay history, and the
    // first rewind then rebuilt an empty state -- which is how this is written down rather than
    // remembered.
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
          "LFM2 holds one recurrent state per decoder, so it supports a single session");
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
          "LFM2 decodes one sequence at a time; batched sessions are not supported");
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
        "LFM2 cannot verify speculatively: its convolution state cannot be rolled back");
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
      throw new IllegalArgumentException("session was not opened by this LFM2 decoder");
    }
  }
}
