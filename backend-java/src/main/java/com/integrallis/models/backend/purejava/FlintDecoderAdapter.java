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

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.backend.purejava.flint.FlintForwardPass;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Adapts Flint sequence state to the architecture-neutral backend contract. */
final class FlintDecoderAdapter implements PureJavaDecoder {

  private final class FlintSession implements Session {
    private final FlintForwardPass.Session delegate = forwardPass.openSession();

    @Override
    public int checkpoint() {
      return delegate.checkpoint();
    }
  }

  private final FlintForwardPass forwardPass;
  private FlintSession defaultSession;

  FlintDecoderAdapter(FlintForwardPass forwardPass) {
    this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
  }

  BackendDiagnostics diagnostics(BackendDiagnostics base) {
    var routes = forwardPass.projectionRoutes();
    var environment = new LinkedHashMap<>(base.environment());
    environment.put("flint-diagnostics-version", "1");
    environment.put("architecture-prefill-batch-size", "1");
    var decisions = new ArrayList<>(base.optimizations());
    decisions.removeIf(
        decision ->
            Set.of(
                    "grouped-projections",
                    "batched-prefill",
                    "batched-attention-scores",
                    "batched-attention-values",
                    "fused-grouped-attention",
                    "final-layer-prefill-pruning",
                    "final-layer-kv-only-prefill")
                .contains(decision.id()));
    boolean eligible = routes.qkvLayers() + routes.gateUpLayers() + routes.expertLayers() > 0;
    decisions.add(
        new OptimizationDecision(
            "flint-injected-projection-groups",
            eligible ? OptimizationStatus.ENABLED : OptimizationStatus.UNSUPPORTED,
            "Flint tests mapped QKV, dense/shared gate-up, and routed-expert shapes against the injected kernel; remaining projections use Java helpers. Counts describe eligibility, not measured invocations.",
            Map.of(
                "implementation",
                routes.implementation(),
                "qkv-eligible-layers",
                Integer.toString(routes.qkvLayers()),
                "dense-shared-gate-up-eligible-layers",
                Integer.toString(routes.gateUpLayers()),
                "routed-expert-eligible-layers",
                Integer.toString(routes.expertLayers()),
                "q4-k-reduction",
                "sequential-block-order")));
    decisions.add(
        new OptimizationDecision(
            "batched-prefill",
            OptimizationStatus.UNSUPPORTED,
            "Flint advances prompt tokens sequentially and skips only discarded vocabulary projections",
            Map.of("batch-size", "1")));
    decisions.add(
        new OptimizationDecision(
            "flint-java-attention",
            OptimizationStatus.ENABLED,
            "Flint uses pinned key scores, sink softmax, vectorized non-fused chronological values, and independent parallel heads above one million elements; generic batched/native attention switches do not apply",
            Map.of("key-score-rows", "1", "value-rows", "1", "implementation", "java")));
    decisions.add(
        new OptimizationDecision(
            "final-layer-prefill-pruning",
            OptimizationStatus.UNSUPPORTED,
            "Flint retains final-layer attention and FFN work for every prompt token",
            Map.of("output-rows", "all")));
    return new BackendDiagnostics(base.backend(), base.planVersion(), environment, decisions);
  }

  private FlintSession defaultSession() {
    if (defaultSession == null) defaultSession = new FlintSession();
    return defaultSession;
  }

  @Override
  public int maxBatchSize() {
    return 1;
  }

  @Override
  public float[] forward(int token, int position) {
    return forwardPass.forward(defaultSession().delegate, token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    return forwardPass.forwardTransient(defaultSession().delegate, token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return prefill(defaultSession().delegate, tokens, startPosition);
  }

  @Override
  public Session openSession() {
    return new FlintSession();
  }

  @Override
  public float[] forward(Session session, int token, int position) {
    return forwardPass.forward(requireSession(session).delegate, token, position);
  }

  @Override
  public float[] forwardTransient(Session session, int token, int position) {
    return forwardPass.forwardTransient(requireSession(session).delegate, token, position);
  }

  @Override
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    return prefill(requireSession(session).delegate, tokens, startPosition);
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
      throw new IllegalArgumentException("Flint currently supports one session per batch");
    }
    FlintSession session = requireSession(sessions[0]);
    float[] logits =
        forwardPass.forwardTransient(session.delegate, tokens[0], session.delegate.checkpoint());
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
    return defaultSession().checkpoint();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    return verifyTransient(tokens, startPosition).snapshot();
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    validate(defaultSession().delegate, tokens, startPosition);
    int vocabularySize = forwardPass.config().vocabSize();
    float[] rows = new float[Math.multiplyExact(tokens.length, vocabularySize)];
    for (int index = 0; index < tokens.length; index++) {
      float[] logits =
          forwardPass.forwardTransient(
              defaultSession().delegate, tokens[index], startPosition + index);
      System.arraycopy(logits, 0, rows, index * vocabularySize, vocabularySize);
    }
    return new LogitBatch(tokens.length, vocabularySize, rows);
  }

  @Override
  public void rewind(int checkpoint) {
    forwardPass.rewind(defaultSession().delegate, checkpoint);
  }

  @Override
  public void reset() {
    forwardPass.reset(defaultSession().delegate);
  }

  @Override
  public void close() {}

  private float[] prefill(FlintForwardPass.Session session, int[] tokens, int startPosition) {
    validate(session, tokens, startPosition);
    int last = tokens.length - 1;
    for (int index = 0; index < last; index++) {
      forwardPass.advance(session, tokens[index], startPosition + index);
    }
    return forwardPass.forward(session, tokens[last], startPosition + last);
  }

  private static void validate(FlintForwardPass.Session session, int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition != session.checkpoint()) {
      throw new IllegalArgumentException(
          "position must be sequential: expected "
              + session.checkpoint()
              + ", got "
              + startPosition);
    }
  }

  private FlintSession requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (!(session instanceof FlintDecoderAdapter.FlintSession flintSession)) {
      throw new IllegalArgumentException("session belongs to a different decoder");
    }
    return flintSession;
  }
}
