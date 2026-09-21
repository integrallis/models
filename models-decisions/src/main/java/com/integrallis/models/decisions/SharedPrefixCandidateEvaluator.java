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
package com.integrallis.models.decisions;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import java.util.List;
import java.util.Objects;

/**
 * Scores candidates by their token log-probabilities over a state evaluated exactly once.
 *
 * <p>This is where the architecture earns its keep. The state is prefilled a single time and the
 * resulting distribution already covers every single-token candidate, so those cost nothing further
 * at all; a multi-token candidate costs only its own tokens beyond the first. A hosted service must
 * re-ingest the state for every question and every candidate, and no hardware closes that gap,
 * because it follows from being outside the process that computed the state rather than from being
 * slow.
 *
 * <p>Scores are length-normalised. Without that the shortest label wins by default, since every
 * additional token subtracts log-probability whatever the evidence says.
 *
 * <p>Nothing is generated. The backend is asked for distributions, never for text.
 */
public final class SharedPrefixCandidateEvaluator {

  private final SharedPrefixInferenceBackend backend;
  private final double temperature;

  /**
   * Creates an evaluator.
   *
   * @param backend a backend that can freeze and fork a physical prefix
   * @param temperature the calibration temperature applied across candidates
   */
  public SharedPrefixCandidateEvaluator(SharedPrefixInferenceBackend backend, double temperature) {
    this.backend = Objects.requireNonNull(backend, "backend");
    if (!Double.isFinite(temperature) || temperature <= 0.0) {
      throw new IllegalArgumentException("temperature must be positive and finite");
    }
    this.temperature = temperature;
  }

  /** Scores every candidate against the state and returns the normalised verdict. */
  public Verdict evaluate(int[] stateTokens, AnswerSpace space, List<int[]> candidateTokens) {
    return evaluateBatch(stateTokens, space, candidateTokens).verdict();
  }

  /**
   * Scores every candidate and reports how the work was actually done.
   *
   * @param stateTokens the shared state, evaluated once
   * @param space the answer space whose labels are the candidates
   * @param candidateTokens one token sequence per label, in declaration order
   */
  public CandidateBatch evaluateBatch(
      int[] stateTokens, AnswerSpace space, List<int[]> candidateTokens) {
    Objects.requireNonNull(stateTokens, "stateTokens");
    Objects.requireNonNull(space, "space");
    Objects.requireNonNull(candidateTokens, "candidateTokens");
    if (stateTokens.length == 0) {
      throw new IllegalArgumentException("stateTokens must not be empty");
    }
    if (candidateTokens.size() != space.size()) {
      throw new IllegalArgumentException(
          "need one candidate per label: " + space.size() + ", got " + candidateTokens.size());
    }

    float[] prefixLogits;
    SharedInferencePrefix prefix;
    InferenceSession state = backend.openSession();
    try {
      prefixLogits = backend.prefill(state, stateTokens, state.checkpoint()).clone();
    } catch (RuntimeException failure) {
      state.close();
      throw failure;
    }
    prefix = backend.freezePrefix(state);

    double[] scores = new double[candidateTokens.size()];
    int tokensEvaluated = 0;
    boolean shared;
    InferenceSession branch = backend.fork(prefix);
    try {
      int checkpoint = branch.checkpoint();
      for (int index = 0; index < candidateTokens.size(); index++) {
        int[] candidate = Objects.requireNonNull(candidateTokens.get(index), "candidate");
        if (candidate.length == 0) {
          throw new IllegalArgumentException("a candidate must contribute at least one token");
        }
        // The first token is free: the state's own logits already predict it.
        double total = logSoftmaxAt(prefixLogits, candidate[0]);
        for (int step = 1; step < candidate.length; step++) {
          float[] logits = backend.forward(branch, candidate[step - 1], checkpoint + step - 1);
          tokensEvaluated++;
          total += logSoftmaxAt(logits, candidate[step]);
        }
        if (candidate.length > 1) {
          backend.rewind(branch, checkpoint);
        }
        scores[index] = total / candidate.length;
      }
      // Proving the sharing needs two distinct forks. Comparing a session with itself would
      // always hold and would establish nothing at all.
      InferenceSession witness = backend.fork(prefix);
      try {
        shared = backend.sharesPrefixStorage(branch, witness);
      } finally {
        witness.close();
      }
    } finally {
      branch.close();
    }

    return new CandidateBatch(
        new Verdict(space, Calibration.softmax(scores, temperature)), 1, tokensEvaluated, shared);
  }

  /** Log of the softmax probability of one token, computed stably. */
  private static double logSoftmaxAt(float[] logits, int token) {
    if (token < 0 || token >= logits.length) {
      throw new IllegalArgumentException("token " + token + " is outside the vocabulary");
    }
    double largest = Double.NEGATIVE_INFINITY;
    for (float value : logits) {
      largest = Math.max(largest, value);
    }
    double sum = 0.0;
    for (float value : logits) {
      sum += Math.exp(value - largest);
    }
    return (logits[token] - largest) - Math.log(sum);
  }
}
