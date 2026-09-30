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
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Takes several independent decisions over one state by freezing that state once and forking every
 * question off the same physical key/value prefix.
 *
 * <p>This is where the tier's economics live. The state is prefilled once; each question then costs
 * its own few tokens plus one head read, and the base's work over the state is never repeated. The
 * evaluator asks the backend to <em>prove</em> the sharing rather than inferring it from timing,
 * and reports the answer either way, so an arm that silently stopped sharing cannot pass as
 * compliant.
 *
 * <p>Nothing here generates. The backend is never asked for logits.
 */
public final class SharedPrefixDecisionEvaluator {

  private final SharedPrefixInferenceBackend backend;

  /**
   * Creates an evaluator over a loaded backend.
   *
   * @param backend a backend that can both freeze a prefix and expose hidden states
   */
  public SharedPrefixDecisionEvaluator(SharedPrefixInferenceBackend backend) {
    this.backend = Objects.requireNonNull(backend, "backend");
  }

  /**
   * Answers every question against one shared state.
   *
   * @param stateTokens the tokens of the state all questions are asked about
   * @param questions the independent questions
   * @return the verdicts and the evidence that the prefix was shared
   * @throws IllegalStateException if the backend cannot share a prefix or expose hidden states
   */
  public DecisionBatch evaluate(int[] stateTokens, List<Question> questions) {
    Objects.requireNonNull(stateTokens, "stateTokens");
    Objects.requireNonNull(questions, "questions");
    if (questions.isEmpty()) {
      return new DecisionBatch(List.of(), 0, 0, true);
    }
    if (stateTokens.length == 0) {
      throw new IllegalArgumentException("stateTokens must not be empty");
    }
    if (!backend.supportsSharedPrefixes()) {
      throw new IllegalStateException(
          "backend " + backend.name() + " cannot freeze a shared prefix");
    }
    if (!backend.supportsHiddenState()) {
      throw new IllegalStateException(
          "backend " + backend.name() + " does not expose hidden states");
    }

    SharedInferencePrefix prefix = freezeState(stateTokens);
    List<Verdict> verdicts = new ArrayList<>(questions.size());
    List<InferenceSession> branches = new ArrayList<>(questions.size());
    try {
      for (Question question : questions) {
        InferenceSession branch = backend.fork(prefix);
        branches.add(branch);
        float[] hidden =
            backend.prefillHiddenState(branch, question.tokensInternal(), branch.checkpoint());
        verdicts.add(question.head().decide(hidden, question.temperature()));
      }
      boolean shared = allShare(branches);
      return new DecisionBatch(List.copyOf(verdicts), questions.size(), 1, shared);
    } finally {
      closeAll(branches);
    }
  }

  private SharedInferencePrefix freezeState(int[] stateTokens) {
    InferenceSession state = backend.openSession();
    try {
      backend.prefillHiddenState(state, stateTokens, state.checkpoint());
    } catch (RuntimeException failure) {
      state.close();
      throw failure;
    }
    return backend.freezePrefix(state);
  }

  private boolean allShare(List<InferenceSession> branches) {
    for (int index = 1; index < branches.size(); index++) {
      if (!backend.sharesPrefixStorage(branches.get(0), branches.get(index))) {
        return false;
      }
    }
    return branches.size() < 2 || backend.sharesPrefixStorage(branches.get(0), branches.get(0));
  }

  private static void closeAll(List<InferenceSession> branches) {
    RuntimeException failure = null;
    for (InferenceSession branch : branches) {
      try {
        branch.close();
      } catch (RuntimeException closeFailure) {
        if (failure == null) {
          failure = closeFailure;
        } else {
          failure.addSuppressed(closeFailure);
        }
      }
    }
    if (failure != null) {
      throw failure;
    }
  }
}
