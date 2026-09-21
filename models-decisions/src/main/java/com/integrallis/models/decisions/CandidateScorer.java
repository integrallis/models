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

import java.util.List;
import java.util.Objects;

/**
 * Scores every candidate of an answer space independently against one shared state, then normalises
 * within the question.
 *
 * <p>One mechanism covers all three primitives: a {@link Noul} is two candidates, a {@link Choice}
 * is up to 255, a {@link Score} is its ordered levels. Nothing here is generated and no candidate
 * is compared against another during scoring, so the probabilities are native rather than read off
 * a letter the model was asked to emit.
 *
 * <p>Independence is the load-bearing property. Because a candidate's score does not depend on the
 * others, permuting the candidates permutes the output and changes nothing else — order
 * equivariance holds by construction rather than by sorting the input first. It also means the
 * ratio between two candidates is unaffected by the presence of a third, which is what allows a
 * question's candidate set to be edited without re-deriving the rest.
 */
public final class CandidateScorer {

  /** Scores how well one candidate fits the state. Higher is better; any finite scale will do. */
  @FunctionalInterface
  public interface Alignment {

    /**
     * Returns the unnormalised score of a candidate against a state.
     *
     * @param state the shared state's representation
     * @param candidate the candidate's representation
     */
    double score(float[] state, float[] candidate);
  }

  private final Alignment alignment;
  private final double temperature;

  /**
   * Creates a scorer.
   *
   * @param alignment how a candidate is scored against the state
   * @param temperature the calibration temperature applied across candidates
   */
  public CandidateScorer(Alignment alignment, double temperature) {
    this.alignment = Objects.requireNonNull(alignment, "alignment");
    if (!Double.isFinite(temperature) || temperature <= 0.0) {
      throw new IllegalArgumentException("temperature must be positive and finite");
    }
    this.temperature = temperature;
  }

  /**
   * Scores every candidate and returns the normalised verdict.
   *
   * @param space the answer space, whose labels are the candidates
   * @param state the shared state's representation, computed once for the whole question
   * @param candidates one representation per label, in the space's declaration order
   * @throws IllegalArgumentException if the candidate count does not match the space
   */
  public Verdict score(AnswerSpace space, float[] state, List<float[]> candidates) {
    Objects.requireNonNull(space, "space");
    Objects.requireNonNull(state, "state");
    Objects.requireNonNull(candidates, "candidates");
    if (candidates.size() != space.size()) {
      throw new IllegalArgumentException(
          "need one candidate per label: " + space.size() + ", got " + candidates.size());
    }

    double[] scores = new double[candidates.size()];
    for (int index = 0; index < candidates.size(); index++) {
      float[] candidate = Objects.requireNonNull(candidates.get(index), "candidate");
      double score = alignment.score(state, candidate);
      if (!Double.isFinite(score)) {
        throw new IllegalArgumentException("candidate score must be finite");
      }
      scores[index] = score;
    }
    return new Verdict(space, Calibration.softmax(scores, temperature));
  }
}
