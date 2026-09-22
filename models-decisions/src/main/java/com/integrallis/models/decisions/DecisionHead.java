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

/**
 * Reads one hidden state the base model has already computed and scores a bounded answer space.
 *
 * <p>A head does not generate. There is no decode loop, no sampling and no token: the cost of a
 * decision is the cost of the read, which is why decisions taken over a shared prefix are close to
 * free.
 */
public interface DecisionHead {

  /** Returns the answer space this head scores. */
  AnswerSpace space();

  /**
   * Scores every outcome of {@link #space()} from a hidden state.
   *
   * @param hiddenState the base model's final normalized hidden state
   * @return one raw score per outcome, in declaration order
   */
  double[] logits(float[] hiddenState);

  /**
   * Scores the hidden state and turns the result into a calibrated verdict.
   *
   * @param hiddenState the base model's final normalized hidden state
   * @param temperature the calibration temperature established on held-out data
   */
  default Verdict decide(float[] hiddenState, double temperature) {
    return new Verdict(space(), Calibration.softmax(logits(hiddenState), temperature));
  }
}
