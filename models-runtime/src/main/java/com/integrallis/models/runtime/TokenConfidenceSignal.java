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
package com.integrallis.models.runtime;

import java.util.Objects;

/**
 * Confidence-oriented distribution statistics for one generated token.
 *
 * <p>This mirrors Needle's lightweight fallback signals: chosen-token probability and distribution
 * entropy. It is not a calibrated confidence head; it is a deterministic signal callers can log,
 * threshold, or combine with model-specific calibration later.
 */
public record TokenConfidenceSignal(float chosenProbability, float entropy) {

  public TokenConfidenceSignal {
    if (!Float.isFinite(chosenProbability)
        || chosenProbability < 0.0f
        || chosenProbability > 1.0f) {
      throw new IllegalArgumentException(
          "chosenProbability must be finite and in [0, 1]: " + chosenProbability);
    }
    if (!Float.isFinite(entropy) || entropy < 0.0f) {
      throw new IllegalArgumentException("entropy must be finite and >= 0: " + entropy);
    }
  }

  /** Computes the chosen-token probability and entropy from raw logits. */
  public static TokenConfidenceSignal fromLogits(float[] logits, int chosenToken) {
    Objects.requireNonNull(logits, "logits");
    if (logits.length == 0) {
      throw new IllegalArgumentException("logits must not be empty");
    }
    if (chosenToken < 0 || chosenToken >= logits.length) {
      throw new IllegalArgumentException("chosenToken out of range: " + chosenToken);
    }

    float max = Float.NEGATIVE_INFINITY;
    for (float logit : logits) {
      if (!Float.isFinite(logit)) {
        throw new IllegalArgumentException("logits must be finite");
      }
      if (logit > max) {
        max = logit;
      }
    }

    double sum = 0.0d;
    double weightedLogProb = 0.0d;
    double chosenExp = 0.0d;
    double[] exps = new double[logits.length];
    for (int token = 0; token < logits.length; token++) {
      double exp = Math.exp(logits[token] - max);
      exps[token] = exp;
      sum += exp;
      if (token == chosenToken) {
        chosenExp = exp;
      }
    }
    for (int token = 0; token < logits.length; token++) {
      double probability = exps[token] / sum;
      weightedLogProb += probability * Math.log(probability);
    }
    return new TokenConfidenceSignal((float) (chosenExp / sum), (float) -weightedLogProb);
  }
}
