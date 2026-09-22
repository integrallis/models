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
package com.integrallis.models.decisions.external;

import java.util.Objects;

/**
 * A verdict obtained from a third-party decision service, held for scoring and nothing else.
 *
 * <p>This type is deliberately impoverished. It carries an item identifier and a probability, and
 * it exposes no hidden state, no logits and no feature vector, because a third-party service's
 * terms may forbid using its output to train, distil or imitate a model. TypeSafe's Master Customer
 * Agreement section 2.3(b) forbids exactly that, and this project's head must remain demonstrably
 * free of any such influence.
 *
 * <p>The isolation is enforced by a test rather than by convention: no class on the training path
 * may reference this package. Keeping the boundary structural means a future contributor cannot
 * cross it by accident, and the provenance of every trained weight stays provable.
 *
 * @param itemId the corpus item this verdict answers, so it can be joined to a label for scoring
 * @param probabilityTrue the service's stated probability that the proposition holds
 * @param service the service and version that produced it, for the evidence record
 */
public record ExternalVerdict(String itemId, double probabilityTrue, String service) {

  /** Validates the verdict. */
  public ExternalVerdict {
    Objects.requireNonNull(itemId, "itemId");
    Objects.requireNonNull(service, "service");
    if (itemId.isBlank() || service.isBlank()) {
      throw new IllegalArgumentException("itemId and service must not be blank");
    }
    if (!Double.isFinite(probabilityTrue) || probabilityTrue < 0.0 || probabilityTrue > 1.0) {
      throw new IllegalArgumentException("probabilityTrue must lie within zero and one");
    }
  }

  /** Returns the decision at a 0.5 threshold, for accuracy scoring against a corpus label. */
  public boolean verdict() {
    return probabilityTrue >= 0.5;
  }
}
