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
package com.integrallis.models.backend.purejava.mobilemoe;

import java.util.Objects;

/** Zero-copy packed routed-expert projections for one MobileMoE layer. */
final class MobileMoeExperts {

  record Expert(MobileMoePackedInt4RightMatrix gateUp, MobileMoePackedInt4RightMatrix down) {
    Expert {
      Objects.requireNonNull(gateUp, "gateUp");
      Objects.requireNonNull(down, "down");
    }
  }

  private final Expert[] experts;

  MobileMoeExperts(Expert[] experts) {
    if (Objects.requireNonNull(experts, "experts").length == 0) {
      throw new IllegalArgumentException("experts must not be empty");
    }
    this.experts = experts.clone();
    for (int index = 0; index < this.experts.length; index++) {
      Objects.requireNonNull(this.experts[index], "experts[" + index + "]");
    }
  }

  Expert expert(int index) {
    if (index < 0 || index >= experts.length) {
      throw new IndexOutOfBoundsException("expert index out of range: " + index);
    }
    return experts[index];
  }
}
