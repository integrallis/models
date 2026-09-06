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
package com.integrallis.models.backend.purejava.deberta;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class DebertaV2ForwardPassTest {

  @Test
  void matchesReferenceLogBucketsAtLinearAndLogarithmicBoundaries() {
    int[] relativePositions = {-511, -300, -129, -128, -127, 0, 127, 128, 129, 300, 511};
    int[] buckets = new int[relativePositions.length];
    for (int index = 0; index < relativePositions.length; index++) {
      buckets[index] = DebertaV2ForwardPass.relativeBucket(relativePositions[index], 256, 512);
    }

    assertThat(buckets).containsExactly(-255, -207, -129, -128, -127, 0, 127, 128, 129, 207, 255);
  }
}
