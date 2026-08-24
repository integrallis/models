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
package com.integrallis.models.backend.purejava.cact;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CactFp16TensorTest {

  @Test
  void readsLittleEndianHalfPrecisionRows() {
    CactFile file =
        CactParser.parseSegment(
            MemorySegment.ofArray(
                new SyntheticCactBuilder()
                    .addFp16Values(new int[] {2, 3}, 0.5f, -1.0f, 2.0f, 3.0f, 4.0f, -5.0f)
                    .build()));
    CactFp16Tensor tensor = CactFp16Tensor.from(file.tensor(0));

    assertThat(tensor.shape()).containsExactly(2, 3);
    assertThat(tensor.get(0, 2)).isEqualTo(2.0f);
    assertThat(tensor.get(1, 0)).isEqualTo(3.0f);
    assertThat(tensor.readRow(1)).containsExactly(3.0f, 4.0f, -5.0f);
    assertThat(tensor.readAll()).containsExactly(0.5f, -1.0f, 2.0f, 3.0f, 4.0f, -5.0f);
  }

  @Test
  void rejectsNonFp16AndOutOfBoundsAccess() {
    CactFile file =
        CactParser.parseSegment(
            MemorySegment.ofArray(new SyntheticCactBuilder().addRaw(new byte[] {1}).build()));

    assertThatThrownBy(() -> CactFp16Tensor.from(file.tensor(0)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("FP16");
  }
}
