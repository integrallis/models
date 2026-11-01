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

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.backend.purejava.tensor.*;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import org.junit.jupiter.api.Test;

class DebertaV2EncoderTest {
  @Test
  void encoderLoadsBothStorageTypesWithoutRequiringARerankerHead() {
    var config = new DebertaV2Config(2, 4, 1, 1, 2, 8, 2, 8, 2, 0, 1e-7f, true, true, false);
    float[][] first =
        DebertaV2Encoder.load(source(true), config, "encoder.")
            .encode(new int[] {1, 0}, new int[] {0, 1});
    float[][] second =
        DebertaV2Encoder.load(source(false), config, "encoder.")
            .encode(new int[] {1, 0}, new int[] {0, 1});
    assertThat(first[0]).containsExactly(second[0]);
    assertThat(first[1]).containsExactly(second[1]);
    assertThat(first[0][0]).isCloseTo(1f, within(1e-5f));
    assertThat(first[0][1]).isCloseTo(-1f, within(1e-5f));
    assertThat(first[1]).containsExactly(0f, 0f);
  }

  private static TensorSource source(boolean f32) {
    return new TensorSource() {
      public String format() {
        return "safetensors";
      }

      public List<String> tensorNames() {
        return List.of();
      }

      public TensorView tensor(String name) {
        assertThat(name).startsWith("encoder.");
        assertThat(name).doesNotContain("pooler", "classifier");
        boolean matrix = name.endsWith("weight") && !name.contains("LayerNorm");
        long[] shape = matrix ? new long[] {2, 2} : new long[] {2};
        if (name.contains("rel_embeddings")) {
          shape = new long[] {4, 2};
        }
        if (name.contains("intermediate.dense")) {
          shape = matrix ? new long[] {4, 2} : new long[] {4};
        }
        if (name.contains(".output.dense.weight") && !name.contains("attention")) {
          shape = new long[] {2, 4};
        }
        int elements = 1;
        for (long dim : shape) {
          elements *= (int) dim;
        }
        int bytes = f32 ? 4 : 2;
        ByteBuffer data = ByteBuffer.allocate(elements * bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < elements; i++) {
          float value = name.contains("LayerNorm.weight") ? 1 : 0;
          if (name.contains("word_embeddings") && i >= 2) {
            value = i == 2 ? 1 : -1;
          }
          if (f32) {
            data.putFloat(value);
          } else {
            data.putShort(Float.floatToFloat16(value));
          }
        }
        return new TensorView(
            name,
            shape,
            new TensorStorage("safetensors", f32 ? "F32" : "F16", 1, bytes),
            MemorySegment.ofArray(data.array()));
      }
    };
  }
}
