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

import com.integrallis.models.backend.purejava.tensor.*;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Isolated;

@Isolated
class DebertaV2ParallelTest {
  @Test
  void unevenRowsAndBufferReuseMatchSingleThreadWithNonzeroWeights() {
    String key = DebertaV2ForwardPass.PARALLELISM_PROPERTY;
    String previous = System.getProperty(key);
    try {
      var config = new DebertaV2Config(8, 13, 3, 2, 11, 32, 8, 32, 8, 0, 1e-7f, true, true, false);
      System.setProperty(key, "1");
      var serial = DebertaV2Encoder.load(source(), config, "encoder.");
      System.setProperty(key, "5");
      var parallel = DebertaV2Encoder.load(source(), config, "encoder.");
      for (int count : new int[] {17, 3, 31, 1}) {
        int[] ids = new int[count];
        int[] positions = new int[count];
        for (int i = 0; i < count; i++) {
          ids[i] = (i * 7 + 1) % 11;
          positions[i] = i;
        }
        float[][] expected = serial.encode(ids, positions);
        float[][] actual = parallel.encode(ids, positions);
        for (int i = 0; i < count; i++) assertThat(actual[i]).containsExactly(expected[i]);
      }
    } finally {
      if (previous == null) System.clearProperty(key);
      else System.setProperty(key, previous);
    }
  }

  private static TensorSource source() {
    return new TensorSource() {
      public String format() {
        return "safetensors";
      }

      public List<String> tensorNames() {
        return List.of();
      }

      public TensorView tensor(String name) {
        boolean matrix = name.endsWith("weight") && !name.contains("LayerNorm");
        long[] shape = matrix ? new long[] {8, 8} : new long[] {8};
        if (name.contains("word_embeddings")) shape = new long[] {11, 8};
        if (name.contains("rel_embeddings")) shape = new long[] {16, 8};
        if (name.contains("intermediate.dense"))
          shape = matrix ? new long[] {13, 8} : new long[] {13};
        if (name.contains(".output.dense.weight") && !name.contains("attention"))
          shape = new long[] {8, 13};
        int elements = 1;
        for (long dim : shape) elements *= (int) dim;
        ByteBuffer data = ByteBuffer.allocate(elements * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < elements; i++) {
          float value = (float) Math.sin(i * .37 + (name.hashCode() & 255)) * .1f;
          if (name.contains("LayerNorm.weight")) value += 1;
          data.putFloat(value);
        }
        return new TensorView(
            name,
            shape,
            new TensorStorage("safetensors", "F32", 1, 4),
            MemorySegment.ofArray(data.array()));
      }
    };
  }
}
