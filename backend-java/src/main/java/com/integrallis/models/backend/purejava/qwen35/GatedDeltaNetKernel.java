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
package com.integrallis.models.backend.purejava.qwen35;

import com.integrallis.vectors.core.MathUtil;
import com.integrallis.vectors.core.VectorRuntimeCapabilities;
import com.integrallis.vectors.core.VectorUtil;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/** Exact per-column recurrence: keep memory/readout in SIMD registers across state rows. */
final class GatedDeltaNetKernel {
  private static final VectorRuntimeCapabilities CAPS = VectorUtil.runtimeCapabilities();
  private static final VectorSpecies<Float> SPECIES =
      VectorSpecies.of(
          float.class, VectorShape.forBitSize(CAPS.vectorApi() ? CAPS.activeVectorBits() : 128));

  private static FloatVector fma(FloatVector a, float b, FloatVector c) {
    var scale = FloatVector.broadcast(SPECIES, b);
    return CAPS.fastVectorFma() ? a.fma(scale, c) : a.mul(scale).add(c);
  }

  private static final ThreadLocal<float[][]> SCRATCH =
      ThreadLocal.withInitial(() -> new float[2][0]);

  private GatedDeltaNetKernel() {}

  static void apply(
      float[] queryInput,
      float[] keyInput,
      float[] value,
      float[] logDecay,
      float[] beta,
      float[] state,
      float[] out,
      int stateBase,
      int tokens,
      int keyHeads,
      int valueHeads,
      int kd,
      int vd,
      int firstHead,
      int endHead) {
    float[][] scratch = SCRATCH.get();
    if (scratch[0].length != kd) {
      scratch = new float[][] {new float[kd], new float[kd]};
      SCRATCH.set(scratch);
    }
    float[] q = scratch[0], k = scratch[1];
    float queryScale = (float) (1.0 / Math.sqrt(kd));
    int vectorEnd = CAPS.vectorApi() ? SPECIES.loopBound(vd) : 0;
    for (int h = firstHead; h < endHead; h++)
      for (int t = 0; t < tokens; t++) {
        int qAt = (t * keyHeads + h % keyHeads) * kd,
            th = t * valueHeads + h,
            at = th * vd,
            stateAt = stateBase + h * kd * vd;
        normalize(queryInput, qAt, q, 0, kd, queryScale);
        normalize(keyInput, qAt, k, 0, kd, 1);
        float decay = (float) Math.exp(logDecay[th]);
        int col = 0;
        for (; col + 4 * SPECIES.length() <= vectorEnd; col += 4 * SPECIES.length()) {
          var memory0 = FloatVector.zero(SPECIES);
          var memory1 = FloatVector.zero(SPECIES);
          var memory2 = FloatVector.zero(SPECIES);
          var memory3 = FloatVector.zero(SPECIES);
          for (int row = 0; row < kd; row++) {
            int cell = stateAt + row * vd + col;
            var key = FloatVector.broadcast(SPECIES, k[row]);
            var decayed0 =
                FloatVector.fromArray(SPECIES, state, cell + 0 * SPECIES.length()).mul(decay);
            memory0 =
                CAPS.fastVectorFma() ? decayed0.fma(key, memory0) : decayed0.mul(key).add(memory0);
            var decayed1 =
                FloatVector.fromArray(SPECIES, state, cell + 1 * SPECIES.length()).mul(decay);
            memory1 =
                CAPS.fastVectorFma() ? decayed1.fma(key, memory1) : decayed1.mul(key).add(memory1);
            var decayed2 =
                FloatVector.fromArray(SPECIES, state, cell + 2 * SPECIES.length()).mul(decay);
            memory2 =
                CAPS.fastVectorFma() ? decayed2.fma(key, memory2) : decayed2.mul(key).add(memory2);
            var decayed3 =
                FloatVector.fromArray(SPECIES, state, cell + 3 * SPECIES.length()).mul(decay);
            memory3 =
                CAPS.fastVectorFma() ? decayed3.fma(key, memory3) : decayed3.mul(key).add(memory3);
          }
          var delta0 =
              FloatVector.fromArray(SPECIES, value, at + col + 0 * SPECIES.length())
                  .sub(memory0)
                  .mul(beta[th]);
          var result0 = FloatVector.zero(SPECIES);
          var delta1 =
              FloatVector.fromArray(SPECIES, value, at + col + 1 * SPECIES.length())
                  .sub(memory1)
                  .mul(beta[th]);
          var result1 = FloatVector.zero(SPECIES);
          var delta2 =
              FloatVector.fromArray(SPECIES, value, at + col + 2 * SPECIES.length())
                  .sub(memory2)
                  .mul(beta[th]);
          var result2 = FloatVector.zero(SPECIES);
          var delta3 =
              FloatVector.fromArray(SPECIES, value, at + col + 3 * SPECIES.length())
                  .sub(memory3)
                  .mul(beta[th]);
          var result3 = FloatVector.zero(SPECIES);
          for (int row = 0; row < kd; row++) {
            int cell = stateAt + row * vd + col;
            var key = FloatVector.broadcast(SPECIES, k[row]);
            var query = FloatVector.broadcast(SPECIES, q[row]);
            var decayed0 =
                FloatVector.fromArray(SPECIES, state, cell + 0 * SPECIES.length()).mul(decay);
            var updated0 =
                CAPS.fastVectorFma() ? delta0.fma(key, decayed0) : delta0.mul(key).add(decayed0);
            updated0.intoArray(state, cell + 0 * SPECIES.length());
            result0 =
                CAPS.fastVectorFma()
                    ? updated0.fma(query, result0)
                    : updated0.mul(query).add(result0);
            var decayed1 =
                FloatVector.fromArray(SPECIES, state, cell + 1 * SPECIES.length()).mul(decay);
            var updated1 =
                CAPS.fastVectorFma() ? delta1.fma(key, decayed1) : delta1.mul(key).add(decayed1);
            updated1.intoArray(state, cell + 1 * SPECIES.length());
            result1 =
                CAPS.fastVectorFma()
                    ? updated1.fma(query, result1)
                    : updated1.mul(query).add(result1);
            var decayed2 =
                FloatVector.fromArray(SPECIES, state, cell + 2 * SPECIES.length()).mul(decay);
            var updated2 =
                CAPS.fastVectorFma() ? delta2.fma(key, decayed2) : delta2.mul(key).add(decayed2);
            updated2.intoArray(state, cell + 2 * SPECIES.length());
            result2 =
                CAPS.fastVectorFma()
                    ? updated2.fma(query, result2)
                    : updated2.mul(query).add(result2);
            var decayed3 =
                FloatVector.fromArray(SPECIES, state, cell + 3 * SPECIES.length()).mul(decay);
            var updated3 =
                CAPS.fastVectorFma() ? delta3.fma(key, decayed3) : delta3.mul(key).add(decayed3);
            updated3.intoArray(state, cell + 3 * SPECIES.length());
            result3 =
                CAPS.fastVectorFma()
                    ? updated3.fma(query, result3)
                    : updated3.mul(query).add(result3);
          }
          result0.intoArray(out, at + col + 0 * SPECIES.length());
          result1.intoArray(out, at + col + 1 * SPECIES.length());
          result2.intoArray(out, at + col + 2 * SPECIES.length());
          result3.intoArray(out, at + col + 3 * SPECIES.length());
        }
        for (; col < vectorEnd; col += SPECIES.length()) {
          var memory = FloatVector.zero(SPECIES);
          for (int row = 0; row < kd; row++) {
            var decayed =
                FloatVector.fromArray(SPECIES, state, stateAt + row * vd + col).mul(decay);
            memory = fma(decayed, k[row], memory);
          }
          var delta = FloatVector.fromArray(SPECIES, value, at + col).sub(memory).mul(beta[th]);
          var result = FloatVector.zero(SPECIES);
          for (int row = 0; row < kd; row++) {
            int cell = stateAt + row * vd + col;
            var decayed = FloatVector.fromArray(SPECIES, state, cell).mul(decay);
            var updated = fma(delta, k[row], decayed);
            updated.intoArray(state, cell);
            result = fma(updated, q[row], result);
          }
          result.intoArray(out, at + col);
        }
        for (col = vectorEnd; col < vd; col++) {
          float memory = 0;
          for (int row = 0; row < kd; row++)
            memory = MathUtil.fma(state[stateAt + row * vd + col] * decay, k[row], memory);
          float delta = (value[at + col] - memory) * beta[th], result = 0;
          for (int row = 0; row < kd; row++) {
            int cell = stateAt + row * vd + col;
            float updated = MathUtil.fma(delta, k[row], state[cell] * decay);
            state[cell] = updated;
            result = MathUtil.fma(updated, q[row], result);
          }
          out[at + col] = result;
        }
      }
  }

  private static void normalize(
      float[] from, int at, float[] to, int dest, int dimension, float scale) {
    float sum = 0;
    for (int n = 0; n < dimension; n++) {
      float x = from[at + n];
      sum += x * x;
    }
    float inverse = (float) (1.0 / Math.sqrt(sum + 1e-6f));
    for (int n = 0; n < dimension; n++) to[dest + n] = (from[at + n] * inverse) * scale;
  }
}
