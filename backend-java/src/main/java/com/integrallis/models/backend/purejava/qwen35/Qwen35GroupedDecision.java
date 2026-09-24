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

/**
 * State for answering several questions about one piece of evidence in lockstep.
 *
 * <p>MEASURED 2026-09-24, Harriet on an 8-vCPU EPYC-Milan box: twenty questions over one contract
 * cost 9.12 s, a flat 0.46 s each, because each question re-read the model's weights. A question's
 * own tokens are a dozen or so; the 2.55 GiB sweep dwarfs them. Answering N questions the obvious
 * way sweeps the weights N times.
 *
 * <p>A projection does not care where its rows came from. Run one token from each of N questions as
 * a batch of N rows and the weights are read <em>once</em> for all of them, which is how a hosted
 * System One model holds latency flat as questions are added.
 *
 * <p>What cannot be shared is anything carrying sequence identity:
 *
 * <ul>
 *   <li><b>Gated DeltaNet state.</b> A running summary of everything read so far. Each question
 *       needs its own, forked from the evidence, or question two would answer having silently read
 *       question one. This is the expensive part -- 16 heads x 256 x 256 floats per layer over 24
 *       layers, about 100 MiB per question.
 *   <li><b>Key and value cache for the eight full-attention layers.</b> The evidence's half is
 *       identical for every question and is read, never written, so it is shared. Only the
 *       question's own dozen positions are private, which is under a megabyte each.
 * </ul>
 *
 * <p>Concatenating the questions into one sequence would be far cheaper and is wrong: causal
 * attention lets a later question read an earlier one's instructions, and the recurrent state
 * carries that contamination even further. Isolation is the whole contract of a typed decision, so
 * each question gets its own state and the weights are what get shared.
 *
 * <p>Group size is bounded because the recurrent state is not free: twenty questions at once is
 * about 2 GiB. Callers answer in groups and the weight sweep amortises across each group.
 */
final class Qwen35GroupedDecision {

  /** The private state one question needs while the evidence's state is shared. */
  static final class Branch {
    private final float[][] recurrentState;
    private final float[][] convolutionHistory;
    private final float[][] suffixKeys;
    private final float[][] suffixValues;
    private int position;

    private Branch(Qwen35Config config, int suffixCapacity) {
      int layers = config.numLayers();
      recurrentState = new float[layers][];
      convolutionHistory = new float[layers][];
      suffixKeys = new float[layers][];
      suffixValues = new float[layers][];
      for (int layer = 0; layer < layers; layer++) {
        if (config.usesFullAttention(layer)) {
          int span = Math.multiplyExact(suffixCapacity, config.attentionKeyDim());
          suffixKeys[layer] = new float[span];
          suffixValues[layer] = new float[span];
        } else {
          convolutionHistory[layer] =
              new float[Math.multiplyExact(config.gdnConvDim(), config.gdnConvKernel() - 1)];
          recurrentState[layer] =
              new float
                  [Math.multiplyExact(
                      Math.multiplyExact(config.gdnValueHeads(), config.gdnHeadDim()),
                      config.gdnHeadDim())];
        }
      }
    }

    float[] recurrentState(int layer) {
      return recurrentState[layer];
    }

    float[] convolutionHistory(int layer) {
      return convolutionHistory[layer];
    }

    float[] suffixKeys(int layer) {
      return suffixKeys[layer];
    }

    float[] suffixValues(int layer) {
      return suffixValues[layer];
    }

    int position() {
      return position;
    }

    void advance() {
      position++;
    }

    /**
     * Forks this branch from the evidence's state.
     *
     * <p>Copied rather than referenced: the recurrence writes the state in place as it reads, so
     * two questions sharing one buffer would answer each other's evidence.
     */
    void forkFrom(float[][] evidenceRecurrent, float[][] evidenceConvolution) {
      copyEach(evidenceRecurrent, recurrentState);
      copyEach(evidenceConvolution, convolutionHistory);
      position = 0;
    }

    private static void copyEach(float[][] source, float[][] destination) {
      for (int layer = 0; layer < source.length; layer++) {
        if (source[layer] != null && destination[layer] != null) {
          System.arraycopy(source[layer], 0, destination[layer], 0, source[layer].length);
        }
      }
    }
  }

  private final Qwen35Config config;
  private final Branch[] branches;
  private final int suffixCapacity;

  Qwen35GroupedDecision(Qwen35Config config, int groupSize, int suffixCapacity) {
    if (groupSize < 1) {
      throw new IllegalArgumentException("groupSize must be >= 1: " + groupSize);
    }
    if (suffixCapacity < 1) {
      throw new IllegalArgumentException("suffixCapacity must be >= 1: " + suffixCapacity);
    }
    this.config = config;
    this.suffixCapacity = suffixCapacity;
    this.branches = new Branch[groupSize];
    for (int index = 0; index < groupSize; index++) {
      branches[index] = new Branch(config, suffixCapacity);
    }
  }

  int groupSize() {
    return branches.length;
  }

  int suffixCapacity() {
    return suffixCapacity;
  }

  Branch branch(int index) {
    return branches[index];
  }

  Qwen35Config config() {
    return config;
  }

  /** Bytes of private state a group holds, which is what bounds how many questions run at once. */
  long stateBytes() {
    long bytes = 0L;
    for (Branch branch : branches) {
      for (float[] layer : branch.recurrentState) {
        bytes += layer == null ? 0L : (long) layer.length * Float.BYTES;
      }
      for (float[] layer : branch.suffixKeys) {
        bytes += layer == null ? 0L : 2L * layer.length * Float.BYTES;
      }
    }
    return bytes;
  }
}
