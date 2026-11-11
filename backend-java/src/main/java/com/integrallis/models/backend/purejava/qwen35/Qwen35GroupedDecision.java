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

import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel.FloatArrayView;

/**
 * State for answering several questions about one piece of evidence in lockstep.
 *
 * <p>Each branch owns its convolution history and attention cache. The evidence prefix is copied
 * into each branch before the question is appended, so one question cannot change another's state.
 * The forward pass supplies the shared step position and manages recurrent state separately. Memory
 * use depends on the model dimensions, prefix and suffix capacities, and branch count.
 */
final class Qwen35GroupedDecision {

  /** The private state one question needs while the evidence's state is shared. */
  static final class Branch {
    private final float[][] convolutionHistory;
    private FloatArrayView[] convolutionHistoryViews;
    private final float[][] suffixKeys;
    private final float[][] suffixValues;

    private Branch(Qwen35Config config, int prefixCapacity, int suffixCapacity) {
      int layers = config.numLayers();
      int span = Math.addExact(prefixCapacity, suffixCapacity);
      convolutionHistory = new float[layers][];
      suffixKeys = new float[layers][];
      suffixValues = new float[layers][];
      for (int layer = 0; layer < layers; layer++) {
        if (config.usesFullAttention(layer)) {
          // Prefix and suffix occupy one owned region for ordinary single-region attention.
          int entries = Math.multiplyExact(span, config.attentionKeyDim());
          suffixKeys[layer] = new float[entries];
          suffixValues[layer] = new float[entries];
        } else {
          convolutionHistory[layer] =
              new float[Math.multiplyExact(config.gdnConvDim(), config.gdnConvKernel() - 1)];
        }
      }
    }

    FloatArrayView convolutionHistoryView(int layer, GgufBatchedMatrixKernel kernel) {
      if (!kernel.supportsRecurrentActivations()) return null;
      if (convolutionHistoryViews == null)
        convolutionHistoryViews = new FloatArrayView[convolutionHistory.length];
      if (convolutionHistoryViews[layer] == null) {
        convolutionHistoryViews[layer] = kernel.prepareFloatArray(convolutionHistory[layer]);
      }
      return convolutionHistoryViews[layer];
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

    /**
     * Forks this branch from the evidence's state.
     *
     * <p>Copying the prefix keeps subsequent convolution and attention writes private to this
     * question.
     */
    void forkFrom(
        float[][] evidenceConvolution,
        float[][] evidenceKeys,
        float[][] evidenceValues,
        int prefixLength,
        int attentionKeyDim,
        Qwen35Checkpoint shared) {
      copyEach(evidenceConvolution, convolutionHistory);
      int prefixEntries = Math.multiplyExact(prefixLength, attentionKeyDim);
      int sharedEntries =
          shared == null ? 0 : Math.multiplyExact(shared.tokens.length, attentionKeyDim);
      for (int layer = 0; layer < evidenceKeys.length; layer++) {
        if (suffixKeys[layer] != null) {
          if (sharedEntries > 0) {
            shared.copyAttention(layer, suffixKeys[layer], suffixValues[layer]);
          }
          if (prefixEntries > sharedEntries) {
            System.arraycopy(
                evidenceKeys[layer],
                0,
                suffixKeys[layer],
                sharedEntries,
                prefixEntries - sharedEntries);
            System.arraycopy(
                evidenceValues[layer],
                0,
                suffixValues[layer],
                sharedEntries,
                prefixEntries - sharedEntries);
          }
        }
      }
    }

    private static void copyEach(float[][] source, float[][] destination) {
      for (int layer = 0; layer < source.length; layer++) {
        if (source[layer] != null && destination[layer] != null) {
          System.arraycopy(source[layer], 0, destination[layer], 0, source[layer].length);
        }
      }
    }
  }

  private final Branch[] branches;
  private final int suffixCapacity;

  /**
   * Recurrent state for every branch of a layer in one array, branch-major.
   *
   * <p>Held here rather than on the branch so a step is a single kernel call. A kernel told to
   * advance one sequence by one token has nothing to spread across a thread pool and runs on the
   * caller; told to advance twenty, it has twenty independent recurrences. MEASURED 2026-09-24:
   * calling the recurrence once per branch made twenty grouped questions cost 12.4 s, and one call
   * for the whole group brought that to 10.8 s.
   */
  private final float[][] recurrentState;

  private final int recurrentStateElements;

  Qwen35GroupedDecision(
      Qwen35Config config, int groupSize, int prefixCapacity, int suffixCapacity) {
    if (groupSize < 1) {
      throw new IllegalArgumentException("groupSize must be >= 1: " + groupSize);
    }
    if (suffixCapacity < 1) {
      throw new IllegalArgumentException("suffixCapacity must be >= 1: " + suffixCapacity);
    }
    this.suffixCapacity = suffixCapacity;
    this.branches = new Branch[groupSize];
    for (int index = 0; index < groupSize; index++) {
      branches[index] = new Branch(config, prefixCapacity, suffixCapacity);
    }
    this.recurrentStateElements =
        Math.multiplyExact(
            Math.multiplyExact(config.gdnValueHeads(), config.gdnHeadDim()), config.gdnHeadDim());
    this.recurrentState = new float[config.numLayers()][];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (!config.usesFullAttention(layer)) {
        recurrentState[layer] = new float[Math.multiplyExact(groupSize, recurrentStateElements)];
      }
    }
  }

  /** Every branch's recurrent state for one layer, branch-major, or null on attention layers. */
  float[] recurrentState(int layer) {
    return recurrentState[layer];
  }

  /** Floats one branch occupies in a layer's recurrent state. */
  int recurrentStateElements() {
    return recurrentStateElements;
  }

  /**
   * Forks a branch from the evidence's state.
   *
   * <p>Copied rather than referenced: the recurrence writes the state in place as it reads, so two
   * questions sharing one buffer would answer each other's evidence.
   */
  void forkBranch(
      int index,
      float[][] evidenceRecurrent,
      float[][] evidenceConvolution,
      float[][] evidenceKeys,
      float[][] evidenceValues,
      int prefixLength,
      int attentionKeyDim) {
    forkBranch(
        index,
        evidenceRecurrent,
        evidenceConvolution,
        evidenceKeys,
        evidenceValues,
        prefixLength,
        attentionKeyDim,
        null);
  }

  void forkBranch(
      int index,
      float[][] evidenceRecurrent,
      float[][] evidenceConvolution,
      float[][] evidenceKeys,
      float[][] evidenceValues,
      int prefixLength,
      int attentionKeyDim,
      Qwen35Checkpoint shared) {
    for (int layer = 0; layer < recurrentState.length; layer++) {
      if (recurrentState[layer] != null && evidenceRecurrent[layer] != null) {
        System.arraycopy(
            evidenceRecurrent[layer],
            0,
            recurrentState[layer],
            index * recurrentStateElements,
            recurrentStateElements);
      }
    }
    branches[index].forkFrom(
        evidenceConvolution, evidenceKeys, evidenceValues, prefixLength, attentionKeyDim, shared);
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
}
