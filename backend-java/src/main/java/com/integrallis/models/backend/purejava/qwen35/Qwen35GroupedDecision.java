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
 * <p>MEASURED 2026-09-24, Harriet on a Hetzner CCX33 (8 vCPU, EPYC Milan, 4 physical cores): twenty
 * questions over one contract cost 8.95 s one at a time, a flat 0.45 s each. Answering them
 * together costs 8.63 s. The gain is 4%.
 *
 * <p>That is worth writing down plainly, because this class was built expecting far more. The
 * expectation was that a decision is bandwidth bound -- that reading 2.55 GiB of weights dominates,
 * that a question's dozen tokens ride along for free, and that N questions therefore pay for the
 * weights N times over. <b>On this hardware that is false.</b> A batched prefill measures 18.5 ms
 * per token, dead linear from 17 tokens to 143 with no fixed cost, and it halves from one thread to
 * two and again from two to four before saturating at the box's four physical cores. It is compute
 * bound and already at the arithmetic ceiling. N questions cost N questions' arithmetic however
 * they are arranged, and no amount of batching changes that.
 *
 * <p>What grouping does still save is the single-token step at the end of each question, which
 * <em>is</em> bandwidth bound -- 65 ms, and flat in thread count past two. One per question becomes
 * one per group. Twenty questions save nineteen of them, about 1.2 s, which is the 4% observed and
 * very nearly all of it. With the native decode kernel switched off that step costs more and the
 * same grouping is worth 1.69x, which is the same finding seen from the other side.
 *
 * <p>So the honest summary: grouping trades a bandwidth-bound step per question for one per group,
 * and it is a wash unless that step is expensive. It loses below about ten questions, where the
 * lockstep walk's own narrow batches are themselves bandwidth bound.
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
 * about 2 GiB.
 */
final class Qwen35GroupedDecision {

  /** The private state one question needs while the evidence's state is shared. */
  static final class Branch {
    private final float[][] convolutionHistory;
    private final float[][] suffixKeys;
    private final float[][] suffixValues;
    private int position;

    private Branch(Qwen35Config config, int prefixCapacity, int suffixCapacity) {
      int layers = config.numLayers();
      int span = Math.addExact(prefixCapacity, suffixCapacity);
      convolutionHistory = new float[layers][];
      suffixKeys = new float[layers][];
      suffixValues = new float[layers][];
      for (int layer = 0; layer < layers; layer++) {
        if (config.usesFullAttention(layer)) {
          // Prefix and suffix in one array. The prefix half is ~10 MiB a branch against the 100 MiB
          // of recurrent state, so copying it buys an ordinary single-region attention rather than
          // a two-region one, and the saving would have been noise.
          int entries = Math.multiplyExact(span, config.attentionKeyDim());
          suffixKeys[layer] = new float[entries];
          suffixValues[layer] = new float[entries];
        } else {
          convolutionHistory[layer] =
              new float[Math.multiplyExact(config.gdnConvDim(), config.gdnConvKernel() - 1)];
        }
      }
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
    void forkFrom(
        float[][] evidenceConvolution,
        float[][] evidenceKeys,
        float[][] evidenceValues,
        int prefixLength,
        int attentionKeyDim) {
      copyEach(evidenceConvolution, convolutionHistory);
      int prefixEntries = Math.multiplyExact(prefixLength, attentionKeyDim);
      for (int layer = 0; layer < evidenceKeys.length; layer++) {
        if (evidenceKeys[layer] != null && suffixKeys[layer] != null) {
          System.arraycopy(evidenceKeys[layer], 0, suffixKeys[layer], 0, prefixEntries);
          System.arraycopy(evidenceValues[layer], 0, suffixValues[layer], 0, prefixEntries);
        }
      }
      position = prefixLength;
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
    this.config = config;
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
        evidenceConvolution, evidenceKeys, evidenceValues, prefixLength, attentionKeyDim);
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
    for (float[] layer : recurrentState) {
      bytes += layer == null ? 0L : (long) layer.length * Float.BYTES;
    }
    for (Branch branch : branches) {
      for (float[] layer : branch.suffixKeys) {
        bytes += layer == null ? 0L : 2L * layer.length * Float.BYTES;
      }
    }
    return bytes;
  }
}
