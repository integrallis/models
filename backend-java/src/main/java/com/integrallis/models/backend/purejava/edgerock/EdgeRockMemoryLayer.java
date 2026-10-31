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
package com.integrallis.models.backend.purejava.edgerock;

import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.util.Objects;

/**
 * One EdgeRock memory layer: read the bank, then write this token into it.
 *
 * <p>Order is the contract. The read runs against committed state only, so a token never attends to
 * itself, and the write stages into the pending chunk which stays invisible until the chunk fills.
 * Swapping the two is the kind of change that improves a benchmark and stops being this model.
 *
 * <p>Projections arrive as {@link Projection} rather than as the quantized {@code Matrix} views the
 * rest of the backend uses. The dequantizing matvec is private to {@code Qwen35ForwardPass}, a
 * 2600-line final class on the hot path of three qualified architectures, so taking a seam here
 * keeps this layer's arithmetic verifiable without duplicating those kernels or cutting into that
 * path. Wiring the seam is one change when the layer loop lands.
 */
public final class EdgeRockMemoryLayer {

  /** A linear projection, {@code out = W x}. Implemented over quantized weights in production. */
  @FunctionalInterface
  interface Projection {
    void apply(float[] input, float[] output);
  }

  /** The weights of one memory layer, in the order the GGUF names them. */
  public record Weights(
      float[] inputNorm,
      Projection queryProjection,
      float[] queryNorm,
      Projection keyProjection,
      float[] keyNorm,
      Projection valueProjection,
      Projection outputProjection,
      Projection writeGate,
      Projection writeUp,
      Projection writeDown,
      Projection admissionDown,
      Projection admissionUp,
      float[][] readGateWeight,
      float[] readGateBias,
      float[] readScale) {}

  private final EdgeRockConfig config;
  private final Weights weights;
  private final float[] normed;
  private final float[] queryFlat;
  private final float[][] queries;
  private final float[] gates;
  private final float[][] readOut;
  private final float[] readFlat;
  private final float[] written;
  private final float[] writeGateOut;
  private final float[] writeUpOut;
  private final float[] writeDownOut;
  private final float[] keyFlat;
  private final float[] valueFlat;
  private final float[][] groupKeys;
  private final float[][] groupValues;
  private final float[] admissionHidden;
  private final float[] scores;

  public EdgeRockMemoryLayer(EdgeRockConfig config, Weights weights) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    int embedding = config.embeddingLength();
    int heads = config.headCount();
    int groups = config.headCountKv();
    int dim = config.headDim();
    this.normed = new float[embedding];
    this.queryFlat = new float[heads * dim];
    // Preallocated rather than filled by a fresh copy per call. These are rotated in place, so a
    // copy is needed, but the array holding it does not have to be new: at 16 heads and 4 groups
    // over 8 memory layers this was 160 short-lived arrays per token.
    this.queries = new float[heads][dim];
    this.gates = new float[heads];
    this.readOut = new float[heads][dim];
    this.readFlat = new float[heads * dim];
    this.written = new float[embedding];
    this.writeGateOut = new float[config.memoryWriterIntermediate()];
    this.writeUpOut = new float[config.memoryWriterIntermediate()];
    this.writeDownOut = new float[embedding];
    this.keyFlat = new float[groups * dim];
    this.valueFlat = new float[groups * dim];
    this.groupKeys = new float[groups][dim];
    this.groupValues = new float[groups][dim];
    this.admissionHidden = new float[config.memoryAdmissionRank()];
    this.scores = new float[groups];
  }

  /**
   * One head's rotary embedding, supplied by the caller.
   *
   * <p>Injected rather than computed here so that the decoder's own {@code RotaryTable} is the only
   * source of rope factors in the model. The memory layer rotates its queries and its written keys
   * with exactly the rotation the attention half uses; a second table initialised independently
   * would be a setting that means two different things on two code paths.
   */
  @FunctionalInterface
  public interface Rotation {
    /** Rotates one head in place. */
    void rotate(float[] head);
  }

  /**
   * The rotation for one position, from precomputed NeoX pair factors.
   *
   * <p>{@code cos} is {@code rope.dimension_count / 2} wide, which is the number of rotated pairs
   * and not half the head: EdgeRock's heads are 256 wide against a rope dimension of 64, so this
   * rotates the leading 64 lanes and leaves the remaining 192 untouched.
   */
  public static Rotation rotation(float[] cos, float[] sin) {
    Objects.requireNonNull(cos, "cos");
    Objects.requireNonNull(sin, "sin");
    return head -> TensorOps.ropeNeox(head, 0, cos, sin);
  }

  /**
   * Runs the layer for one token.
   *
   * @param hidden the block's hidden state, before the block's input norm -- this layer applies its
   *     own {@code memory.input_norm}, and the reference feeds it the pre-norm residual
   * @param position the token's sequence position
   * @param rotation the rotary embedding for this position
   * @param bank this layer's state, mutated
   * @param out the layer's contribution, overwritten
   */
  public void forward(
      float[] hidden, int position, Rotation rotation, MemoryBank bank, float[] out) {
    int heads = config.headCount();
    int groups = config.headCountKv();
    int dim = config.headDim();

    TensorOps.rmsNorm(
        normed,
        0,
        hidden,
        0,
        weights.inputNorm(),
        config.embeddingLength(),
        config.rmsNormEpsilon());

    // --- read, from committed records only ------------------------------------------------
    weights.queryProjection().apply(normed, queryFlat);
    for (int head = 0; head < heads; head++) {
      // In place at this head's offset. Elementwise, so aliasing out and x at the same offset is
      // safe, and it keeps the shared pinned reduction rather than a local sum that would not be
      // bit-identical to every other norm in the model.
      TensorOps.rmsNorm(
          queryFlat,
          head * dim,
          queryFlat,
          head * dim,
          weights.queryNorm(),
          dim,
          config.rmsNormEpsilon());
      System.arraycopy(queryFlat, head * dim, queries[head], 0, dim);
    }
    // The gate is read off the normed query *before* rope, and only then is the query rotated.
    // The reference is explicit about the order and it matters from the second token onward: at
    // position 0 rope is the identity, so gating after rotation agrees there and diverges by
    // ~5e-2 everywhere else. Parity against the reference is what caught it; no unit test would.
    MemoryRead.gates(
        queries, weights.readGateWeight(), weights.readGateBias(), weights.readScale(), gates);
    for (int head = 0; head < heads; head++) {
      rotation.rotate(queries[head]);
    }
    MemoryRead.read(bank, queries, gates, config.headsPerGroup(), readOut);
    for (int head = 0; head < heads; head++) {
      System.arraycopy(readOut[head], 0, readFlat, head * dim, dim);
    }
    weights.outputProjection().apply(readFlat, out);

    // --- write, staged -------------------------------------------------------------------
    // Residual: the writer refines the normed state rather than replacing it, and write_down is
    // zero-initialised in the reference, so an untrained writer is exactly the identity.
    weights.writeGate().apply(normed, writeGateOut);
    weights.writeUp().apply(normed, writeUpOut);
    for (int i = 0; i < writeGateOut.length; i++) {
      writeGateOut[i] = silu(writeGateOut[i]) * writeUpOut[i];
    }
    weights.writeDown().apply(writeGateOut, writeDownOut);
    for (int i = 0; i < written.length; i++) {
      written[i] = normed[i] + writeDownOut[i];
    }

    weights.keyProjection().apply(written, keyFlat);
    weights.valueProjection().apply(written, valueFlat);
    for (int group = 0; group < groups; group++) {
      TensorOps.rmsNorm(
          keyFlat,
          group * dim,
          keyFlat,
          group * dim,
          weights.keyNorm(),
          dim,
          config.rmsNormEpsilon());
      System.arraycopy(keyFlat, group * dim, groupKeys[group], 0, dim);
      rotation.rotate(groupKeys[group]);
      System.arraycopy(valueFlat, group * dim, groupValues[group], 0, dim);
    }

    // Admission reads the normed state, not the written one, and the reference detaches it: the
    // scorer judges what arrived rather than what the writer made of it.
    weights.admissionDown().apply(normed, admissionHidden);
    for (int i = 0; i < admissionHidden.length; i++) {
      admissionHidden[i] = silu(admissionHidden[i]);
    }
    weights.admissionUp().apply(admissionHidden, scores);

    bank.stage(groupKeys, groupValues, scores);
  }

  private static float silu(float value) {
    return (float) (value / (1.0 + Math.exp(-value)));
  }
}
