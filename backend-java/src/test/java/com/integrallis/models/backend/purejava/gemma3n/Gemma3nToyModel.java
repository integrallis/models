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
package com.integrallis.models.backend.purejava.gemma3n;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufHeader;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufTensorInfo;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.io.ByteArrayOutputStream;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A toy Gemma 3n, small enough to recompute by hand and shaped to exercise every branch.
 *
 * <p>Deliberate choices, each so that a branch cannot be skipped:
 *
 * <ul>
 *   <li><b>Three AltUp streams</b>, so the predict coefficients are a 3x3 block and a transpose is
 *       visible. Two streams would make the mixture nearly symmetric.
 *   <li><b>Four layers with one sharing layer</b>, so both the owning and the sharing attention
 *       paths run. Layer 3 shares and is sliding, so it reads layer 1, which is also sliding.
 *   <li><b>Sparsity on layer 0 only</b>, with {@code -inf} elsewhere exactly as the published file
 *       carries it, so both feed-forward branches run and the non-finite guard is exercised.
 *   <li><b>A mixed sliding pattern</b>, so a sliding layer and a full-attention layer both attend.
 * </ul>
 *
 * <p>Every tensor is F32 here. The published file stores {@code laurel_*} and {@code altup_router}
 * as F16 -- that is covered separately, because an all-F32 fixture cannot distinguish a type that
 * is read correctly from one that is ignored.
 */
final class Gemma3nToyModel {

  static final int DIM = 8;
  static final int LAYERS = 4;
  static final int HEADS = 2;
  static final int KV_HEADS = 1;
  static final int HEAD_DIM = 4;
  static final int HIDDEN = 6;
  static final int VOCAB = 5;
  static final int ALTUP = 3;
  static final int ACTIVE = 0;
  static final int PER_LAYER_DIM = 2;
  static final int LAUREL_RANK = 2;
  static final int SHARED_KV_LAYERS = 1;
  static final int SLIDING_WINDOW = 2;
  static final float SPARSITY = 1.0f;
  static final float EPSILON = 1.0e-6f;
  static final float ROPE_THETA = 10_000.0f;

  /** Sliding on 0, 1 and 3; full attention on 2. Layer 3 shares, and reads layer 1. */
  static final List<Boolean> SLIDING = List.of(true, true, false, true);

  private final GgufFile file;
  private final Map<String, float[]> values;

  private Gemma3nToyModel(GgufFile file, Map<String, float[]> values) {
    this.file = file;
    this.values = values;
  }

  GgufFile file() {
    return file;
  }

  /** The raw float values, so a reference implementation can read the same numbers. */
  float[] tensor(String name) {
    float[] found = values.get(name);
    if (found == null) {
      throw new IllegalArgumentException("no such tensor in the fixture: " + name);
    }
    return found;
  }

  boolean has(String name) {
    return values.containsKey(name);
  }

  static Gemma3nConfig config() {
    List<Float> sparsity = new ArrayList<>();
    for (int layer = 0; layer < LAYERS; layer++) {
      // Negative infinity on the layers without sparsity, exactly as published.
      sparsity.add(layer == 0 ? SPARSITY : Float.NEGATIVE_INFINITY);
    }
    return new Gemma3nConfig(
        "gemma3n",
        DIM,
        LAYERS,
        HEADS,
        KV_HEADS,
        HEAD_DIM,
        VOCAB,
        16,
        HIDDEN,
        ALTUP,
        ACTIVE,
        PER_LAYER_DIM,
        SHARED_KV_LAYERS,
        ROPE_THETA,
        EPSILON,
        SLIDING_WINDOW,
        SLIDING,
        sparsity);
  }

  static Gemma3nToyModel create() {
    return create(Map.of(), false);
  }

  /**
   * The fixture with {@code laurel_l}, {@code laurel_r} and {@code altup_router} written as
   * <b>F16</b>, which is how the published file stores them while the rest of the layer is Q8_0.
   *
   * <p>An all-F32 fixture cannot tell a tensor type that is read correctly from one that is
   * ignored, because there every type is the same. The values written are the F16 round-trip of the
   * F32 ones, so an F32 twin built with {@link #halfRounded} is the reference.
   */
  static Gemma3nToyModel createWithHalfPrecisionExtras() {
    return create(Map.of(), true);
  }

  /** What F16 storage rounds each value to, so an F32 twin can hold the same numbers. */
  static float[] halfRounded(float[] source) {
    float[] rounded = new float[source.length];
    for (int index = 0; index < source.length; index++) {
      rounded[index] = Float.float16ToFloat(Float.floatToFloat16(source[index]));
    }
    return rounded;
  }

  /**
   * The same fixture with named tensors replaced.
   *
   * <p>Applied while the tensor is written, not afterwards: the builder APPENDS, so writing a
   * tensor twice leaves two entries under one name and the loader reads the first -- an override
   * after the fact is silently ignored.
   *
   * @param overrides tensor name to the values to write instead
   */
  static Gemma3nToyModel create(Map<String, float[]> overrides) {
    return create(overrides, false);
  }

  static Gemma3nToyModel create(Map<String, float[]> overrides, boolean halfPrecisionExtras) {
    Builder builder = new Builder(overrides, halfPrecisionExtras);
    builder.f32("token_embd.weight", new long[] {DIM, VOCAB}, matrix(VOCAB, DIM, 3));
    builder.f32(
        "per_layer_token_embd.weight",
        new long[] {PER_LAYER_DIM * LAYERS, VOCAB},
        matrix(VOCAB, PER_LAYER_DIM * LAYERS, 5));
    builder.f32(
        "per_layer_model_proj.weight",
        new long[] {DIM, PER_LAYER_DIM * LAYERS},
        matrix(PER_LAYER_DIM * LAYERS, DIM, 7));
    builder.f32("per_layer_proj_norm.weight", new long[] {PER_LAYER_DIM}, norm(9, PER_LAYER_DIM));
    // One [DIM, DIM] matrix per inactive stream, stacked: ALTUP - 1 of them.
    builder.f32(
        "altup_proj.weight", new long[] {DIM, DIM, ALTUP - 1}, matrix((ALTUP - 1) * DIM, DIM, 11));
    builder.f32(
        "altup_unembd_proj.weight",
        new long[] {DIM, DIM, ALTUP - 1},
        matrix((ALTUP - 1) * DIM, DIM, 13));
    builder.f32("output_norm.weight", new long[] {DIM}, norm(15, DIM));

    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      boolean ownsKv = layer < LAYERS - SHARED_KV_LAYERS;
      builder.f32(prefix + "attn_norm.weight", new long[] {DIM}, norm(20 + layer, DIM));
      builder.f32(
          prefix + "attn_q.weight",
          new long[] {DIM, HEADS * HEAD_DIM},
          matrix(HEADS * HEAD_DIM, DIM, 30 + layer));
      if (ownsKv) {
        builder.f32(
            prefix + "attn_k.weight",
            new long[] {DIM, KV_HEADS * HEAD_DIM},
            matrix(KV_HEADS * HEAD_DIM, DIM, 40 + layer));
        builder.f32(
            prefix + "attn_v.weight",
            new long[] {DIM, KV_HEADS * HEAD_DIM},
            matrix(KV_HEADS * HEAD_DIM, DIM, 50 + layer));
      }
      builder.f32(
          prefix + "attn_output.weight",
          new long[] {HEADS * HEAD_DIM, DIM},
          matrix(DIM, HEADS * HEAD_DIM, 60 + layer));
      builder.f32(prefix + "attn_q_norm.weight", new long[] {HEAD_DIM}, norm(70 + layer, HEAD_DIM));
      builder.f32(prefix + "attn_k_norm.weight", new long[] {HEAD_DIM}, norm(80 + layer, HEAD_DIM));
      builder.f32(prefix + "post_attention_norm.weight", new long[] {DIM}, norm(90 + layer, DIM));
      builder.f32(prefix + "ffn_norm.weight", new long[] {DIM}, norm(100 + layer, DIM));
      builder.f32(
          prefix + "ffn_gate.weight", new long[] {DIM, HIDDEN}, matrix(HIDDEN, DIM, 110 + layer));
      builder.f32(
          prefix + "ffn_up.weight", new long[] {DIM, HIDDEN}, matrix(HIDDEN, DIM, 120 + layer));
      builder.f32(
          prefix + "ffn_down.weight", new long[] {HIDDEN, DIM}, matrix(DIM, HIDDEN, 130 + layer));
      builder.f32(prefix + "post_ffw_norm.weight", new long[] {DIM}, norm(140 + layer, DIM));
      builder.f32(
          prefix + "inp_gate.weight",
          new long[] {DIM, PER_LAYER_DIM},
          matrix(PER_LAYER_DIM, DIM, 150 + layer));
      builder.f32(
          prefix + "proj.weight",
          new long[] {PER_LAYER_DIM, DIM},
          matrix(DIM, PER_LAYER_DIM, 160 + layer));
      builder.f32(prefix + "post_norm.weight", new long[] {DIM}, norm(170 + layer, DIM));
      builder.f32(
          prefix + "altup_router.weight", new long[] {DIM, ALTUP}, matrix(ALTUP, DIM, 180 + layer));
      builder.f32(prefix + "altup_router_norm.weight", new long[] {DIM}, norm(190 + layer, DIM));
      builder.f32(
          prefix + "altup_predict_coef.weight",
          new long[] {ALTUP, ALTUP * ALTUP},
          matrix(ALTUP * ALTUP, ALTUP, 200 + layer));
      builder.f32(
          prefix + "altup_correct_coef.weight",
          new long[] {ALTUP, ALTUP},
          matrix(ALTUP, ALTUP, 210 + layer));
      builder.f32(prefix + "altup_correct_scale.weight", new long[] {DIM}, norm(220 + layer, DIM));
      builder.f32(
          prefix + "laurel_l.weight",
          new long[] {DIM, LAUREL_RANK},
          matrix(LAUREL_RANK, DIM, 230 + layer));
      builder.f32(
          prefix + "laurel_r.weight",
          new long[] {LAUREL_RANK, DIM},
          matrix(DIM, LAUREL_RANK, 240 + layer));
      builder.f32(prefix + "laurel_post_norm.weight", new long[] {DIM}, norm(250 + layer, DIM));
    }
    return new Gemma3nToyModel(builder.build(), Map.copyOf(builder.values));
  }

  /** Small, deterministic, and signed, so a dropped term is visible rather than absorbed. */
  static float[] matrix(int rows, int columns, int seed) {
    float[] result = new float[rows * columns];
    for (int index = 0; index < result.length; index++) {
      result[index] = (((seed + index * 7) % 13) - 6) * 0.05f;
    }
    return result;
  }

  /** Norm weights near one, so a norm that is skipped entirely still changes the answer. */
  static float[] norm(int seed, int length) {
    float[] result = new float[length];
    for (int index = 0; index < length; index++) {
      result[index] = 0.85f + ((seed + index * 3) % 7) * 0.04f;
    }
    return result;
  }

  private static final class Builder {
    private final List<GgufTensorInfo> infos = new ArrayList<>();
    private final Map<String, float[]> values = new HashMap<>();
    private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    private final Map<String, float[]> overrides;
    private final boolean halfPrecisionExtras;

    private Builder(Map<String, float[]> overrides, boolean halfPrecisionExtras) {
      this.overrides = overrides;
      this.halfPrecisionExtras = halfPrecisionExtras;
    }

    /** The tensors the published file stores as F16 rather than alongside the K-quants. */
    private boolean storedAsHalf(String name) {
      return halfPrecisionExtras
          && (name.endsWith("laurel_l.weight")
              || name.endsWith("laurel_r.weight")
              || name.endsWith("altup_router.weight"));
    }

    private void f32(String name, long[] shape, float[] declared) {
      float[] tensorValues = overrides.getOrDefault(name, declared);
      if (storedAsHalf(name)) {
        half(name, shape, tensorValues);
        return;
      }
      GgufTensorInfo info =
          new GgufTensorInfo(name, shape.length, shape, GgufTensorType.F32, bytes.size());
      if (tensorValues.length != info.elementCount()) {
        throw new IllegalArgumentException(
            name + " has " + tensorValues.length + " values but needs " + info.elementCount());
      }
      ByteBuffer encoded =
          ByteBuffer.allocate(tensorValues.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      for (float value : tensorValues) {
        encoded.putFloat(value);
      }
      infos.add(info);
      values.put(name, tensorValues.clone());
      bytes.writeBytes(encoded.array());
    }

    private void half(String name, long[] shape, float[] tensorValues) {
      GgufTensorInfo info =
          new GgufTensorInfo(name, shape.length, shape, GgufTensorType.F16, bytes.size());
      if (tensorValues.length != info.elementCount()) {
        throw new IllegalArgumentException(
            name + " has " + tensorValues.length + " values but needs " + info.elementCount());
      }
      ByteBuffer encoded =
          ByteBuffer.allocate(tensorValues.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      for (float value : tensorValues) {
        encoded.putShort(Float.floatToFloat16(value));
      }
      infos.add(info);
      // What a loader reads back is the rounded value, not the original.
      values.put(name, halfRounded(tensorValues));
      bytes.writeBytes(encoded.array());
    }

    private GgufFile build() {
      return new GgufFile(
          new GgufHeader(3, infos.size(), 0),
          new GgufMetadata(Map.of()),
          infos,
          0,
          MemorySegment.ofArray(bytes.toByteArray()));
    }
  }
}
