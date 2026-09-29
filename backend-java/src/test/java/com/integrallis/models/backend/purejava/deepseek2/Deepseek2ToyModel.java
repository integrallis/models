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
package com.integrallis.models.backend.purejava.deepseek2;

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
 * A toy DeepSeek-V2, shaped so every branch runs.
 *
 * <ul>
 *   <li><b>Two heads</b>, so the shared rotary key part has to be copied to more than one -- with
 *       one head, sharing and not sharing are the same bytes.
 *   <li><b>A leading dense layer and two routed ones</b>, so both feed-forward paths run.
 *   <li><b>Four experts, two used</b>, so selection is not a no-op, and a shared expert of double
 *       the expert width, as the published file fuses two into one.
 *   <li><b>YaRN on</b>, so the rotary magnitude and the attention scale both differ from the plain
 *       ones.
 * </ul>
 */
final class Deepseek2ToyModel {

  static final int DIM = 8;
  static final int LAYERS = 3;
  static final int HEADS = 2;
  static final int ROPE_DIM = 2;
  static final int NO_ROPE_DIM = 2;
  static final int KEY_LENGTH = NO_ROPE_DIM + ROPE_DIM;
  static final int VALUE_LENGTH = 3;
  static final int KV_LORA_RANK = 4;
  static final int VOCAB = 5;
  static final int HIDDEN = 6;
  static final int EXPERT_HIDDEN = 3;
  static final int EXPERTS = 4;
  static final int EXPERTS_USED = 2;
  static final int SHARED_EXPERTS = 2;
  static final int LEADING_DENSE = 1;
  static final float ROPE_THETA = 10_000.0f;
  static final float ROPE_FACTOR = 40.0f;
  static final int ROPE_ORIGINAL_CONTEXT = 4_096;
  static final float YARN_LOG_MULTIPLIER = 0.07069999724626541f;
  static final float EPSILON = 1.0e-6f;

  private final GgufFile file;
  private final Map<String, float[]> values;

  private Deepseek2ToyModel(GgufFile file, Map<String, float[]> values) {
    this.file = file;
    this.values = values;
  }

  GgufFile file() {
    return file;
  }

  float[] tensor(String name) {
    float[] found = values.get(name);
    if (found == null) {
      throw new IllegalArgumentException("no such tensor in the fixture: " + name);
    }
    return found;
  }

  static Deepseek2Config config() {
    return new Deepseek2Config(
        DIM,
        LAYERS,
        HEADS,
        KEY_LENGTH,
        VALUE_LENGTH,
        KV_LORA_RANK,
        ROPE_DIM,
        VOCAB,
        32,
        HIDDEN,
        EXPERT_HIDDEN,
        EXPERTS,
        EXPERTS_USED,
        SHARED_EXPERTS,
        1.0f,
        LEADING_DENSE,
        ROPE_THETA,
        ROPE_FACTOR,
        ROPE_ORIGINAL_CONTEXT,
        YARN_LOG_MULTIPLIER,
        EPSILON);
  }

  static Deepseek2ToyModel create() {
    return create(Map.of());
  }

  /**
   * The fixture with named tensors replaced, applied while each is written.
   *
   * <p>Not afterwards: the builder appends, so writing a tensor twice leaves two entries under one
   * name and the loader reads the first.
   */
  static Deepseek2ToyModel create(Map<String, float[]> overrides) {
    Builder builder = new Builder(overrides);
    Deepseek2Config config = config();
    builder.f32("token_embd.weight", new long[] {DIM, VOCAB}, matrix(VOCAB, DIM, 3));
    builder.f32("output_norm.weight", new long[] {DIM}, norm(5, DIM));

    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      builder.f32(prefix + "attn_norm.weight", new long[] {DIM}, norm(10 + layer, DIM));
      builder.f32(
          prefix + "attn_q.weight",
          new long[] {DIM, config.queryDim()},
          matrix(config.queryDim(), DIM, 20 + layer));
      builder.f32(
          prefix + "attn_kv_a_mqa.weight",
          new long[] {DIM, config.compressedKeyValueDim()},
          matrix(config.compressedKeyValueDim(), DIM, 30 + layer));
      builder.f32(
          prefix + "attn_kv_a_norm.weight",
          new long[] {KV_LORA_RANK},
          norm(40 + layer, KV_LORA_RANK));
      builder.f32(
          prefix + "attn_kv_b.weight",
          new long[] {KV_LORA_RANK, config.decompressedKeyValueDim()},
          matrix(config.decompressedKeyValueDim(), KV_LORA_RANK, 50 + layer));
      builder.f32(
          prefix + "attn_output.weight",
          new long[] {config.cachedValueDim(), DIM},
          matrix(DIM, config.cachedValueDim(), 60 + layer));
      builder.f32(prefix + "ffn_norm.weight", new long[] {DIM}, norm(70 + layer, DIM));

      if (layer < LEADING_DENSE) {
        builder.f32(prefix + "ffn_gate.weight", new long[] {DIM, HIDDEN}, matrix(HIDDEN, DIM, 80));
        builder.f32(prefix + "ffn_up.weight", new long[] {DIM, HIDDEN}, matrix(HIDDEN, DIM, 90));
        builder.f32(prefix + "ffn_down.weight", new long[] {HIDDEN, DIM}, matrix(DIM, HIDDEN, 100));
        continue;
      }
      builder.f32(
          prefix + "ffn_gate_inp.weight",
          new long[] {DIM, EXPERTS},
          matrix(EXPERTS, DIM, 110 + layer));
      builder.f32(
          prefix + "ffn_gate_exps.weight",
          new long[] {DIM, EXPERT_HIDDEN, EXPERTS},
          matrix(EXPERTS * EXPERT_HIDDEN, DIM, 120 + layer));
      builder.f32(
          prefix + "ffn_up_exps.weight",
          new long[] {DIM, EXPERT_HIDDEN, EXPERTS},
          matrix(EXPERTS * EXPERT_HIDDEN, DIM, 130 + layer));
      builder.f32(
          prefix + "ffn_down_exps.weight",
          new long[] {EXPERT_HIDDEN, DIM, EXPERTS},
          matrix(EXPERTS * DIM, EXPERT_HIDDEN, 140 + layer));
      int shared = config.sharedExpertHiddenDim();
      builder.f32(
          prefix + "ffn_gate_shexp.weight",
          new long[] {DIM, shared},
          matrix(shared, DIM, 150 + layer));
      builder.f32(
          prefix + "ffn_up_shexp.weight",
          new long[] {DIM, shared},
          matrix(shared, DIM, 160 + layer));
      builder.f32(
          prefix + "ffn_down_shexp.weight",
          new long[] {shared, DIM},
          matrix(DIM, shared, 170 + layer));
    }
    return new Deepseek2ToyModel(builder.build(), Map.copyOf(builder.values));
  }

  static float[] matrix(int rows, int columns, int seed) {
    float[] result = new float[rows * columns];
    for (int index = 0; index < result.length; index++) {
      result[index] = (((seed + index * 7) % 13) - 6) * 0.05f;
    }
    return result;
  }

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

    private Builder(Map<String, float[]> overrides) {
      this.overrides = overrides;
    }

    private void f32(String name, long[] shape, float[] declared) {
      float[] tensorValues = overrides.getOrDefault(name, declared);
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
