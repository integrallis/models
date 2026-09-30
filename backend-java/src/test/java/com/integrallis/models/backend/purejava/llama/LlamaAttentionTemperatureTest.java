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
package com.integrallis.models.backend.purejava.llama;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Mistral 3's position-dependent attention temperature scaling.
 *
 * <p>The formula is {@code log(floor(pos / floorScale) + 1) * scale + 1}, so it is <b>exactly 1.0
 * for every position below the floor</b>. That makes the obvious test -- run a short prompt and
 * compare -- pass while exercising nothing, which is why {@link #ORIGINAL_CONTEXT} here is 4 and
 * the sequences deliberately run past it. Ministral-3-3B declares a floor of 16384 inside a
 * 262144-token window, so on any realistic benchmark prompt this feature is inert; it is
 * implemented for long- context correctness, not to change short-prompt output.
 */
@Tag("unit")
class LlamaAttentionTemperatureTest {

  // 32, not 16: the quantized arm needs every quantized row length to be a multiple of the 32-value
  // Q8_0 block, and the row length here is DIM.
  private static final int DIM = 32;
  private static final int HEADS = 2;
  private static final int KV_HEADS = 1;
  private static final int HEAD_DIM = DIM / HEADS;
  private static final int KV_DIM = KV_HEADS * HEAD_DIM;
  private static final int HIDDEN_DIM = 32;
  private static final int VOCAB_SIZE = 32;
  private static final int LAYERS = 2;
  private static final int CONTEXT = 64;

  /** Small on purpose: the scaling only does anything at positions at or above this. */
  private static final int ORIGINAL_CONTEXT = 4;

  private static final float TEMP_SCALE = 0.1f;
  private static final String ARCH = "mistral3";

  @Test
  void mistral3IsPlainLlamaShapedWithInterleavedRope() {
    GgufFile file = build(sharedTensors(new Random(3)), TEMP_SCALE);
    LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());

    assertThat(config.architecture()).isEqualTo(DecoderArchitecture.MISTRAL3);
    assertThat(config.usesNeoxRope())
        .describedAs("llama.cpp classifies MISTRAL3 as LLAMA_ROPE_TYPE_NORM, the interleaved pair")
        .isFalse();
    assertThat(config.usesMixtureOfExperts()).isFalse();
    assertThat(config.usesPostAttentionNorm()).isFalse();
    assertThat(config.usesPostFfnNorm()).isFalse();
    assertThat(config.usesAttentionTemperatureScaling()).isTrue();
    assertThat(
            LlamaConfig.fromMetadata(build(sharedTensors(new Random(3)), 0.0f).metadata())
                .usesAttentionTemperatureScaling())
        .describedAs("absent key means no scaling, not a default of one")
        .isFalse();
  }

  @Test
  void theScaleIsExactlyOneBelowTheFloorAndGrowsInStepsAboveIt() {
    LlamaConfig config =
        LlamaConfig.fromMetadata(build(sharedTensors(new Random(11)), TEMP_SCALE).metadata());

    // Below the floor floor(pos/4) is 0 and log(1) is 0, so the scale is exactly one -- not
    // approximately one. Anything that multiplies by it is bit-identical.
    for (int position = 0; position < ORIGINAL_CONTEXT; position++) {
      assertThat(config.attentionTemperatureScale(position))
          .describedAs("position %s is below the floor", position)
          .isEqualTo(1.0f);
    }
    // And it steps, not ramps: every position in a block of ORIGINAL_CONTEXT shares one value.
    assertThat(config.attentionTemperatureScale(4))
        .isEqualTo((float) (Math.log(2.0) * TEMP_SCALE + 1.0), within(1.0e-7f));
    assertThat(config.attentionTemperatureScale(7)).isEqualTo(config.attentionTemperatureScale(4));
    assertThat(config.attentionTemperatureScale(8))
        .isEqualTo((float) (Math.log(3.0) * TEMP_SCALE + 1.0), within(1.0e-7f));
    assertThat(config.attentionTemperatureScale(12))
        .isEqualTo((float) (Math.log(4.0) * TEMP_SCALE + 1.0), within(1.0e-7f));
    assertThat(config.attentionTemperatureScale(8))
        .isGreaterThan(config.attentionTemperatureScale(4));
  }

  @Test
  void aModelWithoutTheKeyScalesByOneAtEveryPosition() {
    LlamaConfig config =
        LlamaConfig.fromMetadata(build(sharedTensors(new Random(11)), 0.0f).metadata());

    for (int position : new int[] {0, 1, 4, 9, 40, 63}) {
      assertThat(config.attentionTemperatureScale(position)).isEqualTo(1.0f);
    }
  }

  @Test
  void belowTheFloorTheOutputIsBitIdenticalWithAndWithoutScaling() {
    Map<String, byte[]> shared = sharedTensors(new Random(77));
    GgufFile plain = build(shared, 0.0f);
    GgufFile scaled = build(shared, TEMP_SCALE);

    // Positions 0..3 are all below the floor, so the multiplier is exactly 1.0f and the two arms
    // must agree bit for bit. This is what guarantees the feature cannot perturb ordinary decoding.
    int[] tokens = {3, 5, 7, 9};
    float[] withoutScaling = runSequence(plain, tokens);
    float[] withScaling = runSequence(scaled, tokens);

    assertThat(withScaling).containsExactly(withoutScaling);
  }

  @Test
  void atAndAboveTheFloorTheOutputActuallyChanges() {
    Map<String, byte[]> shared = sharedTensors(new Random(77));
    GgufFile plain = build(shared, 0.0f);
    GgufFile scaled = build(shared, TEMP_SCALE);

    // Eight tokens reach positions 4..7, where the scale is log(2)*0.1+1. Without this case the
    // whole suite would pass with scaleQueryTemperature deleted.
    int[] tokens = {3, 5, 7, 9, 11, 13, 15, 17};
    float[] withoutScaling = runSequence(plain, tokens);
    float[] withScaling = runSequence(scaled, tokens);

    assertThat(withScaling)
        .describedAs("crossing the floor must change the logits")
        .isNotEqualTo(withoutScaling);
  }

  @Test
  void batchedPrefillAgreesWithTokenByTokenAcrossTheFloor() {
    Map<String, byte[]> shared = sharedTensors(new Random(5), GgufTensorType.Q8_0);
    GgufFile scaled = build(shared, TEMP_SCALE, GgufTensorType.Q8_0);
    LlamaConfig config = LlamaConfig.fromMetadata(scaled.metadata());

    int[] tokens = {2, 4, 6, 8, 10, 12};
    LlamaForwardPass batchedPass = newForwardPass(scaled, config);
    assertThat(batchedPass.usesBatchedPrefill())
        .describedAs("F32 is not a batched-matmul type, so this must be quantized to batch at all")
        .isTrue();
    float[] serial = runSequence(scaled, tokens);
    float[] batched = batchedPass.prefill(tokens, 0);

    // The batched path scales per row from startPosition + batch. If it used one position for the
    // whole batch, rows either side of the floor would be wrong.
    assertThat(batched).hasSameSizeAs(serial);
    for (int index = 0; index < serial.length; index++) {
      assertThat(batched[index]).isEqualTo(serial[index], within(1.0e-4f));
    }
  }

  private static LlamaForwardPass newForwardPass(GgufFile file, LlamaConfig config) {
    return new LlamaForwardPass(
        config,
        LlamaWeights.fromGgufFile(file, config),
        new KvCache(
            config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()));
  }

  private static float[] runSequence(GgufFile file, int[] tokens) {
    LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
    LlamaForwardPass pass = newForwardPass(file, config);
    float[] logits = null;
    for (int position = 0; position < tokens.length; position++) {
      logits = pass.forward(tokens[position], position);
    }
    return logits.clone();
  }

  private static Map<String, byte[]> sharedTensors(Random rng) {
    return sharedTensors(rng, GgufTensorType.F32);
  }

  /** Every tensor both arms share, so the only difference between them is the one metadata key. */
  private static Map<String, byte[]> sharedTensors(Random rng, GgufTensorType type) {
    Map<String, byte[]> tensors = new LinkedHashMap<>();
    tensors.put("token_embd.weight", randomF32(rng, VOCAB_SIZE * DIM));
    tensors.put("output_norm.weight", onesF32(DIM));
    tensors.put("output.weight", randomF32(rng, VOCAB_SIZE * DIM));
    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      tensors.put(prefix + "attn_norm.weight", onesF32(DIM));
      tensors.put(prefix + "attn_q.weight", projection(rng, type, DIM * DIM));
      tensors.put(prefix + "attn_k.weight", projection(rng, type, KV_DIM * DIM));
      tensors.put(prefix + "attn_v.weight", projection(rng, type, KV_DIM * DIM));
      tensors.put(prefix + "attn_output.weight", projection(rng, type, DIM * DIM));
      tensors.put(prefix + "ffn_norm.weight", onesF32(DIM));
      tensors.put(prefix + "ffn_gate.weight", projection(rng, type, HIDDEN_DIM * DIM));
      tensors.put(prefix + "ffn_up.weight", projection(rng, type, HIDDEN_DIM * DIM));
      tensors.put(prefix + "ffn_down.weight", projection(rng, type, DIM * HIDDEN_DIM));
    }
    return tensors;
  }

  private static byte[] projection(Random rng, GgufTensorType type, int valueCount) {
    if (type == GgufTensorType.F32) {
      return randomF32(rng, valueCount);
    }
    byte[] data = new byte[(valueCount / 32) * 34];
    rng.nextBytes(data);
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < valueCount / 32; block++) {
      buffer.putShort(block * 34, Float.floatToFloat16(0.01f));
    }
    return data;
  }

  private static GgufFile build(Map<String, byte[]> shared, float tempScale) {
    return build(shared, tempScale, GgufTensorType.F32);
  }

  private static GgufFile build(Map<String, byte[]> shared, float tempScale, GgufTensorType type) {
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", ARCH)
            .addUint32(ARCH + ".embedding_length", DIM)
            .addUint32(ARCH + ".block_count", LAYERS)
            .addUint32(ARCH + ".attention.head_count", HEADS)
            .addUint32(ARCH + ".attention.head_count_kv", KV_HEADS)
            .addUint32(ARCH + ".vocab_size", VOCAB_SIZE)
            .addUint32(ARCH + ".context_length", CONTEXT)
            .addUint32(ARCH + ".feed_forward_length", HIDDEN_DIM)
            // YaRN, as Ministral-3 declares it. Both arms carry it identically, so it is not what
            // the comparisons are measuring.
            .addString(ARCH + ".rope.scaling.type", "yarn")
            .addFloat32(ARCH + ".rope.scaling.factor", 2.0f)
            .addUint32(ARCH + ".rope.scaling.original_context_length", ORIGINAL_CONTEXT);
    if (tempScale != 0.0f) {
      builder.addFloat32(ARCH + ".attention.temperature_scale", tempScale);
    }

    builder.addTensor(
        "token_embd.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        shared.get("token_embd.weight"));
    builder.addTensor(
        "output_norm.weight",
        GgufTensorType.F32,
        new long[] {DIM},
        shared.get("output_norm.weight"));
    builder.addTensor(
        "output.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        shared.get("output.weight"));

    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      builder.addTensor(
          prefix + "attn_norm.weight",
          GgufTensorType.F32,
          new long[] {DIM},
          shared.get(prefix + "attn_norm.weight"));
      builder.addTensor(
          prefix + "attn_q.weight",
          type,
          new long[] {DIM, DIM},
          shared.get(prefix + "attn_q.weight"));
      builder.addTensor(
          prefix + "attn_k.weight",
          type,
          new long[] {DIM, KV_DIM},
          shared.get(prefix + "attn_k.weight"));
      builder.addTensor(
          prefix + "attn_v.weight",
          type,
          new long[] {DIM, KV_DIM},
          shared.get(prefix + "attn_v.weight"));
      builder.addTensor(
          prefix + "attn_output.weight",
          type,
          new long[] {DIM, DIM},
          shared.get(prefix + "attn_output.weight"));
      builder.addTensor(
          prefix + "ffn_norm.weight",
          GgufTensorType.F32,
          new long[] {DIM},
          shared.get(prefix + "ffn_norm.weight"));
      builder.addTensor(
          prefix + "ffn_gate.weight",
          type,
          new long[] {DIM, HIDDEN_DIM},
          shared.get(prefix + "ffn_gate.weight"));
      builder.addTensor(
          prefix + "ffn_up.weight",
          type,
          new long[] {DIM, HIDDEN_DIM},
          shared.get(prefix + "ffn_up.weight"));
      builder.addTensor(
          prefix + "ffn_down.weight",
          type,
          new long[] {HIDDEN_DIM, DIM},
          shared.get(prefix + "ffn_down.weight"));
    }

    byte[] data = builder.build();
    MemorySegment segment = Arena.ofConfined().allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return GgufParser.parseSegment(segment);
  }

  private static byte[] randomF32(Random rng, int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < count; i++) {
      buf.putFloat(i * 4, (rng.nextFloat() - 0.5f) * 0.1f);
    }
    return data;
  }

  private static byte[] onesF32(int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < count; i++) {
      buf.putFloat(i * 4, 1.0f);
    }
    return data;
  }
}
