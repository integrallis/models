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
package com.integrallis.models.backend.purejava.lfm3;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.List;
import java.util.Objects;

/** Artifact-declared convolution, global-attention and sliding-attention layout. */
public record Lfm3Config(
    int embeddingDim,
    int numLayers,
    int numHeads,
    List<Integer> kvHeadsByLayer,
    List<Boolean> slidingWindowByLayer,
    int headDim,
    int vocabSize,
    int contextLength,
    int hiddenDim,
    float ropeTheta,
    float slidingRopeTheta,
    float rmsNormEpsilon,
    int shortConvCache,
    int slidingWindow) {
  public Lfm3Config {
    for (int value :
        new int[] {
          embeddingDim,
          numLayers,
          numHeads,
          headDim,
          vocabSize,
          contextLength,
          hiddenDim,
          slidingWindow
        }) {
      if (value <= 0) throw new IllegalArgumentException("LFM3 dimensions must be positive");
    }
    if (headDim % 2 != 0 || shortConvCache < 2)
      throw new IllegalArgumentException("invalid LFM3 rotary/convolution dimensions");
    for (float value : new float[] {ropeTheta, slidingRopeTheta, rmsNormEpsilon}) {
      if (!(value > 0) || !Float.isFinite(value))
        throw new IllegalArgumentException("invalid LFM3 numerical metadata");
    }
    kvHeadsByLayer = List.copyOf(kvHeadsByLayer);
    slidingWindowByLayer = List.copyOf(slidingWindowByLayer);
    if (kvHeadsByLayer.size() != numLayers || slidingWindowByLayer.size() != numLayers)
      throw new IllegalArgumentException("LFM3 requires per-layer attention metadata");
    boolean attention = false, conv = false, swa = false;
    for (int i = 0; i < numLayers; i++) {
      int kv = kvHeadsByLayer.get(i);
      if (kv < 0 || (kv > 0 && numHeads % kv != 0) || (kv == 0 && slidingWindowByLayer.get(i)))
        throw new IllegalArgumentException("invalid LFM3 layer " + i);
      attention |= kv > 0;
      conv |= kv == 0;
      swa |= slidingWindowByLayer.get(i);
    }
    if (!attention || !conv || !swa)
      throw new IllegalArgumentException("LFM3 requires convolution and sliding attention");
  }

  public static Lfm3Config fromMetadata(GgufMetadata m) {
    Objects.requireNonNull(m, "metadata");
    if (!m.getString("general.architecture").orElse("").equals("lfm3"))
      throw new IllegalArgumentException("Expected general.architecture=lfm3");
    int dim = integer(m, "embedding_length"), heads = integer(m, "attention.head_count");
    int headDim = m.getUint32("lfm3.attention.key_length").orElse(dim / heads);
    if (m.getUint32("lfm3.attention.value_length").orElse(headDim) != headDim
        || m.getUint32("lfm3.rope.dimension_count").orElse(headDim) != headDim)
      throw new IllegalArgumentException(
          "LFM3 requires equal K/V widths and full-head rotary layout");
    return new Lfm3Config(
        dim,
        integer(m, "block_count"),
        heads,
        m.getInt32Array("lfm3.attention.head_count_kv")
            .orElseThrow(() -> new IllegalArgumentException("Missing per-layer KV heads")),
        m.getBoolArray("lfm3.attention.sliding_window_pattern")
            .orElseThrow(
                () -> new IllegalArgumentException("Missing per-layer sliding-window pattern")),
        headDim,
        m.getUint32("lfm3.vocab_size")
            .or(() -> m.getArraySize("tokenizer.ggml.tokens"))
            .orElseThrow(() -> new IllegalArgumentException("Missing vocabulary")),
        integer(m, "context_length"),
        integer(m, "feed_forward_length"),
        number(m, "rope.freq_base"),
        number(m, "rope.freq_base_swa"),
        number(m, "attention.layer_norm_rms_epsilon"),
        integer(m, "shortconv.l_cache"),
        integer(m, "attention.sliding_window"));
  }

  private static int integer(GgufMetadata m, String key) {
    return m.getUint32("lfm3." + key)
        .orElseThrow(() -> new IllegalArgumentException("Missing lfm3." + key));
  }

  private static float number(GgufMetadata m, String key) {
    return m.getFloat32("lfm3." + key)
        .orElseThrow(() -> new IllegalArgumentException("Missing lfm3." + key));
  }

  public boolean usesAttention(int layer) {
    return kvHeadsByLayer.get(layer) > 0;
  }

  public boolean usesSlidingWindow(int layer) {
    return slidingWindowByLayer.get(layer);
  }

  public int numKvHeads(int layer) {
    return kvHeadsByLayer.get(layer);
  }

  public int keyDim(int layer) {
    return numKvHeads(layer) * headDim;
  }

  public int queryDim() {
    return numHeads * headDim;
  }

  public int shortConvStateSize() {
    return Math.multiplyExact(shortConvCache - 1, embeddingDim);
  }
}
