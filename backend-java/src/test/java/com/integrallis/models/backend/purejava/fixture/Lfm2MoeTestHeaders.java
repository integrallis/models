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
package com.integrallis.models.backend.purejava.fixture;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code lfm2moe}'s declared header, as read off the published artifact.
 *
 * <p>Range-fetched from {@code LFM2.5-8B-A1B-Q4_0.gguf} (LiquidAI/LFM2.5-8B-A1B-GGUF) on
 * 2026-10-31: 39 keys, 256 tensors, 4,844,678,368 bytes. The surfaced key count was reconciled
 * against {@code kv_count} (39 of 39) before any of this was trusted — see {@code
 * benchmark-results/2026-10-31-tier-d-header-intake/README.md} §0 for why that reconciliation is
 * mandatory and what reading absences off an unreconciled probe cost.
 *
 * <p><b>This is a different artifact from the one the 2026-10-22 intake probed.</b> That was the
 * Q8_0 at 9.01 GB; #283 plans against this Q4_0 at 4.84 GB, whose header had only ever been sized.
 *
 * <p><b>The key that decides the layer layout is an array</b>, which is why it took two reads to
 * see: {@code attention.head_count_kv} is {@code arr[int32,24]} at {@code [0,0,8,0]} repeated six
 * times, so <b>six layers attend and eighteen convolve</b>. Upstream derives the same split the
 * same way — {@code is_recr_impl[il] = n_head_kv(il) == 0} in {@code src/models/lfm2moe.cpp} — and
 * {@link com.integrallis.models.backend.purejava.lfm2.Lfm2Config} already reads this key per layer
 * and already accepts 0 as the non-attention marker, so {@code lfm2moe} needs no new mechanism for
 * it.
 *
 * <p>It reconciles independently with the tensor count: 256 tensors less two top-level ({@code
 * token_embd}, {@code token_embd_norm}) is 254 block tensors, and {@code 7a + 4(24 - a) + 2x4 +
 * 22x6 = 254} resolves only at {@code a = 6} attention blocks.
 *
 * <p>Dimensions are the artifact's own; the vocabulary is shrunk to keep the tests cheap, as in
 * {@link MapleTestHeaders}.
 */
public final class Lfm2MoeTestHeaders {

  /** The artifact's block count, and the length of the declared per-layer array. */
  public static final int LAYERS = 24;

  /** {@code leading_dense_block_count}: blocks 0 and 1 are dense, the rest routed. */
  public static final int LEADING_DENSE_BLOCKS = 2;

  private Lfm2MoeTestHeaders() {}

  /**
   * @param architecture {@code "lfm2moe"} for the artifact's own string, or {@code "lfm2"} to build
   *     the dense sibling's header for a negative control
   * @param withExperts declare the expert keys; false with {@code lfm2moe} is a malformed file —
   *     the architecture claims a mixture and then describes none
   */
  public static GgufMetadata metadata(String architecture, boolean withExperts) {
    return metadata(architecture, withExperts, SIGMOID_GATING);
  }

  /** llama.cpp's {@code LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID}, which this artifact declares. */
  public static final int SIGMOID_GATING = 2;

  /**
   * @param gatingFunction the declared {@code expert_gating_func}; anything but {@link
   *     #SIGMOID_GATING} builds a file the loader must refuse rather than approximate
   */
  public static GgufMetadata metadata(
      String architecture, boolean withExperts, int gatingFunction) {
    String prefix = architecture + ".";
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue(architecture));
    entries.put(prefix + "block_count", new GgufMetadataValue.Uint32Value(LAYERS));
    entries.put(prefix + "embedding_length", new GgufMetadataValue.Uint32Value(2048));
    entries.put(prefix + "feed_forward_length", new GgufMetadataValue.Uint32Value(7168));
    entries.put(prefix + "attention.head_count", new GgufMetadataValue.Uint32Value(32));
    entries.put(prefix + "context_length", new GgufMetadataValue.Uint32Value(128000));
    entries.put(prefix + "vocab_size", new GgufMetadataValue.Uint32Value(128));
    entries.put(prefix + "rope.freq_base", new GgufMetadataValue.Float32Value(5_000_000.0f));
    entries.put(
        prefix + "attention.layer_norm_rms_epsilon", new GgufMetadataValue.Float32Value(1.0e-5f));
    entries.put(prefix + "shortconv.l_cache", new GgufMetadataValue.Uint32Value(3));

    List<GgufMetadataValue> kvHeads = new ArrayList<>();
    for (int layer = 0; layer < LAYERS; layer++) {
      kvHeads.add(new GgufMetadataValue.Int32Value(attends(layer) ? 8 : 0));
    }
    entries.put(
        prefix + "attention.head_count_kv",
        new GgufMetadataValue.ArrayValue(GgufValueType.INT32, kvHeads));

    if (withExperts) {
      entries.put(prefix + "expert_count", new GgufMetadataValue.Uint32Value(32));
      entries.put(prefix + "expert_used_count", new GgufMetadataValue.Uint32Value(4));
      entries.put(prefix + "expert_feed_forward_length", new GgufMetadataValue.Uint32Value(1792));
      entries.put(
          prefix + "leading_dense_block_count",
          new GgufMetadataValue.Uint32Value(LEADING_DENSE_BLOCKS));
      // 2 is sigmoid. The artifact declares it, and this is the key #283 says must not be
      // approximated with softmax.
      entries.put(prefix + "expert_gating_func", new GgufMetadataValue.Uint32Value(gatingFunction));
    }
    return new GgufMetadata(entries);
  }

  /** The artifact's own shape. */
  public static GgufMetadata metadata() {
    return metadata("lfm2moe", true);
  }

  /**
   * Whether the layer attends, from the declared array: {@code [0,0,8,0]} repeated six times.
   *
   * <p>So layer 2 is the first attention layer and three convolutional ones surround it in each
   * group of four. Note this is <b>not</b> the same phase as a sliding-window stride, which marks
   * the <i>last</i> of each group — reading one as the other would put attention on the wrong six
   * layers while loading cleanly.
   */
  public static boolean attends(int layer) {
    return layer % 4 == 2;
  }

  /** How many of the 24 layers attend; the tensor count reconciles only at this value. */
  public static int attentionLayers() {
    int count = 0;
    for (int layer = 0; layer < LAYERS; layer++) {
      if (attends(layer)) {
        count++;
      }
    }
    return count;
  }

  /** Whether the layer's feed-forward is routed rather than dense. */
  public static boolean routed(int layer) {
    return layer >= LEADING_DENSE_BLOCKS;
  }
}
