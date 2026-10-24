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
 * maple's declared header, as read off the published artifact.
 *
 * <p>Transcribed from maple-preview-TQ1_0-head-Q4_K.gguf (deepgrove/maple-preview-GGUF, revision
 * f5466f918e0c50cdb9d4d47a6f35813509a42a30) on 2026-10-24: {@code attention.sliding_window} 512,
 * {@code attention.sliding_window_pattern} as {@code arr[bool,24]} at {@code [true, true, true,
 * false]} repeated six times, and {@code swiglu_clamp_exp} as {@code arr[f32,24]} at 7.0. The
 * oracle prints the same three keys when it loads the file.
 *
 * <p>Shared rather than copied because two tests need the same shape, and a header fixture that
 * drifts between its copies is worse than no fixture: one of them keeps passing against a model
 * that no longer exists. The dimensions are shrunk to keep the tests cheap; every key that decides
 * attention or the expert FFN is the artifact's own.
 */
public final class MapleTestHeaders {

  /** maple's block count, which both declared arrays must match in length. */
  public static final int LAYERS = 24;

  private MapleTestHeaders() {}

  /**
   * @param withPattern declare {@code attention.sliding_window_pattern}; maple always does, so
   *     false is the shape of a truncated or absent key
   * @param withClamp declare {@code swiglu_clamp_exp}; false is every artifact published before
   *     maple
   */
  public static GgufMetadata metadata(boolean withPattern, boolean withClamp) {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("maple"));
    entries.put("maple.embedding_length", new GgufMetadataValue.Uint32Value(2048));
    entries.put("maple.block_count", new GgufMetadataValue.Uint32Value(LAYERS));
    entries.put("maple.attention.head_count", new GgufMetadataValue.Uint32Value(16));
    entries.put("maple.attention.head_count_kv", new GgufMetadataValue.Uint32Value(4));
    entries.put("maple.attention.key_length", new GgufMetadataValue.Uint32Value(128));
    entries.put("maple.attention.value_length", new GgufMetadataValue.Uint32Value(128));
    entries.put("maple.vocab_size", new GgufMetadataValue.Uint32Value(128));
    entries.put("maple.context_length", new GgufMetadataValue.Uint32Value(4096));
    entries.put("maple.feed_forward_length", new GgufMetadataValue.Uint32Value(4096));
    entries.put("maple.rope.freq_base", new GgufMetadataValue.Float32Value(10000.0f));
    entries.put("maple.rope.dimension_count", new GgufMetadataValue.Uint32Value(64));
    entries.put(
        "maple.attention.layer_norm_rms_epsilon", new GgufMetadataValue.Float32Value(1.0e-6f));
    entries.put("maple.attention.sliding_window", new GgufMetadataValue.Uint32Value(512));
    entries.put("maple.expert_count", new GgufMetadataValue.Uint32Value(256));
    entries.put("maple.expert_used_count", new GgufMetadataValue.Uint32Value(8));
    entries.put("maple.expert_feed_forward_length", new GgufMetadataValue.Uint32Value(512));
    entries.put("maple.expert_gating_func", new GgufMetadataValue.Uint32Value(1));
    if (withPattern) {
      List<GgufMetadataValue> pattern = new ArrayList<>();
      for (int layer = 0; layer < LAYERS; layer++) {
        pattern.add(new GgufMetadataValue.BoolValue(isLocal(layer)));
      }
      entries.put(
          "maple.attention.sliding_window_pattern",
          new GgufMetadataValue.ArrayValue(GgufValueType.BOOL, pattern));
    }
    if (withClamp) {
      List<GgufMetadataValue> clamps = new ArrayList<>();
      for (int layer = 0; layer < LAYERS; layer++) {
        clamps.add(new GgufMetadataValue.Float32Value(7.0f));
      }
      entries.put(
          "maple.swiglu_clamp_exp",
          new GgufMetadataValue.ArrayValue(GgufValueType.FLOAT32, clamps));
    }
    return new GgufMetadata(entries);
  }

  /** Three local layers then one global, six times over: the artifact's declared pattern. */
  public static boolean isLocal(int layer) {
    return layer % 4 != 3;
  }

  /** How many of the 24 layers the declared pattern marks local. */
  public static int localLayers() {
    int local = 0;
    for (int layer = 0; layer < LAYERS; layer++) {
      if (isLocal(layer)) {
        local++;
      }
    }
    return local;
  }
}
