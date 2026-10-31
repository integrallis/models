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
 * mellum's declared header, as read off the published artifact.
 *
 * <p>Range-fetched from {@code Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf}
 * (JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M) on 2026-10-31 with the repo's own GGUF
 * inspector: 43 keys, 339 tensors, 8,071,295,040 bytes, {@code general.license = apache-2.0}.
 *
 * <p><b>This is a different artifact from the one the 2026-10-22 intake probed.</b> That one was
 * {@code Mellum2-12B-A2.5B-Instruct-Q8_0.gguf} from the {@code -Instruct-GGUF-Q8_0} repository,
 * 12.9 GB. #281 plans against the 8.07 GB Q4_K_M, which is the <b>Thinking</b> finetune from a
 * different repository and whose header had only ever been <b>sized</b>, never read — the same
 * "sized but never probed" gap that intake's README flags for parakeet's 110M sibling. The keys
 * below are the ones #281's artifact actually declares.
 *
 * <p><b>It declares its sliding-window pattern explicitly</b>, as {@code arr[bool,28]} at {@code
 * [true, true, true, false]} repeated seven times. An earlier version of this fixture said it did
 * not, and built the tests around a stride default instead. That was an artefact of the probe:
 * {@code inspectGguf} skips every array-typed key not named in {@code retainMetadataArrays}
 * (gguf-inspector.mjs:88-93) and that option defaults to empty, so the first read surfaced no
 * arrays at all and the key looked absent. Re-probed with the arrays retained, and with surfaced
 * key count reconciled against {@code kv_count} (43 of 43), the pattern is there.
 *
 * <p>The stride still matters, because upstream reaches for it when a file does <b>not</b> declare
 * a pattern: {@code load_swa_pattern(ml, 4)} in {@code src/models/mellum.cpp} passes <b>4</b>,
 * where the generic default in {@code LlamaConfig} is 6. This artifact never takes that path — a
 * declared array wins — so the stride is a safety net for a future mellum checkpoint and not this
 * one's shape.
 *
 * <p><b>{@code expert_weights_norm} is genuinely absent</b>, and that one is reportable because the
 * key count reconciles. Upstream sets it {@code true} for mellum before the optional read ({@code
 * src/models/mellum.cpp}), where {@code llama-hparams.h}'s default is false.
 *
 * <p>Dimensions are the artifact's own; the vocabulary is shrunk to keep the tests cheap, as in
 * {@link MapleTestHeaders}.
 */
public final class MellumTestHeaders {

  /** mellum's block count, and the length any per-layer array would have to match. */
  public static final int LAYERS = 28;

  /** The stride upstream hard-codes for mellum: one full-attention layer every fourth. */
  public static final int SWA_STRIDE = 4;

  private MellumTestHeaders() {}

  /**
   * @param withYarn declare the YaRN scaling block; false is the shape of a checkpoint that extends
   *     context by frequency base alone, which this artifact does not
   * @param withPattern declare {@code attention.sliding_window_pattern}; <b>this artifact does</b>,
   *     so false is the shape of a checkpoint that leaves the stride to a default
   */
  public static GgufMetadata metadata(boolean withYarn, boolean withPattern) {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("general.architecture", new GgufMetadataValue.StringValue("mellum"));
    entries.put("mellum.block_count", new GgufMetadataValue.Uint32Value(LAYERS));
    entries.put("mellum.embedding_length", new GgufMetadataValue.Uint32Value(2304));
    entries.put("mellum.feed_forward_length", new GgufMetadataValue.Uint32Value(7168));
    entries.put("mellum.attention.head_count", new GgufMetadataValue.Uint32Value(32));
    entries.put("mellum.attention.head_count_kv", new GgufMetadataValue.Uint32Value(4));
    entries.put("mellum.attention.key_length", new GgufMetadataValue.Uint32Value(128));
    entries.put("mellum.attention.value_length", new GgufMetadataValue.Uint32Value(128));
    entries.put("mellum.context_length", new GgufMetadataValue.Uint32Value(131072));
    entries.put("mellum.vocab_size", new GgufMetadataValue.Uint32Value(128));
    entries.put("mellum.rope.freq_base", new GgufMetadataValue.Float32Value(500000.0f));
    entries.put("mellum.rope.freq_base_swa", new GgufMetadataValue.Float32Value(500000.0f));
    entries.put(
        "mellum.attention.layer_norm_rms_epsilon", new GgufMetadataValue.Float32Value(1.0e-6f));
    entries.put("mellum.attention.sliding_window", new GgufMetadataValue.Uint32Value(1024));
    entries.put("mellum.expert_count", new GgufMetadataValue.Uint32Value(64));
    entries.put("mellum.expert_used_count", new GgufMetadataValue.Uint32Value(8));
    entries.put("mellum.expert_feed_forward_length", new GgufMetadataValue.Uint32Value(896));
    if (withYarn) {
      entries.put("mellum.rope.scaling.type", new GgufMetadataValue.StringValue("yarn"));
      entries.put("mellum.rope.scaling.factor", new GgufMetadataValue.Float32Value(16.0f));
      entries.put(
          "mellum.rope.scaling.original_context_length", new GgufMetadataValue.Uint32Value(8192));
      entries.put(
          "mellum.rope.scaling.yarn_attn_factor",
          new GgufMetadataValue.Float32Value(1.2772588729858398f));
      entries.put("mellum.rope.scaling.yarn_beta_fast", new GgufMetadataValue.Float32Value(32.0f));
      entries.put("mellum.rope.scaling.yarn_beta_slow", new GgufMetadataValue.Float32Value(1.0f));
    }
    if (withPattern) {
      List<GgufMetadataValue> pattern = new ArrayList<>();
      for (int layer = 0; layer < LAYERS; layer++) {
        pattern.add(new GgufMetadataValue.BoolValue(isLocal(layer)));
      }
      entries.put(
          "mellum.attention.sliding_window_pattern",
          new GgufMetadataValue.ArrayValue(GgufValueType.BOOL, pattern));
    }
    return new GgufMetadata(entries);
  }

  /** The artifact's own shape: the declared pattern present. */
  public static GgufMetadata metadata(boolean withYarn) {
    return metadata(withYarn, true);
  }

  /**
   * Whether the layer attends locally, under upstream's hard-coded stride of 4.
   *
   * <p>Upstream's rule, transcribed rather than paraphrased: {@code set_swa_pattern(n, dense_first
   * = false)} assigns {@code is_swa[il] = il % n < (n - 1)} ({@code src/llama-hparams.cpp:14-16}).
   * At {@code n = 4} that makes layers 0, 1 and 2 local and layer 3 global, repeating — the same
   * shape maple declares explicitly, reached here from a hard-coded default instead of from a key.
   */
  public static boolean isLocal(int layer) {
    return layer % SWA_STRIDE < SWA_STRIDE - 1;
  }

  /** How many of the 28 layers the stride marks local. */
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
