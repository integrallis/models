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
package com.integrallis.models.accelerator;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class KvTransferCompatibilityTest {

  @Test
  void acceptsMatchedQwenKvGeometryForCalibrationButNeverDirectReuse() {
    var source = qwen(1_024, List.of("<pad>", "hello", "world"));
    var target = qwen(2_048, List.of("<pad>", "hello", "world"));

    var result = KvTransferCompatibility.screen(source, target);

    assertThat(result.calibrationCandidate()).isTrue();
    assertThat(result.directReuseSafe()).isFalse();
    assertThat(result.source().embeddingLength()).isEqualTo(1_024);
    assertThat(result.target().embeddingLength()).isEqualTo(2_048);
    assertThat(result.reasons())
        .containsExactly(
            "matched 28-layer, 8-head, 128x128 KV geometry",
            "tokenizer vocabulary is identical",
            "raw KV values remain model-specific; calibration and quality gates are required");
  }

  @Test
  void rejectsDifferentKvGeometry() {
    var source = qwen(1_024, List.of("<pad>", "hello"));
    var entries = new LinkedHashMap<>(qwen(2_048, List.of("<pad>", "hello")).entries());
    entries.put("qwen3.attention.head_count_kv", uint(4));

    var result = KvTransferCompatibility.screen(source, new GgufMetadata(entries));

    assertThat(result.calibrationCandidate()).isFalse();
    assertThat(result.directReuseSafe()).isFalse();
    assertThat(result.reasons()).contains("KV head count differs: 8 != 4");
  }

  @Test
  void rejectsDifferentTokenizersForTheFirstLinearMappingExperiment() {
    var source = qwen(1_024, List.of("<pad>", "hello"));
    var target = qwen(2_048, List.of("<pad>", "different"));

    var result = KvTransferCompatibility.screen(source, target);

    assertThat(result.calibrationCandidate()).isFalse();
    assertThat(result.reasons())
        .contains("tokenizer vocabulary differs; aligned-token calibration is not valid");
  }

  private static GgufMetadata qwen(int embeddingLength, List<String> vocabulary) {
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put("general.architecture", string("qwen3"));
    entries.put("qwen3.block_count", uint(28));
    entries.put("qwen3.embedding_length", uint(embeddingLength));
    entries.put("qwen3.attention.head_count", uint(16));
    entries.put("qwen3.attention.head_count_kv", uint(8));
    entries.put("qwen3.attention.key_length", uint(128));
    entries.put("qwen3.attention.value_length", uint(128));
    entries.put("qwen3.rope.freq_base", new GgufMetadataValue.Float32Value(1_000_000.0f));
    entries.put(
        "tokenizer.ggml.tokens",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.STRING,
            vocabulary.stream()
                .<GgufMetadataValue>map(KvTransferCompatibilityTest::string)
                .toList()));
    return new GgufMetadata(entries);
  }

  private static GgufMetadataValue.Uint32Value uint(int value) {
    return new GgufMetadataValue.Uint32Value(value);
  }

  private static GgufMetadataValue.StringValue string(String value) {
    return new GgufMetadataValue.StringValue(value);
  }
}
