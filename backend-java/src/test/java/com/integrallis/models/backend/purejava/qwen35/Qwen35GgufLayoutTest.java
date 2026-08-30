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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Pins the hybrid full-attention/Gated DeltaNet layout of the Qwen3.5 development fixture. */
@Tag("integration")
class Qwen35GgufLayoutTest {

  private static final ModelFixtureRequirement QWEN35_08B_Q4_K_M =
      ModelFixtureRequirement.of("hf://unsloth/Qwen3.5-0.8B-GGUF")
          .version("[3.5.0,3.6.0)")
          .variant("q4_k_m")
          .backend("parser")
          .capability("chat");

  @Test
  void matchesThePinnedHybridArchitecture() throws Exception {
    ModelFixtureDescriptor descriptor =
        ModelFixtureRegistry.fromClasspath().resolve(QWEN35_08B_Q4_K_M).orElseThrow();
    assertThat(Files.isRegularFile(descriptor.localPath().orElseThrow()))
        .as("Run downloadQwen3508BQ4KMGguf before this test")
        .isTrue();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);
      var metadata = file.metadata();

      assertThat(metadata.getString("general.architecture")).contains("qwen35");
      assertThat(metadata.getString("tokenizer.ggml.pre")).contains("qwen35");
      assertThat(metadata.getUint32("qwen35.block_count")).contains(24);
      assertThat(metadata.getUint32("qwen35.context_length")).contains(262_144);
      assertThat(metadata.getUint32("qwen35.embedding_length")).contains(1_024);
      assertThat(metadata.getUint32("qwen35.feed_forward_length")).contains(3_584);
      assertThat(metadata.getUint32("qwen35.attention.head_count")).contains(8);
      assertThat(metadata.getUint32("qwen35.attention.head_count_kv")).contains(2);
      assertThat(metadata.getUint32("qwen35.attention.key_length")).contains(256);
      assertThat(metadata.getUint32("qwen35.attention.value_length")).contains(256);
      assertThat(metadata.getUint32("qwen35.ssm.conv_kernel")).contains(4);
      assertThat(metadata.getUint32("qwen35.ssm.state_size")).contains(128);
      assertThat(metadata.getUint32("qwen35.ssm.group_count")).contains(16);
      assertThat(metadata.getUint32("qwen35.ssm.inner_size")).contains(2_048);
      assertThat(metadata.getUint32("qwen35.full_attention_interval")).contains(4);

      assertThat(file.tensorInfos()).hasSize(320);
      assertThat(file.tensorInfos())
          .extracting(tensor -> tensor.type())
          .contains(GgufTensorType.F32, GgufTensorType.Q4_K, GgufTensorType.Q6_K);
      assertThat(file.tensorInfos().stream().filter(tensor -> isFullAttention(tensor.name())))
          .extracting(tensor -> layer(tensor.name()))
          .containsExactlyInAnyOrder(3, 7, 11, 15, 19, 23);
      assertThat(file.tensorInfos().stream().filter(tensor -> isGatedDeltaNet(tensor.name())))
          .hasSize(18);
    }
  }

  private static boolean isFullAttention(String name) {
    return name.matches("blk\\.\\d+\\.attn_q\\.weight");
  }

  private static boolean isGatedDeltaNet(String name) {
    return name.matches("blk\\.\\d+\\.ssm_a");
  }

  private static int layer(String name) {
    return Integer.parseInt(name.substring(4, name.indexOf('.', 4)));
  }
}
