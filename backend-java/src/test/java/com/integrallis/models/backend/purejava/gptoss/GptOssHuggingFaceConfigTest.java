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
package com.integrallis.models.backend.purejava.gptoss;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class GptOssHuggingFaceConfigTest {

  @Test
  void parsesThePinnedOfficial20BExecutionShape() throws Exception {
    Path path =
        Path.of(
            GptOssHuggingFaceConfigTest.class
                .getResource("/huggingface/gpt-oss-20b-config.json")
                .toURI());

    // openai/gpt-oss-20b at revision 6cee5e81ee83917806bbde320786a8fb61efebee.
    assertThat(sha256(path))
        .isEqualTo("3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce");
    GptOssHuggingFaceConfig config = GptOssHuggingFaceConfig.parse(path);

    assertThat(config.architectures()).containsExactly("GptOssForCausalLM");
    assertThat(config.hiddenSize()).isEqualTo(2_880);
    assertThat(config.numLayers()).isEqualTo(24);
    assertThat(config.numHeads()).isEqualTo(64);
    assertThat(config.numKvHeads()).isEqualTo(8);
    assertThat(config.headDim()).isEqualTo(64);
    assertThat(config.queryDimension()).isEqualTo(4_096);
    assertThat(config.keyValueDimension()).isEqualTo(512);
    assertThat(config.vocabSize()).isEqualTo(201_088);
    assertThat(config.maxPosition()).isEqualTo(131_072);
    assertThat(config.initialContextLength()).isEqualTo(4_096);
    assertThat(config.intermediateSize()).isEqualTo(2_880);
    assertThat(config.numExperts()).isEqualTo(32);
    assertThat(config.expertsPerToken()).isEqualTo(4);
    assertThat(config.slidingWindow()).isEqualTo(128);
    assertThat(config.swigluLimit()).isEqualTo(7.0f);
    assertThat(config.hiddenActAlpha()).isEqualTo(1.702f);
    assertThat(config.ropeTheta()).isEqualTo(150_000.0f);
    assertThat(config.ropeYarnFactor()).isEqualTo(32.0f);
    assertThat(config.ropeOriginalContext()).isEqualTo(4_096);
    assertThat(config.quantizationMethod()).isEqualTo("mxfp4");
    assertThat(config.attentionBias()).isTrue();
    assertThat(config.tieWordEmbeddings()).isFalse();
    assertThat(config.eosTokenId()).isEqualTo(200_002);
    assertThat(config.padTokenId()).isEqualTo(199_999);
    assertThat(config.usesSlidingAttention(0)).isTrue();
    assertThat(config.usesSlidingAttention(1)).isFalse();
    assertThat(config.attentionStartPosition(0, 300)).isEqualTo(173);
    assertThat(config.attentionStartPosition(1, 300)).isZero();
  }

  @Test
  void rejectsUnsupportedQuantizationAndMalformedLayerSchedule(@TempDir Path directory)
      throws IOException {
    Path wrongQuantization = directory.resolve("wrong-quantization.json");
    Files.writeString(wrongQuantization, minimalConfig("bf16", "[\"full_attention\"]"));
    Path wrongSchedule = directory.resolve("wrong-schedule.json");
    Files.writeString(wrongSchedule, minimalConfig("mxfp4", "[\"cross_attention\"]"));

    assertThatThrownBy(() -> GptOssHuggingFaceConfig.parse(wrongQuantization))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("mxfp4");
    assertThatThrownBy(() -> GptOssHuggingFaceConfig.parse(wrongSchedule))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("layer_types");
  }

  @Test
  void rejectsDuplicateConfigFields(@TempDir Path directory) throws IOException {
    Path duplicate = directory.resolve("duplicate.json");
    Files.writeString(duplicate, "{\"hidden_size\":32,\"hidden_size\":32}");

    assertThatThrownBy(() -> GptOssHuggingFaceConfig.parse(duplicate))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("duplicate config JSON");
  }

  private static String minimalConfig(String quantization, String layerTypes) {
    return """
        {
          "architectures": ["GptOssForCausalLM"],
          "attention_bias": true,
          "eos_token_id": 2,
          "head_dim": 32,
          "hidden_act": "silu",
          "hidden_size": 32,
          "initial_context_length": 128,
          "intermediate_size": 32,
          "layer_types": %s,
          "max_position_embeddings": 256,
          "model_type": "gpt_oss",
          "num_attention_heads": 2,
          "num_experts_per_tok": 1,
          "num_hidden_layers": 1,
          "num_key_value_heads": 1,
          "num_local_experts": 1,
          "pad_token_id": 0,
          "quantization_config": {"quant_method": "%s"},
          "rms_norm_eps": 0.00001,
          "rope_scaling": {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 2.0,
            "original_max_position_embeddings": 128,
            "rope_type": "yarn",
            "truncate": false
          },
          "rope_theta": 10000,
          "sliding_window": 64,
          "swiglu_limit": 7.0,
          "tie_word_embeddings": false,
          "vocab_size": 32
        }
        """
        .formatted(layerTypes, quantization);
  }

  private static String sha256(Path path) throws IOException, NoSuchAlgorithmException {
    return HexFormat.of()
        .formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)));
  }
}
