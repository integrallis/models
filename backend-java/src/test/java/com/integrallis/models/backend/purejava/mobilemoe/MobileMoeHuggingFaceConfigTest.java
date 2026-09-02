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
package com.integrallis.models.backend.purejava.mobilemoe;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class MobileMoeHuggingFaceConfigTest {

  @Test
  void parsesThePinnedSmallQatExecutionContract(@TempDir Path directory) throws Exception {
    Path config = directory.resolve("config.json");
    Files.writeString(config, smallConfig("mobilemoe-int4-g32", true));

    assertThat(MobileMoeHuggingFaceConfig.matches(config)).isTrue();
    MobileMoeHuggingFaceConfig parsed = MobileMoeHuggingFaceConfig.parse(config);

    assertThat(parsed.hiddenSize()).isEqualTo(768);
    assertThat(parsed.intermediateSize()).isEqualTo(384);
    assertThat(parsed.sharedIntermediateSize()).isEqualTo(1_536);
    assertThat(parsed.numLayers()).isEqualTo(20);
    assertThat(parsed.queryDimension()).isEqualTo(768);
    assertThat(parsed.keyValueDimension()).isEqualTo(256);
    assertThat(parsed.vocabSize()).isEqualTo(128_256);
    assertThat(parsed.numExperts()).isEqualTo(60);
    assertThat(parsed.expertsPerToken()).isEqualTo(4);
    assertThat(parsed.routeScale()).isEqualTo(2.5f);
    assertThat(parsed.moeLayers())
        .containsExactlyElementsOf(java.util.stream.IntStream.range(0, 20).boxed().toList());
    assertThat(parsed.quantization()).isPresent();
    assertThat(parsed.quantization().orElseThrow().groupSize()).isEqualTo(32);
    assertThat(parsed.quantization().orElseThrow().scaleType()).isEqualTo("float16");
  }

  @Test
  void acceptsTheBf16SftContractWithoutQuantization(@TempDir Path directory) throws Exception {
    Path config = directory.resolve("config.json");
    Files.writeString(config, smallConfig("mobilemoe-int4-g32", false));

    assertThat(MobileMoeHuggingFaceConfig.parse(config).quantization()).isEmpty();
  }

  @Test
  void rejectsAFormatThatWouldBeDecodedWithTheWrongNibbleContract(@TempDir Path directory)
      throws Exception {
    Path config = directory.resolve("config.json");
    Files.writeString(config, smallConfig("generic-int4", true));

    assertThatThrownBy(() -> MobileMoeHuggingFaceConfig.parse(config))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("mobilemoe-int4-g32");
  }

  @Test
  void doesNotClaimAnotherHuggingFaceArchitecture(@TempDir Path directory) throws Exception {
    Path config = directory.resolve("config.json");
    Files.writeString(config, "{\"model_type\":\"qwen2\"}");

    assertThat(MobileMoeHuggingFaceConfig.matches(config)).isFalse();
  }

  private static String smallConfig(String quantizationFormat, boolean quantized) {
    String quantization =
        quantized
            ? """
              ,"quantization": {
                "format": "%s",
                "group_size": 32,
                "embedding_group_size": 32,
                "qmin": -8,
                "qmax": 7,
                "symmetric": true,
                "packed": true,
                "scale_dtype": "float16",
                "linear_group_axis": "in (last dim of [out, in])",
                "expert_group_axis": "out (last dim of [E, in, out])"
              }
              """
                .formatted(quantizationFormat)
            : "";
    return """
        {
          "model_type": "mobilemoe",
          "architectures": ["MobileMoEForCausalLM"],
          "vocab_size": 128256,
          "hidden_size": 768,
          "intermediate_size": 384,
          "intermediate_size_mlp": 1536,
          "num_hidden_layers": 20,
          "num_attention_heads": 12,
          "num_key_value_heads": 4,
          "head_dim": 64,
          "hidden_act": "silu",
          "max_position_embeddings": 8192,
          "rms_norm_eps": 0.00001,
          "rope_theta": 500000.0,
          "rope_scaling": {
            "rope_type": "llama3",
            "factor": 16,
            "original_max_position_embeddings": 8192,
            "low_freq_factor": 1.0,
            "high_freq_factor": 1.0
          },
          "num_local_experts": 60,
          "num_experts_per_tok": 4,
          "interleave_moe_layer_step": 1,
          "no_rope_layers": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
          "layer_types": [
            "full_attention","full_attention","full_attention","full_attention","full_attention",
            "full_attention","full_attention","full_attention","full_attention","full_attention",
            "full_attention","full_attention","full_attention","full_attention","full_attention",
            "full_attention","full_attention","full_attention","full_attention","full_attention"
          ],
          "use_qk_norm": true,
          "tie_word_embeddings": true,
          "attention_bias": false,
          "routed_scaling_factor": 2.5,
          "norm_topk_prob": true
          %s
        }
        """
        .formatted(quantization);
  }
}
