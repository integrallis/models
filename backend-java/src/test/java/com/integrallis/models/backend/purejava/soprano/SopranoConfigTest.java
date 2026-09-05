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
package com.integrallis.models.backend.purejava.soprano;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SopranoConfigTest {

  @Test
  void readsThePublishedSopranoArchitecture() {
    SopranoConfig config =
        SopranoConfig.fromJson(
            """
            {
              "model_type": "qwen3",
              "hidden_size": 512,
              "intermediate_size": 2304,
              "num_hidden_layers": 17,
              "num_attention_heads": 4,
              "num_key_value_heads": 1,
              "head_dim": 128,
              "vocab_size": 8192,
              "max_position_embeddings": 1024,
              "rms_norm_eps": 0.000001,
              "rope_theta": 10000,
              "bos_token_id": 3,
              "eos_token_id": 3
            }
            """);

    assertThat(config.hiddenSize()).isEqualTo(512);
    assertThat(config.intermediateSize()).isEqualTo(2304);
    assertThat(config.layers()).isEqualTo(17);
    assertThat(config.attentionHeads()).isEqualTo(4);
    assertThat(config.kvHeads()).isEqualTo(1);
    assertThat(config.headDim()).isEqualTo(128);
    assertThat(config.vocabSize()).isEqualTo(8192);
    assertThat(config.contextLength()).isEqualTo(1024);
    assertThat(config.sampleRate()).isEqualTo(32_000);
    assertThat(config.decoderDim()).isEqualTo(768);
    assertThat(config.decoderLayers()).isEqualTo(8);
  }

  @Test
  void refusesToSilentlyRunACompatibleLookingDifferentArchitecture() {
    assertThatThrownBy(
            () ->
                SopranoConfig.fromJson(
                    """
                    {"model_type":"llama","hidden_size":512}
                    """))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("qwen3");
  }
}
