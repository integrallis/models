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
package com.integrallis.models.backend.purejava.huggingface;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.llama.DecoderArchitecture;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class Qwen2HuggingFaceConfigTest {

  @Test
  void parsesPinnedOfficialQwen25Config() throws Exception {
    Path path =
        Path.of(
            Qwen2HuggingFaceConfigTest.class
                .getResource("/huggingface/qwen2.5-0.5b-instruct-config.json")
                .toURI());

    // Qwen/Qwen2.5-0.5B-Instruct at revision 7ae557604adf67be50417f59c2c2f167def9a775.
    assertThat(sha256(path))
        .isEqualTo("18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45");
    Qwen2HuggingFaceConfig parsed = Qwen2HuggingFaceConfig.parse(path);

    assertThat(parsed.architectures()).containsExactly("Qwen2ForCausalLM");
    assertThat(parsed.hiddenActivation()).isEqualTo("silu");
    assertThat(parsed.torchDtype()).isEqualTo("bfloat16");
    assertThat(parsed.tieWordEmbeddings()).isTrue();
    assertThat(parsed.bosTokenId()).isEqualTo(151643);
    assertThat(parsed.eosTokenId()).isEqualTo(151645);
    assertThat(parsed.model().architecture()).isEqualTo(DecoderArchitecture.QWEN2);
    assertThat(parsed.model().embeddingDim()).isEqualTo(896);
    assertThat(parsed.model().hiddenDim()).isEqualTo(4864);
    assertThat(parsed.model().numLayers()).isEqualTo(24);
    assertThat(parsed.model().numHeads()).isEqualTo(14);
    assertThat(parsed.model().numKvHeads()).isEqualTo(2);
    assertThat(parsed.model().headDim()).isEqualTo(64);
    assertThat(parsed.model().vocabSize()).isEqualTo(151936);
    assertThat(parsed.model().contextLength()).isEqualTo(32768);
    assertThat(parsed.model().ropeTheta()).isEqualTo(1_000_000.0f);
    assertThat(parsed.model().ropeFrequencyScale()).isEqualTo(1.0f);
    assertThat(parsed.model().rmsNormEps()).isEqualTo(1.0e-6f);
    assertThat(parsed.model().slidingWindow()).isZero();
  }

  @Test
  void appliesFreeTokenQwen2Fallbacks(@TempDir Path directory) throws IOException {
    Path config = directory.resolve("config.json");
    Files.writeString(
        config,
        """
        {
          "hidden_act": "silu",
          "hidden_size": 16,
          "intermediate_size": 48,
          "max_position_embeddings": 2048,
          "num_attention_heads": 4,
          "num_hidden_layers": 2,
          "rms_norm_eps": 0.00001,
          "vocab_size": 32,
          "rope_scaling": {"rope_theta": 500000.0}
        }
        """);

    Qwen2HuggingFaceConfig parsed = Qwen2HuggingFaceConfig.parse(config);

    assertThat(parsed.architectures()).containsExactly("Qwen2ForCausalLM");
    assertThat(parsed.tieWordEmbeddings()).isFalse();
    assertThat(parsed.model().architecture()).isEqualTo(DecoderArchitecture.QWEN2);
    assertThat(parsed.model().numKvHeads()).isEqualTo(4);
    assertThat(parsed.model().headDim()).isEqualTo(4);
    assertThat(parsed.model().ropeTheta()).isEqualTo(500_000.0f);
  }

  @Test
  void explicitHeadAndRopeValuesTakePrecedence(@TempDir Path directory) throws IOException {
    Path config = directory.resolve("config.json");
    Files.writeString(
        config,
        """
        {
          "architectures": ["Qwen2ForCausalLM"],
          "model_type": "qwen2",
          "hidden_act": "silu",
          "hidden_size": 18,
          "intermediate_size": 42,
          "max_position_embeddings": 4096,
          "num_attention_heads": 3,
          "num_hidden_layers": 2,
          "num_key_value_heads": 1,
          "head_dim": 8,
          "rms_norm_eps": 0.000001,
          "rope_theta": 750000.0,
          "rope_scaling": {"rope_theta": 500000.0},
          "torch_dtype": "bfloat16",
          "vocab_size": 64
        }
        """);

    Qwen2HuggingFaceConfig parsed = Qwen2HuggingFaceConfig.parse(config);

    assertThat(parsed.model().headDim()).isEqualTo(8);
    assertThat(parsed.model().ropeTheta()).isEqualTo(750_000.0f);
  }

  @Test
  void rejectsWrongArchitecture(@TempDir Path directory) throws IOException {
    Path wrongArchitecture = directory.resolve("wrong.json");
    Files.writeString(
        wrongArchitecture,
        """
        {"model_type":"llama"}
        """);

    assertThatThrownBy(() -> Qwen2HuggingFaceConfig.parse(wrongArchitecture))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("model_type qwen2");
  }

  @Test
  void rejectsDuplicateFields(@TempDir Path directory) throws IOException {
    Path duplicate = directory.resolve("duplicate.json");
    Files.writeString(duplicate, "{\"hidden_size\":16,\"hidden_size\":16}");

    assertThatThrownBy(() -> Qwen2HuggingFaceConfig.parse(duplicate))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("duplicate config JSON");
  }

  private static String sha256(Path path) throws IOException, NoSuchAlgorithmException {
    return HexFormat.of()
        .formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)));
  }
}
