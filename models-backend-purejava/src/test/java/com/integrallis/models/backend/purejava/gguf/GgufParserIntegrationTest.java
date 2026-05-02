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
package com.integrallis.models.backend.purejava.gguf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assumptions.assumeThat;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Integration tests that parse a real GGUF model file (Qwen3-0.6B-Q4_0). Validates the parser
 * against real-world data, not synthetic test fixtures.
 *
 * <p>Requires the model to be downloaded to ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf
 */
@Tag("integration")
class GgufParserIntegrationTest {

  private static final Path MODEL_PATH =
      Path.of(System.getProperty("user.home"), ".jvllm", "models", "Qwen3-0.6B-Q4_0.gguf");

  @BeforeAll
  static void checkModelExists() {
    assumeThat(Files.exists(MODEL_PATH))
        .as(
            "Qwen3-0.6B-Q4_0.gguf must be present at %s. Download with: curl -L -o %s"
                + " https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf",
            MODEL_PATH, MODEL_PATH)
        .isTrue();
  }

  @Nested
  class HeaderParsing {

    @Test
    void parsesHeaderSuccessfully() throws IOException {
      try (Arena arena = Arena.ofConfined()) {
        GgufFile file = GgufParser.parse(MODEL_PATH, arena);

        assertThat(file.header().version()).isIn(2, 3);
        assertThat(file.header().tensorCount()).isGreaterThan(0);
        assertThat(file.header().metadataKvCount()).isGreaterThan(0);
      }
    }

    @Test
    void metadataContainsArchitecture() throws IOException {
      try (Arena arena = Arena.ofConfined()) {
        GgufFile file = GgufParser.parse(MODEL_PATH, arena);

        assertThat(file.metadata().getString("general.architecture")).isPresent();
      }
    }

    @Test
    void metadataContainsModelName() throws IOException {
      try (Arena arena = Arena.ofConfined()) {
        GgufFile file = GgufParser.parse(MODEL_PATH, arena);

        assertThat(file.metadata().getString("general.name")).isPresent();
      }
    }
  }

  @Nested
  class ModelArchitecture {

    @Test
    void hasExpectedLlamaMetadataKeys() throws IOException {
      try (Arena arena = Arena.ofConfined()) {
        GgufFile file = GgufParser.parse(MODEL_PATH, arena);
        GgufMetadata meta = file.metadata();

        // Qwen3 uses llama architecture keys
        String arch = meta.getString("general.architecture").orElse("");
        assertThat(arch).isIn("llama", "qwen2", "qwen3");

        // Should have embedding dimension
        assertThat(
                meta.getUint32(arch + ".embedding_length")
                    .or(() -> meta.getUint32("llama.embedding_length")))
            .isPresent();
      }
    }

    @Test
    void hasTensorsWithExpectedTypes() throws IOException {
      try (Arena arena = Arena.ofConfined()) {
        GgufFile file = GgufParser.parse(MODEL_PATH, arena);

        assertThat(file.tensorInfos()).isNotEmpty();

        // Token embedding should exist
        boolean hasTokenEmbed =
            file.tensorInfos().stream().anyMatch(t -> t.name().contains("token_embd"));
        assertThat(hasTokenEmbed).as("Should have token embedding tensor").isTrue();

        // Should have attention weights
        boolean hasAttnQ = file.tensorInfos().stream().anyMatch(t -> t.name().contains("attn_q"));
        assertThat(hasAttnQ).as("Should have attention Q weight").isTrue();

        // Q4_0 model should have quantized tensors
        boolean hasQ4Tensors =
            file.tensorInfos().stream().anyMatch(t -> t.type() == GgufTensorType.Q4_0);
        assertThat(hasQ4Tensors).as("Q4_0 model should have Q4_0 quantized tensors").isTrue();
      }
    }

    @Test
    void tensorDataOffsetIsValid() throws IOException {
      try (Arena arena = Arena.ofConfined()) {
        GgufFile file = GgufParser.parse(MODEL_PATH, arena);

        // Tensor data offset must be aligned
        assertThat(file.tensorDataOffset() % GgufConstants.DEFAULT_ALIGNMENT).isZero();

        // Tensor data must be within file bounds
        long fileSize = Files.size(MODEL_PATH);
        assertThat(file.tensorDataOffset()).isLessThan(fileSize);

        // Each tensor's data should be accessible without exception
        for (GgufTensorInfo info : file.tensorInfos()) {
          long tensorEnd = file.tensorDataOffset() + info.offset() + info.byteSize();
          assertThat(tensorEnd)
              .as("Tensor '%s' data must be within file bounds", info.name())
              .isLessThanOrEqualTo(fileSize);
        }
      }
    }
  }

  @Nested
  class TokenizerMetadata {

    @Test
    void hasTokenizerVocabulary() throws IOException {
      try (Arena arena = Arena.ofConfined()) {
        GgufFile file = GgufParser.parse(MODEL_PATH, arena);

        assertThat(file.metadata().getStringArray("tokenizer.ggml.tokens")).isPresent();
        assertThat(file.metadata().getStringArray("tokenizer.ggml.tokens").get())
            .hasSizeGreaterThan(1000);
      }
    }

    @Test
    void hasTokenizerSpecialTokens() throws IOException {
      try (Arena arena = Arena.ofConfined()) {
        GgufFile file = GgufParser.parse(MODEL_PATH, arena);

        assertThat(file.metadata().getUint32("tokenizer.ggml.bos_token_id")).isPresent();
        assertThat(file.metadata().getUint32("tokenizer.ggml.eos_token_id")).isPresent();
      }
    }

    @Test
    void vocabularySizeMatchesModelConfig() throws IOException {
      try (Arena arena = Arena.ofConfined()) {
        GgufFile file = GgufParser.parse(MODEL_PATH, arena);
        GgufMetadata meta = file.metadata();

        int vocabFromTokens = meta.getStringArray("tokenizer.ggml.tokens").get().size();

        // Token embedding tensor shape should match vocab size
        GgufTensorInfo tokenEmbed =
            file.tensorInfos().stream()
                .filter(t -> t.name().equals("token_embd.weight"))
                .findFirst()
                .orElseThrow();

        // The embedding tensor is [embedding_dim x vocab_size] or [vocab_size x embedding_dim]
        long[] shape = tokenEmbed.shape();
        boolean vocabInShape =
            shape[0] == vocabFromTokens || (shape.length > 1 && shape[1] == vocabFromTokens);
        assertThat(vocabInShape)
            .as(
                "Token embedding shape %s should contain vocab size %d",
                java.util.Arrays.toString(shape), vocabFromTokens)
            .isTrue();
      }
    }
  }
}
