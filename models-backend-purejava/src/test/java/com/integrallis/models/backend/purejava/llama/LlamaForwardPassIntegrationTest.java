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
package com.integrallis.models.backend.purejava.llama;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assumptions.assumeThat;

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Integration tests for the Llama forward pass using a real Qwen3-0.6B Q4_0 model. These tests
 * validate that the full inference pipeline (GGUF parsing, weight loading, tokenization, forward
 * pass) produces correct results with real model weights.
 */
@Tag("integration")
class LlamaForwardPassIntegrationTest {

  private static final Path MODEL_PATH =
      Path.of(System.getProperty("user.home"), ".jvllm", "models", "Qwen3-0.6B-Q4_0.gguf");

  private static Arena arena;
  private static GgufFile file;
  private static LlamaConfig config;
  private static LlamaWeights weights;
  private static GgufTokenizer tokenizer;

  @BeforeAll
  static void loadModel() throws IOException {
    assumeThat(Files.exists(MODEL_PATH))
        .as("Qwen3-0.6B-Q4_0.gguf must be present at %s", MODEL_PATH)
        .isTrue();

    arena = Arena.ofShared();
    file = GgufParser.parse(MODEL_PATH, arena);

    // Qwen3 may use "qwen2" or "qwen3" architecture keys but is structurally llama-compatible
    config = LlamaConfig.fromMetadata(file.metadata());
    weights = LlamaWeights.fromGgufFile(file, config);
    tokenizer = GgufTokenizer.fromMetadata(file.metadata());
  }

  @AfterAll
  static void closeArena() {
    if (arena != null) {
      arena.close();
    }
  }

  @Nested
  class ModelConfiguration {

    @Test
    void configHasReasonableEmbeddingDim() {
      // Qwen3-0.6B has embedding_dim of 1024
      assertThat(config.embeddingDim()).isGreaterThanOrEqualTo(512);
      assertThat(config.embeddingDim()).isLessThanOrEqualTo(4096);
    }

    @Test
    void configHasMultipleLayers() {
      // Qwen3-0.6B has 28 layers
      assertThat(config.numLayers()).isGreaterThanOrEqualTo(4);
    }

    @Test
    void configHasValidHeadCounts() {
      assertThat(config.numHeads()).isGreaterThan(0);
      assertThat(config.numKvHeads()).isGreaterThan(0);
      assertThat(config.numHeads() % config.numKvHeads()).isZero();
      assertThat(config.embeddingDim() % config.numHeads()).isZero();
    }

    @Test
    void configHasReasonableVocabSize() {
      // Qwen3 has 151k+ vocab
      assertThat(config.vocabSize()).isGreaterThan(10_000);
    }
  }

  @Nested
  class SingleForwardPass {

    @Test
    void producesNonZeroLogits() {
      KvCache cache = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);

      // Encode a simple token (BOS token)
      float[] logits = forwardPass.forward(tokenizer.bosToken(), 0);

      assertThat(logits).hasSize(config.vocabSize());

      // At least some logits should be non-zero
      boolean hasNonZero = false;
      for (float l : logits) {
        if (l != 0.0f) {
          hasNonZero = true;
          break;
        }
      }
      assertThat(hasNonZero).as("Logits should not all be zero").isTrue();
    }

    @Test
    void logitsAreFinite() {
      KvCache cache = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);

      float[] logits = forwardPass.forward(tokenizer.bosToken(), 0);

      for (int i = 0; i < logits.length; i++) {
        assertThat(logits[i]).as("Logit at index %d should be finite", i).isFinite();
      }
    }

    @Test
    void argmaxIsWithinVocabRange() {
      KvCache cache = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);

      float[] logits = forwardPass.forward(tokenizer.bosToken(), 0);

      int argmax = 0;
      for (int i = 1; i < logits.length; i++) {
        if (logits[i] > logits[argmax]) argmax = i;
      }
      assertThat(argmax).isGreaterThanOrEqualTo(0).isLessThan(config.vocabSize());

      // The argmax should decode to a real token
      String token = tokenizer.decode(argmax);
      assertThat(token).isNotNull();
    }
  }

  @Nested
  class MultiTokenForward {

    @Test
    void consecutiveForwardPassesProduceDifferentLogits() {
      KvCache cache = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);

      // Encode "Hello" and run forward on each token
      int[] tokens = tokenizer.encode("Hello");
      assumeThat(tokens.length).isGreaterThan(0);

      float[] logits0 = forwardPass.forward(tokens[0], 0);

      if (tokens.length > 1) {
        float[] logits1 = forwardPass.forward(tokens[1], 1);
        // Different positions and accumulated context should produce different logits
        assertThat(logits0).isNotEqualTo(logits1);
      }
    }

    @Test
    void prefillProducesConsistentLogits() {
      // Run two identical prefill sequences and verify they produce the same final logits
      int[] tokens = tokenizer.encode("Hello");
      assumeThat(tokens.length).isGreaterThan(0);

      KvCache cache1 = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass pass1 = new LlamaForwardPass(config, weights, cache1);
      float[] lastLogits1 = null;
      for (int i = 0; i < tokens.length; i++) {
        lastLogits1 = pass1.forward(tokens[i], i);
      }

      KvCache cache2 = new KvCache(config.numLayers(), config.contextLength(), config.kvDim());
      LlamaForwardPass pass2 = new LlamaForwardPass(config, weights, cache2);
      float[] lastLogits2 = null;
      for (int i = 0; i < tokens.length; i++) {
        lastLogits2 = pass2.forward(tokens[i], i);
      }

      assertThat(lastLogits1).isEqualTo(lastLogits2);
    }
  }
}
