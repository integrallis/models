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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.llama.DecoderArchitecture;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJar;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;

@Tag("integration")
class Gemma3ModelJarsIntegrationTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;

  private static final ModelJar GEMMA_3_1B_Q4_K_M =
      ModelJar.of("hf://bartowski/google_gemma-3-1b-it-GGUF")
          .version("[3.0.0,4.0.0)")
          .variant("q4_k_m")
          .backend("rust-ffm")
          .capability("chat");

  @Test
  void pinnedArtifactContainsGemma3TensorContract() throws Exception {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaWeights.LayerWeights firstLayer = weights.layer(0);

      assertThat(config.architecture()).isEqualTo(DecoderArchitecture.GEMMA3);
      assertThat(config.embeddingDim()).isEqualTo(1152);
      assertThat(config.keyLength()).isEqualTo(256);
      assertThat(config.valueLength()).isEqualTo(256);
      assertThat(config.slidingWindow()).isEqualTo(512);
      assertThat(file.getTensor("token_embd.weight").type()).isEqualTo(GgufTensorType.Q8_0);
      assertThat(firstLayer.qNorm()).hasSize(256);
      assertThat(firstLayer.kNorm()).hasSize(256);
      assertThat(firstLayer.attentionPostNorm()).hasSize(1152);
      assertThat(firstLayer.ffnPostNorm()).hasSize(1152);
    }
  }

  @Test
  void matchesLlamaCppCandidateOracle() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor.localPath().orElseThrow())) {
      int[] promptTokens = backend.tokenizer().encode("Hello");

      assertThat(promptTokens).containsExactly(2, 9259);
      backend.reset();
      float[] logits = backend.prefill(promptTokens, 0);
      int[] expected = {607, 496, 2934, 236761};
      int[][] expectedTopTwo = {{607, 236764}, {496, 506}, {2934, 861}, {236761, 236881}};
      // Quantized accumulation order may swap near ties without changing the candidate set. The
      // first b10012 decision has a wide enough margin to require exact argmax parity.
      for (int index = 0; index < expected.length; index++) {
        int[] actualTopTwo = topTokenIds(logits, 2);
        assertThat(actualTopTwo)
            .as(
                "top candidates for token %s must match llama.cpp b10012; Java top logits: %s",
                index, topLogits(logits, 10))
            .containsExactlyInAnyOrder(expectedTopTwo[index]);
        if (index == 0) {
          assertThat(argmax(logits))
              .as("the robust first-token argmax must match llama.cpp b10012")
              .isEqualTo(expected[index]);
        }
        logits = backend.forward(expected[index], promptTokens.length + index);
      }
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  @Test
  void batchedPrefillMatchesSequentialStateExactly() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    String previousBatch =
        System.getProperty(PureJavaPlanConfiguration.PREFILL_BATCH_SIZE_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try {
      int[] promptTokens;
      float[] expected;
      int nextToken;
      float[] expectedNext;
      System.setProperty(PureJavaPlanConfiguration.PREFILL_BATCH_SIZE_PROPERTY, "1");
      try (PureJavaBackend sequential =
          PureJavaBackend.load(descriptor.localPath().orElseThrow())) {
        promptTokens = sequential.tokenizer().encode("Hello");
        expected = sequential.prefill(promptTokens, 0).clone();
        nextToken = argmax(expected);
        expectedNext = sequential.forward(nextToken, promptTokens.length).clone();
      }

      System.setProperty(PureJavaPlanConfiguration.PREFILL_BATCH_SIZE_PROPERTY, "32");
      try (PureJavaBackend batched = PureJavaBackend.load(descriptor.localPath().orElseThrow())) {
        assertThat(batched.prefill(promptTokens, 0)).containsExactly(expected);
        assertThat(batched.forward(nextToken, promptTokens.length)).containsExactly(expectedNext);
      }
    } finally {
      restoreSystemProperty(PureJavaPlanConfiguration.PREFILL_BATCH_SIZE_PROPERTY, previousBatch);
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previousContext);
    }
  }

  private static ModelJarDescriptor descriptorWithInstalledArtifact() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(GEMMA_3_1B_Q4_K_M).orElseThrow();
    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as(
            "%s must be present. Run the Gemma 3 fixture download task before this test.",
            descriptor.localPath().orElseThrow())
        .isTrue();
    return descriptor;
  }

  private static String topLogits(float[] values, int count) {
    int[] tokenIds = topTokenIds(values, count);
    StringBuilder result = new StringBuilder();
    for (int rank = 0; rank < tokenIds.length; rank++) {
      int tokenId = tokenIds[rank];
      if (rank != 0) {
        result.append(", ");
      }
      result.append(tokenId).append('=').append(values[tokenId]);
    }
    return result.toString();
  }

  private static int[] topTokenIds(float[] values, int count) {
    int[] tokenIds = new int[count];
    boolean[] selected = new boolean[values.length];
    for (int rank = 0; rank < tokenIds.length; rank++) {
      int best = -1;
      for (int index = 0; index < values.length; index++) {
        if (!selected[index] && (best < 0 || values[index] > values[best])) {
          best = index;
        }
      }
      selected[best] = true;
      tokenIds[rank] = best;
    }
    return tokenIds;
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  private static void restoreSystemProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }
}
