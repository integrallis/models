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
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;

@Tag("integration")
class SmolLm2ModelJarsIntegrationTest {

  private static final int INTEGRATION_CONTEXT_LENGTH = 128;
  private static final String PREFILL_BATCH_SIZE_PROPERTY = "models.purejava.prefillBatchSize";

  private static final ModelJarRequirement SMOLLM2_360M_Q8_0 =
      ModelJarRequirement.forSource("hf://HuggingFaceTB/SmolLM2-360M-Instruct-GGUF")
          .versionRange("[2.0.0,3.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("chat")
          .build();

  @Test
  void loadsSmolLm2360MQ80ThroughModelJars() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.name()).isEqualTo("pure-java");
      assertThat(backend.metadata().modelFamily()).isEqualTo("llama");
      assertThat(backend.metadata().contextLength()).isEqualTo(8_192);

      int[] tokens = backend.tokenizer().encode("Hello from Java");
      assertThat(tokens).isNotEmpty();

      float[] logits = backend.forward(tokens[0], 0);
      assertThat(logits).hasSize(backend.metadata().vocabSize());
      for (int i = 0; i < logits.length; i++) {
        assertThat(logits[i]).as("Logit at index %d should be finite", i).isFinite();
      }
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  @Test
  void groupedQ8GateUpProjectionsMatchSeparateLayerMatmulsExactly() throws Exception {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();

    try (Arena arena = Arena.ofShared()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaWeights.LayerWeights layer = weights.layer(0);

      assertThat(layer.wqType()).isEqualTo(GgufTensorType.Q8_0);
      assertThat(layer.wkType()).isEqualTo(GgufTensorType.Q8_0);
      assertThat(layer.wvType()).isEqualTo(GgufTensorType.Q8_0);
      assertThat(layer.ffnGateType()).isEqualTo(GgufTensorType.Q8_0);
      assertThat(layer.ffnUpType()).isEqualTo(GgufTensorType.Q8_0);

      int cols = config.embeddingDim();
      int rows = config.hiddenDim();
      float[] input = new float[cols];
      float[] normalized = new float[cols];
      weights.embedToken(1, input);
      TensorOps.rmsNorm(normalized, input, layer.ffnNorm(), cols, config.rmsNormEps());

      float[] expectedGate = new float[rows];
      float[] expectedUp = new float[rows];
      float[] actualGate = new float[rows];
      float[] actualUp = new float[rows];
      TensorOps.ggufMatmul(
          expectedGate, normalized, layer.ffnGate(), layer.ffnGateType(), rows, cols);
      TensorOps.ggufMatmul(expectedUp, normalized, layer.ffnUp(), layer.ffnUpType(), rows, cols);

      TensorOps.ggufDualMatmul(
          actualGate,
          layer.ffnGate(),
          layer.ffnGateType(),
          rows,
          actualUp,
          layer.ffnUp(),
          layer.ffnUpType(),
          rows,
          normalized,
          cols,
          new byte[cols],
          new float[cols / 32],
          new short[cols / 16],
          GgufQ4Kernel.WIDENED);

      assertThat(actualGate).containsExactly(expectedGate);
      assertThat(actualUp).containsExactly(expectedUp);
    }
  }

  @Test
  void q8PrefillBatchMatchesSequentialExecutionAndPreservesDecodeState() {
    ModelJarDescriptor descriptor = descriptorWithInstalledArtifact();
    String previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    String previousBatchSize = System.getProperty(PREFILL_BATCH_SIZE_PROPERTY);
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(INTEGRATION_CONTEXT_LENGTH));

    try {
      int[] tokens;
      float[] expectedPrefill;
      float[] expectedDecode;
      System.setProperty(PREFILL_BATCH_SIZE_PROPERTY, "1");
      try (PureJavaBackend sequential = PureJavaBackend.load(descriptor)) {
        tokens =
            sequential
                .tokenizer()
                .encode("Explain why Java runs anywhere in one concise sentence.");
        assertThat(tokens.length).isGreaterThan(1);
        expectedPrefill = sequential.prefill(tokens, 0).clone();
        expectedDecode = sequential.forward(tokens[0], tokens.length);
      }

      System.setProperty(PREFILL_BATCH_SIZE_PROPERTY, "32");
      try (PureJavaBackend batched = PureJavaBackend.load(descriptor)) {
        assertThat(TensorOps.supportsBatchedMatmul(GgufTensorType.Q8_0)).isTrue();
        assertThat(batched.prefill(tokens, 0)).containsExactly(expectedPrefill);
        assertThat(batched.forward(tokens[0], tokens.length)).containsExactly(expectedDecode);
      }
    } finally {
      restoreSystemProperty(PREFILL_BATCH_SIZE_PROPERTY, previousBatchSize);
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previousContext);
    }
  }

  private static ModelJarDescriptor descriptorWithInstalledArtifact() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(SMOLLM2_360M_Q8_0).orElseThrow();
    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as(
            "%s must be present. Run :models-backend-purejava:integrationTest or the"
                + " fixture download task before running this test.",
            descriptor.localPath().orElseThrow())
        .isTrue();
    return descriptor;
  }

  private static void restoreSystemProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }
}
