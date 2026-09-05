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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class SopranoBackendIntegrationTest {

  @Test
  void loadsAndExecutesTheRealStandaloneQ8Artifact() throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));

    try (SopranoBackend backend = SopranoBackend.load(Path.of(configured))) {
      SopranoBackend.Step prompt = backend.begin("The JVM can speak for itself.");

      assertThat(prompt.logits()).hasSize(8192);
      assertThat(prompt.hiddenState()).hasSize(512);
      assertThat(allFinite(prompt.logits())).isTrue();
      assertThat(allFinite(prompt.hiddenState())).isTrue();
      assertThat(backend.checkpoint()).isGreaterThan(3);
      assertThat(backend.eosToken()).isEqualTo(3);
      assertThat(backend.sampleRate()).isEqualTo(32_000);
      assertThat(backend.diagnostics().backend()).isEqualTo("pure-java");
      assertThat(backend.diagnostics().optimizations()).isNotEmpty();
      assertThat(backend.diagnostics().optimization("soprano.vocoder.prepared-f32"))
          .get()
          .satisfies(
              decision -> {
                assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
                assertThat(Long.parseLong(decision.settings().get("expandedBytes")))
                    .isGreaterThan(Long.parseLong(decision.settings().get("serializedBytes")));
              });
    }
  }

  @Test
  void routesTheRealLanguageModelThroughAnInjectedQ8KernelAndOwnsItsLifetime() throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    assumeTrue(hasQ8LanguageModelWeights(Path.of(configured)));
    RecordingQ8Kernel kernel = new RecordingQ8Kernel();

    try (SopranoBackend backend = SopranoBackend.load(Path.of(configured), kernel)) {
      SopranoBackend.Step prompt = backend.begin("The JVM can speak for itself.");

      assertThat(allFinite(prompt.logits())).isTrue();
      assertThat(kernel.multiplyCalls).isGreaterThan(0);
      assertThat(backend.diagnostics().environment())
          .containsEntry("matrix-kernel", "recording-q8");
    }

    assertThat(kernel.closed).isTrue();
  }

  private static boolean hasQ8LanguageModelWeights(Path artifact) throws Exception {
    try (Arena arena = Arena.ofConfined()) {
      return GgufParser.parse(artifact, arena)
              .getTensor("model.layers.0.self_attn.q_proj.weight")
              .type()
          == GgufTensorType.Q8_0;
    }
  }

  private static boolean allFinite(float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        return false;
      }
    }
    return true;
  }

  private static final class RecordingQ8Kernel implements GgufBatchedMatrixKernel {
    private int multiplyCalls;
    private boolean closed;

    @Override
    public String implementation() {
      return "recording-q8";
    }

    @Override
    public boolean supports(GgufTensorType type) {
      return type == GgufTensorType.Q8_0;
    }

    @Override
    public void multiply(
        float[] output,
        float[] input,
        MemorySegment weights,
        GgufTensorType type,
        int batchSize,
        int rows,
        int cols) {
      multiplyCalls++;
      VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
          input,
          weights,
          batchSize,
          rows,
          cols,
          output,
          new byte[Math.multiplyExact(batchSize, cols)],
          new float[Math.multiplyExact(batchSize, cols / 32)]);
    }

    @Override
    public void close() {
      closed = true;
    }
  }
}
