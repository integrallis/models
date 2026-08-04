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
package com.integrallis.models.runtime;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalStateException;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class InferencePipelineTest {

  @Test
  void exposesTokenizerMetadataAndManagedContextOperations() {
    StubBackend backend = new StubBackend();

    try (InferencePipeline pipeline = new InferencePipeline(backend)) {
      ModelPrompt prompt = ModelPrompt.builder().control("<control>").text("hello").build();

      assertThat(pipeline.metadata().modelName()).isEqualTo("fixture");
      assertThat(pipeline.tokenizer()).isSameAs(backend.tokenizer());
      assertThat(pipeline.contextWindow().capacity()).isEqualTo(8);
      assertThat(pipeline.contextWindow().position()).hasValue(0);
      assertThat(pipeline.tokenize(prompt)).containsExactly(0, 1);
      assertThat(pipeline.prefill(prompt, 0)).containsExactly(0.0f, 0.0f, 10.0f);
      assertThat(pipeline.contextWindow().position()).hasValue(2);
      assertThat(pipeline.contextWindow().remaining()).hasValue(6);

      pipeline.forward(0, 2);
      assertThat(pipeline.checkpoint()).isEqualTo(3);
      pipeline.rewind(1);
      assertThat(pipeline.contextWindow().position()).hasValue(1);
      pipeline.resetContext();
      assertThat(pipeline.contextWindow().position()).hasValue(0);
    }

    assertThat(backend.closeCount).isEqualTo(1);
  }

  @Test
  void preservesStructuredPromptTokenizationDuringGeneration() {
    StubBackend backend = new StubBackend();
    ModelPrompt prompt = ModelPrompt.builder().control("<control>").text("hello").build();

    try (InferencePipeline pipeline = new InferencePipeline(backend)) {
      String generated = pipeline.generate(prompt, deterministicOptions());

      assertThat(generated).isEmpty();
      assertThat(backend.structuredEncodes).isEqualTo(1);
      assertThat(backend.plainEncodes).isZero();
    }
  }

  @Test
  void directContextAccessInvalidatesTheHighLevelPrefixCache() {
    StubBackend backend = new StubBackend();

    try (InferencePipeline pipeline = new InferencePipeline(backend)) {
      pipeline.generate(ModelPrompt.text("hello"), deterministicOptions());
      pipeline.generate(ModelPrompt.text("hello"), deterministicOptions());
      pipeline.resetContext();
      pipeline.generate(ModelPrompt.text("hello"), deterministicOptions());
    }

    assertThat(backend.prefillStartPositions).containsExactly(0, 1, 0);
  }

  @Test
  void closesTheOwnedBackendExactlyOnceAndRejectsFurtherUse() {
    StubBackend backend = new StubBackend();
    InferencePipeline pipeline = new InferencePipeline(backend);

    pipeline.close();
    pipeline.close();

    assertThat(backend.closeCount).isEqualTo(1);
    assertThatIllegalStateException()
        .isThrownBy(pipeline::metadata)
        .withMessageContaining("closed");
  }

  private static SamplingOptions deterministicOptions() {
    return SamplingOptions.builder().temperature(0.0f).maxTokens(1).build();
  }

  private static final class StubBackend implements RewindableInferenceBackend {
    private final List<Integer> prefillStartPositions = new ArrayList<>();
    private int position;
    private int plainEncodes;
    private int structuredEncodes;
    private int closeCount;

    private final Tokenizer tokenizer =
        new Tokenizer() {
          @Override
          public int[] encode(String text) {
            plainEncodes++;
            return new int[] {0, 1};
          }

          @Override
          public int[] encode(ModelPrompt prompt) {
            structuredEncodes++;
            return new int[] {0, 1};
          }

          @Override
          public String decode(int[] tokens) {
            return "";
          }

          @Override
          public String decode(int token) {
            return "";
          }

          @Override
          public int vocabSize() {
            return 3;
          }

          @Override
          public int bosToken() {
            return 0;
          }

          @Override
          public int eosToken() {
            return 2;
          }
        };

    @Override
    public String name() {
      return "stub";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("fixture", "fixture", 16, 3, 2, 1, 1, 1);
    }

    @Override
    public int contextCapacity() {
      return 8;
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("stub");
    }

    @Override
    public Tokenizer tokenizer() {
      return tokenizer;
    }

    @Override
    public float[] forward(int token, int tokenPosition) {
      position = tokenPosition + 1;
      return terminalLogits();
    }

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      prefillStartPositions.add(startPosition);
      position = startPosition + tokens.length;
      return terminalLogits();
    }

    @Override
    public void reset() {
      position = 0;
    }

    @Override
    public int checkpoint() {
      return position;
    }

    @Override
    public void rewind(int checkpoint) {
      position = checkpoint;
    }

    @Override
    public void close() {
      closeCount++;
    }

    private static float[] terminalLogits() {
      return new float[] {0.0f, 0.0f, 10.0f};
    }
  }
}
