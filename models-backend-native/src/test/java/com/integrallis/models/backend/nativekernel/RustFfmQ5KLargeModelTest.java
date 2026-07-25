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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
class RustFfmQ5KLargeModelTest {
  private static final Path SQLCODER_MODEL_PATH =
      Path.of(System.getProperty("user.home"), ".jvllm", "models", "sqlcoder-7b-q5_k_m.gguf");

  @Test
  void groupedKernelMatchesRealSqlCoderQ5_KQ5_KQ6_KAttentionWeights() throws Exception {
    assertThat(SQLCODER_MODEL_PATH)
        .as("the SQLCoder slow-test task must resolve the pinned Q5_K_M fixture")
        .isRegularFile();
    Path library = Path.of(System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY));
    assertThat(library).isRegularFile();

    try (Arena arena = Arena.ofShared();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(library, true)) {
      var file = GgufParser.parse(SQLCODER_MODEL_PATH, arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights.LayerWeights layer = LlamaWeights.fromGgufFile(file, config).layer(0);
      assertThat(layer.wqType()).isEqualTo(GgufTensorType.Q5_K);
      assertThat(layer.wkType()).isEqualTo(GgufTensorType.Q5_K);
      assertThat(layer.wvType()).isEqualTo(GgufTensorType.Q6_K);

      float[] input = deterministicInput(config.embeddingDim());
      float[] expectedQuery =
          referenceProjection(
              input, layer.wq(), layer.wqType(), config.queryDim(), config.embeddingDim());
      float[] expectedKey =
          referenceProjection(
              input, layer.wk(), layer.wkType(), config.keyDim(), config.embeddingDim());
      float[] expectedValue =
          referenceProjection(
              input, layer.wv(), layer.wvType(), config.valueDim(), config.embeddingDim());
      float[] actualQuery = new float[expectedQuery.length];
      float[] actualKey = new float[expectedKey.length];
      float[] actualValue = new float[expectedValue.length];

      assertThat(
              kernel.isTripleEligible(
                  layer.wqType(),
                  config.queryDim(),
                  layer.wkType(),
                  config.keyDim(),
                  layer.wvType(),
                  config.valueDim(),
                  1,
                  config.embeddingDim()))
          .isTrue();
      kernel.multiplyTriple(
          actualQuery,
          layer.wq(),
          layer.wqType(),
          config.queryDim(),
          actualKey,
          layer.wk(),
          layer.wkType(),
          config.keyDim(),
          actualValue,
          layer.wv(),
          layer.wvType(),
          config.valueDim(),
          input,
          1,
          config.embeddingDim());

      assertProjectionMatches(actualQuery, expectedQuery, "query");
      assertProjectionMatches(actualKey, expectedKey, "key");
      assertProjectionMatches(actualValue, expectedValue, "value");
    }
  }

  private static float[] deterministicInput(int size) {
    float[] input = new float[size];
    for (int index = 0; index < size; index++) {
      input[index] = ((index * 17) % 29 - 14) * 0.0078125f;
    }
    return input;
  }

  private static float[] referenceProjection(
      float[] input, MemorySegment weights, GgufTensorType type, int rows, int cols) {
    float[] output = new float[rows];
    TensorOps.ggufMatmul(output, input, weights, type, rows, cols);
    return output;
  }

  private static void assertProjectionMatches(float[] actual, float[] expected, String projection) {
    assertThat(actual).hasSameSizeAs(expected);
    for (int index = 0; index < actual.length; index++) {
      assertThat(actual[index])
          .as("%s projection row %s", projection, index)
          .isCloseTo(expected[index], within(1e-4f));
    }
  }
}
