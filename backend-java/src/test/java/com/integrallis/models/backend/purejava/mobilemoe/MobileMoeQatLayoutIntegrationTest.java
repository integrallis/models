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
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import com.integrallis.models.backend.purejava.tensor.TensorStorage;
import com.integrallis.models.backend.purejava.tensor.TensorView;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("model-fixture")
class MobileMoeQatLayoutIntegrationTest {

  private static final String Q_PROJECTION = "model.layers.0.self_attn.q_proj";

  @Test
  void pinnedSmallQatProjectionMatchesTheIndependentSafetensorsOracle() throws Exception {
    Path directory = fixtureDirectory();
    assertThat(Files.size(directory.resolve("model.safetensors"))).isEqualTo(713_916_240L);
    MobileMoeHuggingFaceConfig config =
        MobileMoeHuggingFaceConfig.parse(directory.resolve("config.json"));

    try (Arena arena = Arena.ofConfined()) {
      SafetensorsTensorSource source =
          new SafetensorsTensorSource(
              SafetensorsBundle.open(directory.resolve("model.safetensors"), arena));
      assertThat(source.tensorNames()).hasSize(443);
      TensorView qweight = source.tensor(Q_PROJECTION + ".qweight");
      TensorView scale = source.tensor(Q_PROJECTION + ".weight_scale");
      assertThat(qweight.shape()).containsExactly(768, 384);
      assertThat(qweight.storage()).isEqualTo(new TensorStorage("safetensors", "U8", 1, 1));
      assertThat(scale.shape()).containsExactly(768, 24);
      assertThat(scale.storage())
          .isEqualTo(new TensorStorage("safetensors", "F16", 1, Short.BYTES));

      MobileMoePackedInt4Matrix matrix =
          MobileMoePackedInt4Matrix.of(
              qweight.data(),
              scale.data(),
              config.queryDimension(),
              config.hiddenSize(),
              config.quantization().orElseThrow().groupSize());
      float[] input = new float[config.hiddenSize()];
      for (int index = 0; index < input.length; index++) {
        input[index] = ((index % 17) - 8) / 16.0f;
      }
      float[] output = new float[config.queryDimension()];

      matrix.multiply(input, output);

      float[] expected = {
        -0.191688418f,
        -0.0102274418f,
        -0.0554738045f,
        0.115440249f,
        0.075104773f,
        -0.0938560963f,
        0.110870838f,
        0.160048127f
      };
      assertThat(output).startsWith(expected, within(2.0e-6f));
      double sum = 0.0;
      for (float value : output) {
        sum += value;
      }
      assertThat(sum).isCloseTo(-2.4575814306735992, within(2.0e-5));
    }
  }

  @Test
  void loadsEveryRequiredQatTensorAndReadsTheTiedEmbeddingInPlace() throws Exception {
    Path directory = fixtureDirectory();
    MobileMoeHuggingFaceConfig config =
        MobileMoeHuggingFaceConfig.parse(directory.resolve("config.json"));
    try (Arena arena = Arena.ofConfined()) {
      SafetensorsTensorSource source =
          new SafetensorsTensorSource(
              SafetensorsBundle.open(directory.resolve("model.safetensors"), arena));

      MobileMoeWeights weights = MobileMoeWeights.load(source, config);

      float[] expected = {
        0.0222625732f,
        0.0278282166f,
        0.0222625732f,
        0.0f,
        0.0278282166f,
        -0.0389595032f,
        0.0f,
        0.0166969299f
      };
      float[] actual = new float[expected.length];
      for (int index = 0; index < actual.length; index++) {
        actual[index] = weights.embedding().value(9_906, index);
      }
      assertThat(actual).containsExactly(expected, within(1.0e-8f));
      assertThat(weights.outputNorm()).hasSize(768);
      assertThat(weights.layer(19).expertBias()).hasSize(60);
    }
  }

  private static Path fixtureDirectory() {
    String configured = System.getProperty("models.fixtures.mobileMoeQatDirectory", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.mobileMoeQatDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isDirectory(directory), "MobileMoE QAT fixture is not installed");
    return directory;
  }
}
