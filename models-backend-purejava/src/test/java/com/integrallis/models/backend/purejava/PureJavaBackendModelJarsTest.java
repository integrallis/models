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

import java.io.IOException;
import java.nio.file.Path;
import java.util.Properties;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;
import org.modeljars.PropertiesModelJarRegistry;

@Tag("unit")
class PureJavaBackendModelJarsTest {

  private static final ModelJarRequirement QWEN3_Q4_0 =
      ModelJarRequirement.forSource("hf://ggml-org/Qwen3-0.6B-GGUF")
          .versionRange("[3.0.0,4.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("text-generation")
          .build();

  private static final ModelJarRequirement QWEN25_CODER_0_5B_Q4_0 =
      ModelJarRequirement.forSource("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF")
          .versionRange("[2.5.0,3.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("code-completion")
          .build();

  private static final ModelJarRequirement QWEN25_CODER_1_5B_Q4_0 =
      ModelJarRequirement.forSource("hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF")
          .versionRange("[2.5.0,3.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("code-completion")
          .build();

  @Test
  void resolvesQwenMarkerJarFromClasspath() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(QWEN3_Q4_0).orElseThrow();

    assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
    assertThat(descriptor.format()).isEqualTo("gguf");
    assertThat(descriptor.quantization()).isEqualTo("Q4_0");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/Qwen3-0.6B-Q4_0.gguf");
  }

  @Test
  void resolvesQwen25CoderPureJavaMarkerJarFromClasspath() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(QWEN25_CODER_0_5B_Q4_0).orElseThrow();

    assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
    assertThat(descriptor.format()).isEqualTo("gguf");
    assertThat(descriptor.architecture()).isEqualTo("qwen2");
    assertThat(descriptor.quantization()).isEqualTo("Q4_0");
    assertThat(descriptor.capabilities()).contains("text-generation", "chat", "code-completion");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/qwen2.5-coder-0.5b-instruct-q4_0.gguf");
  }

  @Test
  void resolvesQwen25Coder15BPureJavaMarkerJarFromClasspath() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(QWEN25_CODER_1_5B_Q4_0).orElseThrow();

    assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
    assertThat(descriptor.format()).isEqualTo("gguf");
    assertThat(descriptor.architecture()).isEqualTo("qwen2");
    assertThat(descriptor.quantization()).isEqualTo("Q4_0");
    assertThat(descriptor.capabilities()).contains("text-generation", "chat", "code-completion");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf");
  }

  @Test
  void plainJavaAppCanLoadBackendFromModelJarDescriptor(@TempDir Path dir) throws IOException {
    Path modelPath = PureJavaBackendTest.buildNanoModelFile(dir, new Random(11));
    ModelJarDescriptor descriptor = syntheticDescriptor(modelPath);

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor)) {
      assertThat(backend.name()).isEqualTo("pure-java");
      assertThat(backend.metadata().modelName()).isEqualTo("NanoTest");
      assertThat(backend.forward(5, 0)).hasSize(32);
    }
  }

  private static ModelJarDescriptor syntheticDescriptor(Path modelPath) {
    Properties properties = new Properties();
    properties.setProperty("model.nano.sourceId", "local://nano");
    properties.setProperty("model.nano.markerCoordinate", "org.modeljars.local:nano:0.1.0");
    properties.setProperty("model.nano.modelVersion", "0.1.0");
    properties.setProperty("model.nano.variant", "f32");
    properties.setProperty("model.nano.format", "gguf");
    properties.setProperty("model.nano.architecture", "llama");
    properties.setProperty("model.nano.quantization", "F32");
    properties.setProperty("model.nano.path", modelPath.toString());
    properties.setProperty("model.nano.capabilities", "text-generation");
    properties.setProperty("model.nano.backend.pure-java", "true");
    return PropertiesModelJarRegistry.fromProperties(properties)
        .resolve(ModelJarRequirement.forSource("local://nano").backend("pure-java").build())
        .orElseThrow();
  }
}
