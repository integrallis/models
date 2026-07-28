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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJar;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;

@Tag("unit")
class PureJavaBackendModelJarsTest {

  private static final ModelJar QWEN3_Q4_0 =
      ModelJar.of("hf://ggml-org/Qwen3-0.6B-GGUF")
          .version("[3.0.0,4.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("text-generation");

  private static final ModelJar QWEN25_CODER_0_5B_Q4_0 =
      ModelJar.of("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF")
          .version("[2.5.0,3.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("code-completion");

  private static final ModelJar QWEN25_CODER_0_5B_Q8_0 =
      ModelJar.of("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF")
          .version("[2.5.0,3.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("code-completion");

  private static final ModelJar QWEN3_1_7B_Q8_0 =
      ModelJar.of("hf://Qwen/Qwen3-1.7B-GGUF")
          .version("[3.0.0,4.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("text-generation");

  private static final ModelJar QWEN25_CODER_1_5B_Q4_0 =
      ModelJar.of("hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF")
          .version("[2.5.0,3.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("code-completion");

  private static final ModelJar QWEN25_CODER_1_5B_Q8_0 =
      ModelJar.of("hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF")
          .version("[2.5.0,3.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("code-completion");

  private static final ModelJar QWEN25_CODER_3B_Q4_0 =
      ModelJar.of("hf://Qwen/Qwen2.5-Coder-3B-Instruct-GGUF")
          .version("[2.5.0,3.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("code-completion");

  private static final ModelJar QWEN25_CODER_7B_Q4_0 =
      ModelJar.of("hf://Qwen/Qwen2.5-Coder-7B-Instruct-GGUF")
          .version("[2.5.0,3.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("code-completion");

  private static final ModelJar HUATUOGPT_O1_7B_Q4_K_M =
      ModelJar.of("hf://bartowski/HuatuoGPT-o1-7B-GGUF")
          .version("[1.0.0,2.0.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("medical-reasoning");

  private static final ModelJar SMOLLM2_360M_Q8_0 =
      ModelJar.of("hf://HuggingFaceTB/SmolLM2-360M-Instruct-GGUF")
          .version("[2.0.0,3.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("chat");

  private static final ModelJar TINYLLAMA_1_1B_CHAT_V1_0_Q4_0 =
      ModelJar.of("hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
          .version("[1.0.0,2.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("chat");

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
  void resolvesQwen25CoderQ8PureJavaMarkerJarFromClasspath() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(QWEN25_CODER_0_5B_Q8_0).orElseThrow();

    assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
    assertThat(descriptor.format()).isEqualTo("gguf");
    assertThat(descriptor.architecture()).isEqualTo("qwen2");
    assertThat(descriptor.quantization()).isEqualTo("Q8_0");
    assertThat(descriptor.capabilities()).contains("text-generation", "chat", "code-completion");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/qwen2.5-coder-0.5b-instruct-q8_0.gguf");
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
  void resolvesQwen25Coder15BQ8PureJavaMarkerJarFromClasspath() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(QWEN25_CODER_1_5B_Q8_0).orElseThrow();

    assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
    assertThat(descriptor.format()).isEqualTo("gguf");
    assertThat(descriptor.architecture()).isEqualTo("qwen2");
    assertThat(descriptor.quantization()).isEqualTo("Q8_0");
    assertThat(descriptor.capabilities()).contains("text-generation", "chat", "code-completion");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/qwen2.5-coder-1.5b-instruct-q8_0.gguf");
  }

  @Test
  void resolvesQwen25Coder3BPureJavaMarkerJarFromClasspath() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(QWEN25_CODER_3B_Q4_0).orElseThrow();

    assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
    assertThat(descriptor.format()).isEqualTo("gguf");
    assertThat(descriptor.architecture()).isEqualTo("qwen2");
    assertThat(descriptor.quantization()).isEqualTo("Q4_0");
    assertThat(descriptor.capabilities()).contains("text-generation", "chat", "code-completion");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/qwen2.5-coder-3b-instruct-q4_0.gguf");
  }

  @Test
  void resolvesQwen25Coder7BPureJavaMarkerJarFromClasspath() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(QWEN25_CODER_7B_Q4_0).orElseThrow();

    assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
    assertThat(descriptor.format()).isEqualTo("gguf");
    assertThat(descriptor.architecture()).isEqualTo("qwen2");
    assertThat(descriptor.quantization()).isEqualTo("Q4_0");
    assertThat(descriptor.capabilities()).contains("text-generation", "chat", "code-completion");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/qwen2.5-coder-7b-instruct-q4_0.gguf");
  }

  @Test
  void resolvesHuatuoGptPureJavaMarkerJarFromClasspath() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(HUATUOGPT_O1_7B_Q4_K_M).orElseThrow();

    assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
    assertThat(descriptor.architecture()).isEqualTo("qwen2");
    assertThat(descriptor.quantization()).isEqualTo("Q4_K_M");
    assertThat(descriptor.capabilities()).contains("medical-reasoning", "bilingual");
    assertThat(descriptor.features()).contains("medical-use-warning");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/HuatuoGPT-o1-7B-Q4_K_M.gguf");
  }

  @Test
  void resolvesSmolLm2PureJavaMarkerJarFromClasspath() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(SMOLLM2_360M_Q8_0).orElseThrow();

    assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
    assertThat(descriptor.format()).isEqualTo("gguf");
    assertThat(descriptor.architecture()).isEqualTo("llama");
    assertThat(descriptor.quantization()).isEqualTo("Q8_0");
    assertThat(descriptor.capabilities()).contains("text-generation", "chat");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/smollm2-360m-instruct-q8_0.gguf");
  }

  @Test
  void resolvesQwen3OnePointSevenBPureJavaMarkerJarFromClasspath() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(QWEN3_1_7B_Q8_0).orElseThrow();

    assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
    assertThat(descriptor.format()).isEqualTo("gguf");
    assertThat(descriptor.architecture()).isEqualTo("qwen3");
    assertThat(descriptor.quantization()).isEqualTo("Q8_0");
    assertThat(descriptor.capabilities()).contains("text-generation", "chat");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/Qwen3-1.7B-Q8_0.gguf");
  }

  @Test
  void resolvesTinyLlamaSentencePieceMarkerJarFromClasspath() {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(TINYLLAMA_1_1B_CHAT_V1_0_Q4_0).orElseThrow();

    assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
    assertThat(descriptor.format()).isEqualTo("gguf");
    assertThat(descriptor.architecture()).isEqualTo("llama");
    assertThat(descriptor.quantization()).isEqualTo("Q4_0");
    assertThat(descriptor.capabilities()).contains("text-generation", "chat");
    assertThat(descriptor.localPath().orElseThrow().toString())
        .endsWith(".jvllm/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf");
  }
}
