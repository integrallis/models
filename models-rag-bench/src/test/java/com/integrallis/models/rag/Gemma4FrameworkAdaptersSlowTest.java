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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.langchain4j.ModelsChatModel;
import com.integrallis.models.runtime.RuntimeTextGenerationModel;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.spring.ai.ModelsSpringAiChatModel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

@Tag("slow")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class Gemma4FrameworkAdaptersSlowTest {
  private static final String FILE_NAME = "gemma-4-26B-A4B-it-Q4_K_M.gguf";
  private static final long FILE_SIZE = 16_796_015_136L;
  private static final String EXPECTED = "Hello! How can I help you today?";
  private static final SamplingOptions GREEDY =
      SamplingOptions.builder()
          .temperature(0.0f)
          .topP(1.0f)
          .topK(1)
          .maxTokens(10)
          .seed(42L)
          .build();

  private String previousContext;
  private String previousNativeDecode;
  private String previousLoadWarmup;
  private RustFfmBackend backend;

  @BeforeAll
  void loadPinnedArtifact() {
    Path model = fixturePath();
    assertThat(model).isRegularFile();
    assertThat(fileSize(model)).isEqualTo(FILE_SIZE);

    previousContext = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    previousNativeDecode = System.getProperty(RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY);
    previousLoadWarmup = System.getProperty(RustFfmBackend.LOAD_WARMUP_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "512");
    System.setProperty(RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY, "true");
    System.setProperty(RustFfmBackend.LOAD_WARMUP_PROPERTY, "true");
    backend = RustFfmBackend.load(model);
  }

  @AfterAll
  void closeBackend() {
    try {
      if (backend != null) {
        backend.close();
      }
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previousContext);
      restoreSystemProperty(
          RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY, previousNativeDecode);
      restoreSystemProperty(RustFfmBackend.LOAD_WARMUP_PROPERTY, previousLoadWarmup);
    }
  }

  @BeforeEach
  void resetSequence() {
    backend.reset();
  }

  @Test
  void plainJavaRuntimeGeneratesTheQualifiedAnswer() {
    RuntimeTextGenerationModel model = new RuntimeTextGenerationModel(backend);

    assertThat(
            model.generate(ChatTemplate.GEMMA4.render(List.of(ChatMessage.user("Hello"))), GREEDY))
        .isEqualTo(EXPECTED);
  }

  @Test
  void langChain4jAdapterGeneratesTheQualifiedAnswer() {
    ModelsChatModel model = new ModelsChatModel(backend, ChatTemplate.GEMMA4, GREEDY);

    assertThat(model.chat("Hello")).isEqualTo(EXPECTED);
  }

  @Test
  void springAiAdapterGeneratesTheQualifiedAnswer() {
    ModelsSpringAiChatModel model =
        new ModelsSpringAiChatModel(backend, ChatTemplate.GEMMA4, GREEDY);

    assertThat(model.call("Hello")).isEqualTo(EXPECTED);
  }

  private static Path fixturePath() {
    String configured = System.getProperty("models.fixtures.directory");
    Path directory =
        configured == null || configured.isBlank()
            ? Path.of(System.getProperty("user.home"), ".jvllm", "models")
            : Path.of(configured);
    return directory.resolve(FILE_NAME);
  }

  private static long fileSize(Path path) {
    try {
      return Files.size(path);
    } catch (java.io.IOException failure) {
      throw new IllegalStateException("Cannot read Gemma 4 fixture size", failure);
    }
  }

  private static void restoreSystemProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }
}
