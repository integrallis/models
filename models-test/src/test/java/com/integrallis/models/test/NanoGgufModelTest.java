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
package com.integrallis.models.test;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class NanoGgufModelTest {

  @Test
  void writesDeterministicGgufThatTheJavaBackendCanExecute(@TempDir Path directory)
      throws Exception {
    Path first = NanoGgufModel.write(directory.resolve("first.gguf"));
    Path second = NanoGgufModel.write(directory.resolve("second.gguf"));

    assertThat(Files.readAllBytes(first)).isEqualTo(Files.readAllBytes(second));

    try (var backend = PureJavaBackend.load(first)) {
      assertThat(backend.metadata().modelName()).isEqualTo("ModelsNano");
      assertThat(backend.metadata().modelFamily()).isEqualTo("llama");
      assertThat(backend.tokenizer().vocabSize()).isEqualTo(32);
      assertThat(backend.forward(5, 0)).hasSize(32);
    }
  }
}
