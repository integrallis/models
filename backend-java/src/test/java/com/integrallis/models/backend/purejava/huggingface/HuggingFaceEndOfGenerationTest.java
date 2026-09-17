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
package com.integrallis.models.backend.purejava.huggingface;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class HuggingFaceEndOfGenerationTest {

  @Test
  void readsASingleIntegerEosFromGenerationConfig(@TempDir Path directory) throws IOException {
    Files.writeString(directory.resolve("generation_config.json"), "{\"eos_token_id\": 7}");

    assertThat(HuggingFaceEndOfGeneration.tokenIds(directory)).containsExactly(7);
  }

  @Test
  void readsAListOfEosIdsFromGenerationConfig(@TempDir Path directory) throws IOException {
    // Shape of Qwen2.5-Instruct's generation_config.json.
    Files.writeString(
        directory.resolve("generation_config.json"),
        """
        {"bos_token_id": 151643, "do_sample": true, "eos_token_id": [151645, 151643],
         "pad_token_id": 151643, "temperature": 0.7}
        """);

    assertThat(HuggingFaceEndOfGeneration.tokenIds(directory)).containsExactly(151645, 151643);
  }

  @Test
  void unionsConfigAndGenerationConfigInDeclarationOrderWithoutDuplicates(@TempDir Path directory)
      throws IOException {
    Files.writeString(
        directory.resolve("config.json"),
        "{\"model_type\": \"x\", \"eos_token_id\": [200002, 199999], \"nested\": {\"a\": [1]}}");
    Files.writeString(
        directory.resolve("generation_config.json"),
        "{\"eos_token_id\": [200002, 199999, 200012]}");

    assertThat(HuggingFaceEndOfGeneration.tokenIds(directory))
        .containsExactly(200002, 199999, 200012);
  }

  @Test
  void missingFilesAndMissingOrNullKeysContributeNothing(@TempDir Path directory)
      throws IOException {
    assertThat(HuggingFaceEndOfGeneration.tokenIds(directory)).isEmpty();

    Files.writeString(directory.resolve("config.json"), "{\"eos_token_id\": null}");
    Files.writeString(directory.resolve("generation_config.json"), "{\"do_sample\": false}");

    assertThat(HuggingFaceEndOfGeneration.tokenIds(directory)).isEmpty();
  }

  @Test
  void rejectsMalformedEosDeclarations(@TempDir Path directory) throws IOException {
    Path generationConfig = directory.resolve("generation_config.json");

    Files.writeString(generationConfig, "{\"eos_token_id\": \"</s>\"}");
    assertThatThrownBy(() -> HuggingFaceEndOfGeneration.tokenIds(directory))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("eos_token_id");

    Files.writeString(generationConfig, "{\"eos_token_id\": [1, -2]}");
    assertThatThrownBy(() -> HuggingFaceEndOfGeneration.tokenIds(directory))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("eos_token_id");

    Files.writeString(generationConfig, "{\"eos_token_id\": [1, [2]]}");
    assertThatThrownBy(() -> HuggingFaceEndOfGeneration.tokenIds(directory))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("eos_token_id");
  }
}
