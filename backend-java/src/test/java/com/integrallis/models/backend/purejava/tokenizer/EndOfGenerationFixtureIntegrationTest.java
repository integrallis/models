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
package com.integrallis.models.backend.purejava.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * End-of-generation sets resolved from pinned real GGUF fixtures, checked against the upstream
 * {@code generation_config.json} of each model (read 2026-09-16: Qwen2.5-Coder and Qwen3 declare
 * {@code [151645, 151643]}; Gemma 3 1B IT declares {@code [1, 106]}).
 */
@Tag("integration")
class EndOfGenerationFixtureIntegrationTest {

  private static final ModelFixtureRequirement QWEN25_CODER_0_5B_Q4_0 =
      ModelFixtureRequirement.of("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF")
          .version("[2.5.0,3.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("code-completion");

  private static final ModelFixtureRequirement GEMMA3_1B_Q4_K_M =
      ModelFixtureRequirement.of("hf://bartowski/google_gemma-3-1b-it-GGUF")
          .version("[3.0.0,4.0.0)")
          .variant("q4_k_m")
          .backend("rust-ffm")
          .capability("chat");

  @Test
  void qwenStopsOnItsDeclaredTerminatorsAndDecodesTheOrdinaryClosingTagAsText() throws IOException {
    GgufTokenizer tokenizer = load(QWEN25_CODER_0_5B_Q4_0);

    assertThat(tokenizer.endOfGenerationTokenIds()).containsExactly(151643, 151645);
    assertThat(tokenizer.tokenId("</s>")).isEqualTo(128247);
    assertThat(tokenizer.isEndOfGeneration(128247)).isFalse();
    assertThat(tokenizer.decode(128247)).isEqualTo("</s>");
  }

  @Test
  void gemma3StopsOnEosAndEndOfTurnButNotOnTheHtmlStrikethroughTag() throws IOException {
    GgufTokenizer tokenizer = load(GEMMA3_1B_Q4_K_M);

    assertThat(tokenizer.endOfGenerationTokenIds()).containsExactly(1, 106);
    assertThat(tokenizer.tokenId("</s>")).isEqualTo(212);
    assertThat(tokenizer.isEndOfGeneration(212)).isFalse();
  }

  @Test
  void gemma3AndMiniCpm5RecordTheirTemplatesEndOfTurnMarker() throws IOException {
    GgufTokenizer gemma = load(GEMMA3_1B_Q4_K_M);
    GgufTokenizer miniCpm =
        load(
            ModelFixtureRequirement.of("hf://openbmb/MiniCPM5-1B-GGUF")
                .version("[5.0.0,6.0.0)")
                .variant("q4_k_m")
                .backend("pure-java")
                .capability("text-generation"));

    // Both headers declare only id 1 as EOS; the turn ends on a different token.
    assertThat(gemma.chatTemplateEndOfTurnResolution()).isEqualTo("resolved:106");
    assertThat(gemma.endOfGenerationSources())
        .containsEntry(1, List.of("tokenizer.ggml.eos_token_id", "vocabulary-text"))
        .containsEntry(106, List.of("vocabulary-text", "chat-template-end-of-turn"));
    assertThat(miniCpm.chatTemplateEndOfTurnResolution()).isEqualTo("resolved:130073");
    assertThat(miniCpm.endOfGenerationSources())
        .containsEntry(130073, List.of("vocabulary-text", "chat-template-end-of-turn"));
  }

  /**
   * Every pinned GGUF fixture paired with the native template ModelJars serves it with stops where
   * that template ends an assistant turn.
   */
  @ParameterizedTest
  @CsvSource(
      delimiter = '|',
      value = {
        "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF | [2.5.0,3.0.0) | q4_0 | pure-java | code-completion | CHATML",
        "hf://ggml-org/Qwen3-0.6B-GGUF | [3.0.0,4.0.0) | q4_0 | pure-java | text-generation | CHATML_NO_THINK",
        "hf://HuggingFaceTB/SmolLM2-360M-Instruct-GGUF | [2.0.0,3.0.0) | q8_0 | pure-java | chat | CHATML",
        "hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF | [1.0.0,2.0.0) | q4_0 | pure-java | chat | ZEPHYR",
        "hf://bartowski/google_gemma-3-1b-it-GGUF | [3.0.0,4.0.0) | q4_k_m | rust-ffm | chat | GEMMA",
        "hf://openbmb/MiniCPM5-1B-GGUF | [5.0.0,6.0.0) | q4_k_m | pure-java | text-generation | MINICPM5_NO_THINK",
        "hf://TheBloke/deepseek-coder-1.3b-instruct-GGUF | [1.3.0,2.0.0) | q4_k_m | pure-java | code-completion | DEEPSEEK",
        "hf://mradermacher/EuroLLM-1.7B-Instruct-GGUF | [1.0.0,2.0.0) | q4_k_m | pure-java | translation | CHATML"
      })
  void nativeTemplateEndOfTurnIsInTheStopSet(
      String coordinate,
      String versions,
      String variant,
      String backend,
      String capability,
      ChatTemplate template)
      throws IOException {
    GgufTokenizer tokenizer =
        load(
            ModelFixtureRequirement.of(coordinate)
                .version(versions)
                .variant(variant)
                .backend(backend)
                .capability(capability));

    assertThat(template.endOfTurnTokenId(tokenizer)).isPresent();
    assertThat(template.endOfTurnStopsGeneration(tokenizer)).isTrue();
  }

  private static GgufTokenizer load(ModelFixtureRequirement requirement) throws IOException {
    Path path =
        ModelFixtureRegistry.fromClasspath()
            .resolve(requirement)
            .orElseThrow(() -> new IllegalStateException("no fixture for " + requirement))
            .localPath()
            .orElseThrow();
    try (Arena arena = Arena.ofConfined()) {
      return GgufTokenizer.fromMetadata(GgufParser.parse(path, arena).metadata());
    }
  }
}
