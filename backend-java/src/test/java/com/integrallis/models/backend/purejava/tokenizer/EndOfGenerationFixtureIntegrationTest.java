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
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

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

  private static GgufTokenizer load(ModelFixtureRequirement requirement) throws IOException {
    Path path =
        ModelFixtureRegistry.fromClasspath()
            .resolve(requirement)
            .orElseThrow()
            .localPath()
            .orElseThrow();
    try (Arena arena = Arena.ofConfined()) {
      return GgufTokenizer.fromMetadata(GgufParser.parse(path, arena).metadata());
    }
  }
}
