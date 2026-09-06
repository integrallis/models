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
package com.integrallis.models.backend.purejava.deberta;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.tokenizer.DebertaV2Tokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("model-fixture")
class MxbaiDebertaTokenizerIntegrationTest {

  @Test
  void matchesThePinnedTransformersPairTokenizer() throws Exception {
    Path directory = fixtureDirectory();
    DebertaV2Tokenizer tokenizer =
        DebertaV2Tokenizer.fromJson(directory.resolve("tokenizer.json"), 512);

    assertThat(
            tokenizer
                .encodePair(
                    "What is the population of Berlin?",
                    "Berlin has a population of about 3.7 million people.")
                .tokens())
        .containsExactly(
            1, 458, 269, 262, 1755, 265, 5842, 302, 2, 5842, 303, 266, 1755, 265, 314, 404, 260,
            819, 705, 355, 260, 2);
    assertThat(
            tokenizer
                .encodePair(
                    "What is the population of Berlin?",
                    "Berlin has a population of about 3.7 million people.")
                .tokenTypes())
        .containsExactly(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

    assertThat(tokenizer.encodePair("Hello, world!", "  Multiple   spaces and café. ").tokens())
        .containsExactly(1, 5365, 261, 447, 300, 2, 10189, 3654, 263, 15924, 260, 2);
  }

  private static Path fixtureDirectory() {
    String configured = System.getProperty("models.fixtures.mxbaiRerankerDirectory", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.mxbaiRerankerDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isRegularFile(directory.resolve("tokenizer.json")));
    return directory;
  }
}
