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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.tokenizer.HuggingFaceTokenizer;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("model-fixture")
class MobileMoeTokenizerIntegrationTest {

  @Test
  void matchesThePinnedLlama3TokenizerOracle() throws Exception {
    Path directory = fixtureDirectory();
    MobileMoeHuggingFaceConfig config =
        MobileMoeHuggingFaceConfig.parse(directory.resolve("config.json"));
    Tokenizer tokenizer =
        HuggingFaceTokenizer.fromMobileMoe(
            directory.resolve("tokenizer.json"),
            directory.resolve("tokenizer_config.json"),
            config);

    assertThat(tokenizer.encode("Hello, MobileMoE!"))
        .containsExactly(128_000, 9_906, 11, 13_716, 26_694, 36, 0);
    assertThat(tokenizer.decode(new int[] {9_906, 11, 13_716, 26_694, 36, 0}))
        .isEqualTo("Hello, MobileMoE!");

    assertThat(
            tokenizer.encode(
                ChatTemplate.MOBILE_MOE.render(
                    List.of(ChatMessage.user("Name one JVM language.")))))
        .containsExactly(
            128_000, 128_005, 882, 128_006, 271, 678, 832, 73_479, 4_221, 13, 128_008, 128_005,
            78_191, 128_006, 271);
  }

  private static Path fixtureDirectory() {
    String configured = System.getProperty("models.fixtures.mobileMoeQatDirectory", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.mobileMoeQatDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isDirectory(directory), "MobileMoE QAT fixture is not installed");
    return directory;
  }
}
