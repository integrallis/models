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
package com.integrallis.models.backend.purejava.soprano;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.gguf.GgufEmbeddedFiles;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class SopranoTokenizerIntegrationTest {

  @Test
  void matchesTheOfficialEmbeddedTokenizerPipeline() throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(Path.of(configured), arena);
      var embedded = GgufEmbeddedFiles.from(file.metadata());
      var tokenizer = SopranoTokenizer.fromJson(embedded.readUtf8("tokenizer.json"));

      assertThat(tokenizer.encodePrompt("The JVM can speak for itself."))
          .containsExactly(
              3, 1, 8063, 8004, 8040, 8052, 8043, 8004, 8141, 8004, 8049, 8163, 8031, 8041, 8004,
              8109, 8004, 8066, 8079, 8042, 8036, 8015, 2);
      assertThat(tokenizer.encodePrompt("Route 123, now!"))
          .containsExactly(
              3, 1, 8048, 8152, 8035, 8004, 8018, 8019, 8020, 8013, 8004, 8117, 8005, 2);
      assertThat(tokenizer.encodePrompt("Repeated   whitespace\tworks."))
          .containsExactly(
              3, 1, 8060, 8163, 8062, 8111, 8004, 8090, 8066, 8125, 8046, 8115, 8035, 8004, 8053,
              8076, 8041, 8049, 8015, 2);
      assertThat(tokenizer.encodePrompt("Café costs €5."))
          .containsExactly(3, 1, 8165, 8036, 0, 8004, 8114, 8073, 8049, 8004, 0, 8022, 8015, 2);
      assertThat(tokenizer.encodePrompt("“Hello”—world?"))
          .containsExactly(3, 1, 0, 8077, 8070, 8045, 0, 0, 8053, 8076, 8139, 8029, 2);
    }
  }
}
