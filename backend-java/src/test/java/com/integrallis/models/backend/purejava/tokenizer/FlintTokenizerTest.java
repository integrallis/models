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

import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class FlintTokenizerTest {
  @Test
  void splitsCjkSeparatelyFromLatinAndGroupsThreeDigits() {
    var pre = BpePreTokenizer.forName("flint");
    assertThat(pre.isImplemented()).isTrue();
    assertThat(pre.ignoresMerges()).isFalse();
    assertThat(pre.split("hello世界abc20261029"))
        .containsExactly("hello", "世界", "abc", "202", "610", "29");
    assertThat(pre.split("don't stop\n")).containsExactly("don't", " stop", "\n");
  }

  @Test
  void speakerTransitionsStopOnlyFlintGeneration(@TempDir Path directory) throws Exception {
    var tokens =
        List.of(
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|user|>",
            "<|assistant|>",
            "<|observation|>",
            "<|developer|>",
            "<think>",
            "</think>");
    for (String architecture : List.of("flint", "llama")) {
      var builder =
          new SyntheticGgufBuilder()
              .addString("general.architecture", architecture)
              .addStringArray("tokenizer.ggml.tokens", tokens)
              .addInt32Array("tokenizer.ggml.token_type", List.of(3, 3, 3, 3, 3, 3, 4, 4))
              .addUint32("tokenizer.ggml.eos_token_id", 1);
      Path path = directory.resolve(architecture + ".gguf");
      Files.write(path, builder.build());
      try (Arena arena = Arena.ofConfined()) {
        var tokenizer = GgufTokenizer.fromMetadata(GgufParser.parse(path, arena).metadata());
        for (int id : new int[] {2, 4, 5}) {
          assertThat(tokenizer.isEndOfGeneration(id)).isEqualTo(architecture.equals("flint"));
        }
        assertThat(tokenizer.isEndOfGeneration(1)).isTrue();
        for (int id : new int[] {0, 3, 6, 7}) assertThat(tokenizer.isEndOfGeneration(id)).isFalse();
      }
    }
  }
}
