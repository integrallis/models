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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NativeHuggingFaceTokenizerTest {
  @TempDir Path temporary;

  @Test
  void recognizesAddedTokensAndOwnsItsLifecycle() throws Exception {
    Path json = temporary.resolve("tokenizer.json");
    Files.writeString(
        json,
        """
        {"version":"1.0","truncation":null,"padding":null,
        "added_tokens":[{"id":2,"content":"[L]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"[L]":2},"unk_token":"[UNK]"}}
        """);
    var tokenizer =
        NativeHuggingFaceTokenizer.load(
            Path.of(System.getProperty("models.native.kernels.library")), json);
    try {
      assertThat(tokenizer.encodePiece("hello [L] world")).containsExactly(1, 2, 0);
      assertThat(tokenizer.encodePiece("")).isEmpty();
      int[] output = tokenizer.encodePiece("hello");
      output[0] = 99;
      assertThat(tokenizer.encodePiece("hello")).containsExactly(1);
    } finally {
      tokenizer.close();
    }
    tokenizer.close();
    assertThatThrownBy(() -> tokenizer.encodePiece("hello"))
        .isInstanceOf(IllegalStateException.class);
  }
}
