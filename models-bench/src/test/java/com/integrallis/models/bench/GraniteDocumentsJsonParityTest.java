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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.runtime.chat.GraniteDocumentsPrompt;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The documents renderer moved out of Jackson's reach; its string escaping must still produce the
 * bytes the 620-prompt oracle comparison was run with, which Jackson generated.
 */
class GraniteDocumentsJsonParityTest {
  @Test
  void documentJsonMatchesJacksonForEveryEscapeClass() throws Exception {
    ObjectMapper jackson = new ObjectMapper();
    List<String> corpus =
        List.of(
            "",
            "plain text",
            "quotes \"inside\" and back\\slash",
            "line\nbreak tab\t cr\r bs\b ff\f",
            "del \u007f passes through",
            "caf\u00e9 \u2014 \u65e5\u672c\u8a9e \ud83d\ude00 emoji",
            "<documents> & 'apostrophe' / slash",
            "{\"doc_id\": 1, \"text\": \"nested\"}");
    for (String text : corpus) {
      assertThat(GraniteDocumentsPrompt.documentJson(7, text))
          .as(text)
          .isEqualTo("{\"doc_id\": 7, \"text\": " + jackson.writeValueAsString(text) + "}");
    }
  }

  @Test
  void rareControlCharactersFollowPythonNotJackson() {
    // json.dumps writes lowercase hex digits; Jackson writes uppercase. The oracle is Python
    // (Transformers), and no oracle-compared document contained such a character.
    assertThat(GraniteDocumentsPrompt.documentJson(1, "a\u001fb"))
        .isEqualTo("{\"doc_id\": 1, \"text\": \"a\\u001fb\"}");
  }
}
