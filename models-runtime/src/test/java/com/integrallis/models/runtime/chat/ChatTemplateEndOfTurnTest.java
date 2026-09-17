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
package com.integrallis.models.runtime.chat;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.Tokenizer;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

@Tag("unit")
class ChatTemplateEndOfTurnTest {

  /**
   * The declared marker is what the renderer itself writes after an assistant message, so the table
   * cannot drift from the envelope. GPT-OSS is the documented exception: history closes a turn with
   * {@code <|end|>}, but a final answer is terminated by {@code <|return|>}.
   */
  @ParameterizedTest
  @EnumSource(
      value = ChatTemplate.class,
      names = {"RAW", "GPT_OSS", "NEEDLE2", "DEEPSEEK"},
      mode = EnumSource.Mode.EXCLUDE)
  void theRendererClosesAnAssistantMessageWithTheDeclaredMarker(ChatTemplate template) {
    String rendered =
        template
            .render(
                List.of(
                    ChatMessage.user("question"),
                    ChatMessage.assistant("ANSWER"),
                    ChatMessage.user("again")))
            .text();

    String marker = template.endOfTurnMarker().orElseThrow();
    assertThat(rendered).contains("ANSWER" + marker);
  }

  @Test
  void familiesWhoseAssistantTextIsNotDirectlyFollowedByTheirTerminator() {
    assertThat(ChatTemplate.RAW.endOfTurnMarker()).isEmpty();
    assertThat(ChatTemplate.GPT_OSS.endOfTurnMarker()).contains("<|return|>");
    assertThat(ChatTemplate.DEEPSEEK.endOfTurnMarker()).contains("<|EOT|>");
    assertThat(ChatTemplate.NEEDLE2.endOfTurnMarker()).contains("<|im_end|>");
    assertThat(render(ChatTemplate.DEEPSEEK)).contains("ANSWER\n<|EOT|>");
    assertThat(render(ChatTemplate.NEEDLE2)).contains("<|im_end|>");
  }

  @Test
  void resolvesTheMarkerAgainstALoadedVocabularyAndReportsWhetherItStopsGeneration() {
    Tokenizer stopsOnMarker = tokenizer(7, Set.of(7));
    Tokenizer ignoresMarker = tokenizer(7, Set.of(2));
    Tokenizer lacksMarker = tokenizer(-1, Set.of(2));

    assertThat(ChatTemplate.GEMMA.endOfTurnTokenId(stopsOnMarker)).hasValue(7);
    assertThat(ChatTemplate.GEMMA.endOfTurnStopsGeneration(stopsOnMarker)).isTrue();
    assertThat(ChatTemplate.GEMMA.endOfTurnStopsGeneration(ignoresMarker)).isFalse();
    assertThat(ChatTemplate.GEMMA.endOfTurnTokenId(lacksMarker)).isEmpty();
    assertThat(ChatTemplate.GEMMA.endOfTurnStopsGeneration(lacksMarker)).isFalse();
    assertThat(ChatTemplate.RAW.endOfTurnStopsGeneration(ignoresMarker)).isTrue();
  }

  private static String render(ChatTemplate template) {
    return template
        .render(
            List.of(
                ChatMessage.user("question"),
                ChatMessage.assistant("ANSWER"),
                ChatMessage.user("x")))
        .text();
  }

  private static Tokenizer tokenizer(int markerId, Set<Integer> terminators) {
    return new Tokenizer() {
      @Override
      public int[] encode(String text) {
        return new int[0];
      }

      @Override
      public String decode(int[] tokens) {
        return "";
      }

      @Override
      public String decode(int token) {
        return "";
      }

      @Override
      public int tokenId(String text) {
        return Optional.of(text)
            .filter("<end_of_turn>"::equals)
            .map(ignored -> markerId)
            .orElse(-1);
      }

      @Override
      public int vocabSize() {
        return 8;
      }

      @Override
      public int bosToken() {
        return 0;
      }

      @Override
      public int eosToken() {
        return 2;
      }

      @Override
      public boolean isEndOfGeneration(int token) {
        return terminators.contains(token);
      }
    };
  }
}
