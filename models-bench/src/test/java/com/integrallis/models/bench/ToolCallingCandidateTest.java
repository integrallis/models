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
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import com.integrallis.models.runtime.chat.ChatTemplate;
import org.junit.jupiter.api.Test;

class ToolCallingCandidateTest {

  @Test
  void pinsTheSixQualificationCandidatesAndTheirProtocols() {
    assertThat(ToolCallingCandidate.values()).hasSize(6);
    assertThat(ToolCallingCandidate.parse("needle2").template()).isEqualTo(ChatTemplate.NEEDLE2);
    assertThat(ToolCallingCandidate.parse("qwen3-0.6b").template())
        .isEqualTo(ChatTemplate.CHATML_NO_THINK);
    assertThat(ToolCallingCandidate.parse("qwen3-1.7b").template())
        .isEqualTo(ChatTemplate.CHATML_NO_THINK);
    assertThat(ToolCallingCandidate.parse("smollm3-3b").template())
        .isEqualTo(ChatTemplate.SMOLLM3_NO_THINK);
    assertThat(ToolCallingCandidate.parse("minicpm5-1b").template())
        .isEqualTo(ChatTemplate.MINICPM5_NO_THINK);
    assertThat(ToolCallingCandidate.parse("llama3.2-3b").template()).isEqualTo(ChatTemplate.LLAMA3);
  }

  @Test
  void rejectsAnUnpinnedCandidate() {
    assertThatIllegalArgumentException()
        .isThrownBy(() -> ToolCallingCandidate.parse("some-local-file"))
        .withMessageContaining("unknown tool-calling candidate");
  }
}
