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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GreedyDecodeTest {

  @TempDir Path directory;

  @Test
  @DisplayName("argmax breaks ties on the lowest token id so the rule is total")
  void argmaxBreaksTiesOnTheLowestTokenId() {
    // A tie-break that depends on iteration order would make G1 fail intermittently for reasons
    // that have nothing to do with the kernels.
    assertThat(GreedyDecode.argmax(new float[] {1.0f, 3.0f, 3.0f, 2.0f})).isEqualTo(1);
    assertThat(GreedyDecode.argmax(new float[] {5.0f})).isEqualTo(0);
    assertThat(GreedyDecode.argmax(new float[] {-2.0f, -9.0f, -2.0f})).isEqualTo(0);
    assertThatThrownBy(() -> GreedyDecode.argmax(new float[0]))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("prompt files carry their own provenance in comments")
  void promptFilesCarryTheirOwnProvenanceInComments() throws Exception {
    Path prompts =
        Files.writeString(
            directory.resolve("prompts.txt"),
            """
            # chosen so the file explains itself
            first prompt

            second prompt
            #  indented comment
            """);

    assertThat(GreedyDecode.readPrompts(prompts)).containsExactly("first prompt", "second prompt");
  }

  @Test
  @DisplayName("an empty prompt file is refused rather than silently measuring nothing")
  void anEmptyPromptFileIsRefused() throws Exception {
    Path prompts = Files.writeString(directory.resolve("empty.txt"), "# only a comment\n");
    assertThatThrownBy(() -> GreedyDecode.readPrompts(prompts))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("the first token comes from prefill and costs no decode step")
  void theFirstTokenComesFromPrefillAndCostsNoDecodeStep() {
    // maxTokens tokens, maxTokens-1 forward passes. Charging the prefill-produced token to the
    // decode clock would inflate decode tokens/s on short runs, which is exactly the regime G4
    // measures.
    List<String> steps = new ArrayList<>();
    ScriptedBackend backend = new ScriptedBackend(List.of(7, 8, 9, 10), 0);

    GreedyDecode.Sequence sequence =
        GreedyDecode.generate(
            backend,
            0,
            "prompt",
            4,
            64,
            (promptIndex, tokenIndex, tokenId, decodeStep) ->
                steps.add(tokenIndex + ":" + tokenId + ":" + decodeStep));

    assertThat(sequence.tokenIds()).containsExactly(7, 8, 9, 10);
    assertThat(sequence.decodeSteps()).isEqualTo(3);
    assertThat(backend.forwardCalls()).isEqualTo(3);
    assertThat(steps).containsExactly("0:7:false", "1:8:true", "2:9:true", "3:10:true");
  }

  @Test
  @DisplayName("end of generation is recorded and deliberately not obeyed")
  void endOfGenerationIsRecordedAndNotObeyed() {
    // Stopping early would shorten the compared sequence, weakening G1 exactly where the arms
    // agree. The report says it happened instead.
    ScriptedBackend backend = new ScriptedBackend(List.of(1, 2, 99, 3), 99);

    GreedyDecode.Sequence sequence =
        GreedyDecode.generate(backend, 0, "prompt", 4, 64, GreedyDecode.StepListener.none());

    assertThat(sequence.tokenIds()).containsExactly(1, 2, 99, 3);
    assertThat(sequence.hitEndOfGeneration()).isTrue();
    assertThat(sequence.endOfGenerationIndex()).isEqualTo(2);
  }

  @Test
  @DisplayName("a prompt that would overrun the context is refused, not truncated")
  void aPromptThatWouldOverrunTheContextIsRefused() {
    // One over-long input once killed two entire datasets silently. A cap that is worked around
    // rather than reported caps the metric for reasons unrelated to what is being measured.
    ScriptedBackend backend = new ScriptedBackend(List.of(1, 2, 3, 4), 0);

    assertThatThrownBy(
            () ->
                GreedyDecode.generate(backend, 3, "prompt", 4, 5, GreedyDecode.StepListener.none()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("prompt 3")
        .hasMessageContaining("--context");
  }

  @Test
  @DisplayName("the warmup sequence is generated but never reported as a measured prompt")
  void theWarmupSequenceIsNotAMeasuredPrompt() {
    List<Integer> seenPrompts = new ArrayList<>();
    ScriptedBackend backend = new ScriptedBackend(List.of(1, 2, 3, 4, 5, 6, 7, 8), 0);

    GreedyDecode.warmUp(backend, "warm", 3, 64);
    GreedyDecode.generateAll(
        backend,
        List.of("a", "b"),
        2,
        64,
        (promptIndex, tokenIndex, tokenId, decodeStep) -> seenPrompts.add(promptIndex));

    assertThat(seenPrompts).containsExactly(0, 0, 1, 1);
  }

  @Test
  @DisplayName("decode throughput counts only the tokens that cost a forward pass")
  void decodeThroughputCountsOnlyTokensThatCostAForwardPass() {
    GreedyDecode.Sequence sequence =
        new GreedyDecode.Sequence(
            0, "digest", 10, List.of(1, 2, 3, 4, 5), 1_000_000_000L, 2_000_000_000L, -1);

    assertThat(sequence.decodeSteps()).isEqualTo(4);
    assertThat(sequence.decodeTokensPerSecond()).isEqualTo(2.0);
    assertThat(sequence.prefillTokensPerSecond()).isEqualTo(10.0);
  }
}
