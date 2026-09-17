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
package com.integrallis.models.runtime;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.RepetitionLoopDetection;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

@Tag("unit")
class RepetitionLoopDetectorTest {

  /** Returns the 0-based index of the token at which the detector fired, or -1. */
  private static int firstTrigger(RepetitionLoopDetection config, int[] tokens) {
    RepetitionLoopDetector detector = new RepetitionLoopDetector(config);
    for (int index = 0; index < tokens.length; index++) {
      if (detector.accept(tokens[index])) {
        return index;
      }
    }
    return -1;
  }

  private static int[] repeat(int[] span, int times, int[] prefix) {
    int[] tokens = new int[prefix.length + span.length * times];
    System.arraycopy(prefix, 0, tokens, 0, prefix.length);
    for (int copy = 0; copy < times; copy++) {
      System.arraycopy(span, 0, tokens, prefix.length + copy * span.length, span.length);
    }
    return tokens;
  }

  private static int[] distinctSpan(int length, int firstToken) {
    int[] span = new int[length];
    for (int index = 0; index < length; index++) {
      span[index] = firstToken + index;
    }
    return span;
  }

  @ParameterizedTest
  @ValueSource(ints = {1, 2, 3, 7, 16, 32})
  void firesExactlyWhenTheMinimumRepeatCountIsReachedForEverySpanUpToTheWindow(int span) {
    RepetitionLoopDetection config = new RepetitionLoopDetection(32, 4, 0);
    int[] prefix = {900, 901, 902, 903, 904};
    int[] tokens = repeat(distinctSpan(span, 100), 10, prefix);

    int trigger = firstTrigger(config, tokens);

    // Four complete copies end at prefix + 4 * span - 1.
    assertThat(trigger).isEqualTo(prefix.length + 4 * span - 1);
  }

  @Test
  void reportsTheShortestLoopingSpan() {
    RepetitionLoopDetector detector =
        new RepetitionLoopDetector(new RepetitionLoopDetection(16, 3, 0));
    int[] tokens = repeat(new int[] {5, 6, 7}, 6, new int[] {1});
    boolean fired = false;
    for (int token : tokens) {
      if (detector.accept(token)) {
        fired = true;
        break;
      }
    }

    assertThat(fired).isTrue();
    assertThat(detector.detectedSpan()).isEqualTo(3);
  }

  @Test
  void spanLongerThanTheWindowIsNotALoop() {
    RepetitionLoopDetection config = new RepetitionLoopDetection(16, 3, 0);

    assertThat(firstTrigger(config, repeat(distinctSpan(17, 100), 20, new int[0]))).isEqualTo(-1);
  }

  @Test
  void nearLoopsWithAPerturbationEveryPeriodDoNotFire() {
    RepetitionLoopDetection config = new RepetitionLoopDetection(16, 3, 0);
    List<Integer> tokens = new ArrayList<>();
    // "A B C" repeated, but every copy's last token cycles, so no two adjacent copies match.
    for (int copy = 0; copy < 200; copy++) {
      tokens.add(1);
      tokens.add(2);
      tokens.add(10 + copy % 7);
    }
    int[] array = tokens.stream().mapToInt(Integer::intValue).toArray();

    // Period 21 (7 * 3) exists but is beyond the 16-token window.
    assertThat(firstTrigger(config, array)).isEqualTo(-1);
  }

  @Test
  void twoCopiesAreNotALoopWhenThreeAreRequired() {
    RepetitionLoopDetection config = new RepetitionLoopDetection(16, 3, 0);
    int[] tokens = {1, 2, 3, 4, 1, 2, 3, 4, 9, 1, 2, 3, 4, 1, 2, 3, 4, 8};

    assertThat(firstTrigger(config, tokens)).isEqualTo(-1);
  }

  @Test
  void numberedListWithRepeatedItemTextDoesNotFire() {
    // "1. Item\n2. Item\n..." — every line shares its text but the ordinal changes, so the
    // sequence is never exactly periodic.
    RepetitionLoopDetection config = new RepetitionLoopDetection(32, 3, 0);
    List<Integer> tokens = new ArrayList<>();
    for (int item = 1; item <= 500; item++) {
      tokens.add(1_000 + item); // ordinal
      tokens.add(7); // "."
      tokens.add(8); // " Item"
      tokens.add(9); // "\n"
    }

    assertThat(firstTrigger(config, tokens.stream().mapToInt(Integer::intValue).toArray()))
        .isEqualTo(-1);
  }

  @Test
  void minimumLoopTokensHoldsBackShortSpans() {
    // A run of one token repeated 7 times satisfies minRepeats=3 but not minLoopTokens=8.
    RepetitionLoopDetection config = new RepetitionLoopDetection(16, 3, 8);

    assertThat(firstTrigger(config, new int[] {1, 5, 5, 5, 5, 5, 5, 5, 2})).isEqualTo(-1);
    assertThat(firstTrigger(config, new int[] {1, 5, 5, 5, 5, 5, 5, 5, 5})).isEqualTo(8);
    // For long spans minRepeats dominates: 3 copies of a 4-token span is 12 >= 8 tokens.
    assertThat(firstTrigger(config, repeat(new int[] {1, 2, 3, 4}, 3, new int[] {9})))
        .isEqualTo(12);
  }

  @Test
  void legitimateLongRepetitionNeedsAHighEnoughThresholdToPass() {
    // A zero-initialized array literal "0, 0, 0, ..." is exact period-2 repetition. It is the
    // documented limitation of an exact-period detector: it fires unless the thresholds allow it.
    int[] zeros = repeat(new int[] {0, 11}, 40, new int[] {3});

    assertThat(firstTrigger(new RepetitionLoopDetection(16, 4, 0), zeros)).isEqualTo(8);
    assertThat(firstTrigger(new RepetitionLoopDetection(16, 64, 0), zeros)).isEqualTo(-1);
  }

  @Test
  void disabledDetectionNeverFiresAndDoesNoWork() {
    RepetitionLoopDetector detector =
        new RepetitionLoopDetector(RepetitionLoopDetection.disabled());
    for (int index = 0; index < 10_000; index++) {
      assertThat(detector.accept(5)).isFalse();
    }

    assertThat(detector.comparisons()).isZero();
  }

  @Test
  void workPerTokenIsBoundedByTheWindowIndependentOfGenerationLength() {
    int window = 24;
    RepetitionLoopDetector detector =
        new RepetitionLoopDetector(new RepetitionLoopDetection(window, 1_000_000, 0));
    Random random = new Random(20260916L);
    int tokens = 200_000;
    for (int index = 0; index < tokens; index++) {
      detector.accept(random.nextInt(50));
    }

    // Exactly min(window, tokens seen so far) comparisons per token: O(window), never O(history).
    long expected = (long) window * (tokens - window) + (long) window * (window - 1) / 2;
    assertThat(detector.comparisons()).isEqualTo(expected);
  }
}
