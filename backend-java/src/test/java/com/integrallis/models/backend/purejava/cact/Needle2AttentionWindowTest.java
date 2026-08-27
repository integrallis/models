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
package com.integrallis.models.backend.purejava.cact;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class Needle2AttentionWindowTest {

  @Test
  void keepsTheToolDocumentPrefixVisibleOutsideTheRecentWindow() {
    Needle2AttentionWindow window = new Needle2AttentionWindow(3, 9);

    for (int position = 0; position <= 7; position++) {
      window.accept(position == 2 ? 9 : 100 + position, position);
    }

    assertThat(window.visiblePositions(7)).containsExactly(0, 1, 2, 5, 6, 7);
  }

  @Test
  void avoidsRepeatingSinkPositionsThatAreStillRecent() {
    Needle2AttentionWindow window = new Needle2AttentionWindow(4, 9);

    window.accept(100, 0);
    window.accept(9, 1);
    window.accept(101, 2);

    assertThat(window.visiblePositions(2)).containsExactly(0, 1, 2);
  }

  @Test
  void anUnboundedContextNeedsNoSinkSpecialCase() {
    Needle2AttentionWindow window = new Needle2AttentionWindow(0, 9);

    window.accept(9, 1);

    assertThat(window.visiblePositions(4)).containsExactly(0, 1, 2, 3, 4);
  }

  @Test
  void rewindingBeforeTheToolDocumentEndForgetsTheOldSink() {
    Needle2AttentionWindow window = new Needle2AttentionWindow(2, 9);
    window.accept(100, 0);
    window.accept(9, 1);

    window.rewind(1);

    assertThat(window.visiblePositions(4)).containsExactly(3, 4);
  }
}
