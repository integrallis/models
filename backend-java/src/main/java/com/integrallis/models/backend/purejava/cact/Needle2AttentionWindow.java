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

/** Needle's bounded recent window with its tool-document prefix retained as an attention sink. */
final class Needle2AttentionWindow {

  private final int recentWindow;
  private final int toolDocumentEndToken;
  private int toolDocumentPrefixLength;

  Needle2AttentionWindow(int recentWindow, int toolDocumentEndToken) {
    if (recentWindow < 0) {
      throw new IllegalArgumentException("recentWindow must be >= 0");
    }
    if (toolDocumentEndToken < 0) {
      throw new IllegalArgumentException("toolDocumentEndToken must be >= 0");
    }
    this.recentWindow = recentWindow;
    this.toolDocumentEndToken = toolDocumentEndToken;
  }

  void accept(int token, int position) {
    if (position < 0) {
      throw new IllegalArgumentException("position must be >= 0");
    }
    if (toolDocumentPrefixLength == 0 && token == toolDocumentEndToken) {
      toolDocumentPrefixLength = position + 1;
    }
  }

  int[] visiblePositions(int position) {
    if (position < 0) {
      throw new IllegalArgumentException("position must be >= 0");
    }
    int recentStart = recentWindow == 0 ? 0 : Math.max(0, position - recentWindow + 1);
    int sinkCount = Math.min(toolDocumentPrefixLength, recentStart);
    int recentCount = position - recentStart + 1;
    int[] visible = new int[Math.addExact(sinkCount, recentCount)];
    for (int index = 0; index < sinkCount; index++) {
      visible[index] = index;
    }
    for (int index = 0; index < recentCount; index++) {
      visible[sinkCount + index] = recentStart + index;
    }
    return visible;
  }

  void rewind(int checkpoint) {
    if (checkpoint < 0) {
      throw new IllegalArgumentException("checkpoint must be >= 0");
    }
    if (toolDocumentPrefixLength > checkpoint) {
      toolDocumentPrefixLength = 0;
    }
  }
}
