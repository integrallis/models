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
package com.integrallis.models.bench.fusion;

import java.util.Objects;

/**
 * A generated response split into its thinking block and its final, user-facing answer.
 *
 * @param reasoning text inside the think block, or {@code null} when no think block was opened
 * @param reasoningStart offset of {@code reasoning} within the full text (0 when absent)
 * @param finalText text after the closing tag; the whole text when no block; empty if truncated
 * @param finalStart offset of {@code finalText} within the full text
 * @param thinkPresent whether an opening think tag was emitted
 * @param thinkTruncated whether the think block was opened but never closed
 */
public record ReasoningTrace(
    String reasoning,
    int reasoningStart,
    String finalText,
    int finalStart,
    boolean thinkPresent,
    boolean thinkTruncated) {

  static final String OPEN = "<think>";
  static final String CLOSE = "</think>";

  /** Splits a response on the first opening and the first following closing think tag. */
  public static ReasoningTrace split(String text) {
    Objects.requireNonNull(text, "text");
    int open = text.indexOf(OPEN);
    if (open < 0) {
      return new ReasoningTrace(null, 0, text, 0, false, false);
    }
    int reasoningStart = open + OPEN.length();
    int close = text.indexOf(CLOSE, reasoningStart);
    if (close < 0) {
      return new ReasoningTrace(
          text.substring(reasoningStart), reasoningStart, "", text.length(), true, true);
    }
    int finalStart = close + CLOSE.length();
    return new ReasoningTrace(
        text.substring(reasoningStart, close),
        reasoningStart,
        text.substring(finalStart),
        finalStart,
        true,
        false);
  }
}
