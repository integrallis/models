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
 * One extracted answer and the character span it came from.
 *
 * @param value canonical answer (a plain decimal, a choice label, or the raw inner LaTeX)
 * @param start inclusive start offset of the answer text within the analysed string
 * @param end exclusive end offset
 */
public record AnswerExtraction(String value, int start, int end) {
  public AnswerExtraction {
    Objects.requireNonNull(value, "value");
    if (start < 0 || end < start) {
      throw new IllegalArgumentException("invalid span " + start + ".." + end);
    }
  }

  AnswerExtraction shift(int offset) {
    return new AnswerExtraction(value, start + offset, end + offset);
  }
}
