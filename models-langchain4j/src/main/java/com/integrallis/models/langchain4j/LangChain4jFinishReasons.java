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
package com.integrallis.models.langchain4j;

import com.integrallis.models.api.StopReason;
import dev.langchain4j.model.output.FinishReason;

/** Maps the runtime's typed stop reason onto LangChain4j's coarser finish reasons. */
final class LangChain4jFinishReasons {

  private LangChain4jFinishReasons() {}

  /**
   * Returns the LangChain4j finish reason for a runtime stop reason, or {@code null} when the
   * engine did not report one.
   *
   * <p>Every model-chosen end is {@link FinishReason#STOP}; only the token limit is {@link
   * FinishReason#LENGTH}. A detected repetition loop and a consumer cancellation have no
   * LangChain4j equivalent and map to {@link FinishReason#OTHER}, so neither is mistaken for a
   * natural end.
   */
  static FinishReason of(StopReason stopReason) {
    if (stopReason == null) {
      return null;
    }
    return switch (stopReason) {
      case EOS, STOP_SEQUENCE, CONSTRAINT_COMPLETE -> FinishReason.STOP;
      case MAX_TOKENS -> FinishReason.LENGTH;
      case REPETITION_LOOP, CANCELLED -> FinishReason.OTHER;
    };
  }
}
