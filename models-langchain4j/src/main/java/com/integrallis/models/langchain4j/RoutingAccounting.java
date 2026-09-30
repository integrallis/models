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

import com.integrallis.models.router.RoutingExecutionOptions;
import com.integrallis.models.router.RoutingUsage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;

final class RoutingAccounting {
  private RoutingAccounting() {}

  static void validate(ChatRequest request, RoutingExecutionOptions options) {
    if (options.budgeted()
        && (request.maxOutputTokens() == null
            || request.maxOutputTokens() < 0
            || request.maxOutputTokens() > options.tokenBounds().outputTokens())) {
      throw new IllegalArgumentException(
          "budgeted requests require an explicit provider maxOutputTokens within the declared output bound");
    }
  }

  static RoutingUsage usage(ChatResponse response) {
    if (response == null
        || response.tokenUsage() == null
        || response.tokenUsage().inputTokenCount() == null
        || response.tokenUsage().outputTokenCount() == null) return null;
    return new RoutingUsage(
        response.tokenUsage().inputTokenCount(), response.tokenUsage().outputTokenCount());
  }
}
