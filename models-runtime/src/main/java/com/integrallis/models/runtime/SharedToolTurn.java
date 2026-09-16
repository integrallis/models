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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;

/** One request-scoped tool-specialist branch and exact base-response branch. */
public interface SharedToolTurn extends AutoCloseable {

  String generateToolCall(SamplingOptions options, TokenConstraint constraint);

  void generateToolCall(SamplingOptions options, TokenStream stream, TokenConstraint constraint);

  /**
   * Extends this turn's exact base branch into the next activated selection boundary.
   *
   * <p>The current turn is consumed. Previously evaluated KV blocks remain physically shared and
   * only the new context suffix is evaluated before the returned base/adapter fork is opened. This
   * can follow either a tool selection that needs another tool or a completed conversational base
   * response followed by a later user turn.
   */
  default SharedToolTurn continueToolSelection(ModelPrompt renderedToolPrompt) {
    throw new UnsupportedOperationException("this tool turn cannot extend its shared KV prefix");
  }

  String generateBaseResponse(ModelPrompt prompt, SamplingOptions options);

  void generateBaseResponse(ModelPrompt prompt, SamplingOptions options, TokenStream stream);

  int sharedPrefixTokens();

  long sharedPrefixBytes();

  boolean physicallySharesPrefix();

  GenerationMetrics toolMetrics();

  GenerationMetrics responseMetrics();

  @Override
  void close();
}
