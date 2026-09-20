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
package com.integrallis.models.decisions;

import java.util.List;

/**
 * A finite set of outcomes a decision is allowed to return.
 *
 * <p>The space is declared before the decision is taken, so a verdict can only ever name one of
 * these labels. That is what makes the result safe to hand straight to code: there is no parsing
 * step to fail and no value outside the declaration for a model to invent.
 */
public sealed interface AnswerSpace permits Noul, Choice, Score {

  /** Returns the question this space answers. */
  String question();

  /** Returns the outcomes, in declaration order. The list is unmodifiable. */
  List<String> labels();

  /** Returns how many outcomes the space holds. */
  default int size() {
    return labels().size();
  }
}
