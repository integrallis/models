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
package com.integrallis.models.rag;

import java.util.List;
import java.util.Objects;

/** Canonical prompt renderer used by every framework and backend. */
public final class RagPromptRenderer {
  private static final String INSTRUCTIONS =
      "You answer questions using only the supplied context.\n"
          + "Rules:\n"
          + "- If the context does not contain the answer, reply exactly INSUFFICIENT_CONTEXT.\n"
          + "- Otherwise answer in one short sentence and cite every supporting source as "
          + "[source-id].\n"
          + "- Do not use prior knowledge.\n\n";

  private RagPromptRenderer() {}

  public static String render(String question, List<RetrievedDocument> retrieved) {
    Objects.requireNonNull(question, "question");
    Objects.requireNonNull(retrieved, "retrieved");
    if (question.isBlank()) {
      throw new IllegalArgumentException("question must not be blank");
    }

    StringBuilder prompt = new StringBuilder(INSTRUCTIONS).append("CONTEXT\n");
    for (RetrievedDocument hit : retrieved) {
      RagDocument document = hit.document();
      prompt
          .append('[')
          .append(document.id())
          .append("] ")
          .append(document.title())
          .append('\n')
          .append(document.text())
          .append("\n\n");
    }
    return prompt.append("QUESTION\n").append(question).append("\n\nANSWER\n").toString();
  }
}
