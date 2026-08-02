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

/** Canonical prompt renderer used by every framework and backend. */
public final class RagPromptRenderer {
  private RagPromptRenderer() {}

  public static String render(String question, List<RetrievedDocument> retrieved) {
    return render(question, retrieved, RagPromptTemplate.RAW);
  }

  public static String render(
      String question, List<RetrievedDocument> retrieved, RagPromptTemplate template) {
    java.util.Objects.requireNonNull(retrieved, "retrieved");
    java.util.Objects.requireNonNull(template, "template");
    List<GroundingDocument> evidence =
        retrieved.stream()
            .map(
                hit ->
                    new GroundingDocument(
                        hit.document().id(),
                        hit.document().title(),
                        hit.document().text(),
                        hit.score(),
                        hit.rank()))
            .toList();
    GroundedRagPrompt prompt = GroundedRagPrompt.formatUnchecked(question, evidence);
    return template.apply(prompt.instructions(), prompt.request());
  }
}
