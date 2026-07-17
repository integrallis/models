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

/** Dependency-light Java RAG application using the shared retriever and Models generation. */
public final class PlainJavaRagApplication implements RagApplication {
  private final RagRetriever retriever;
  private final GenerationClient client;
  private final int topK;
  private final RagPromptTemplate promptTemplate;

  public PlainJavaRagApplication(RagRetriever retriever, GenerationClient client, int topK) {
    this(retriever, client, topK, RagPromptTemplate.RAW);
  }

  public PlainJavaRagApplication(
      RagRetriever retriever, GenerationClient client, int topK, RagPromptTemplate promptTemplate) {
    this.retriever = Objects.requireNonNull(retriever, "retriever");
    this.client = Objects.requireNonNull(client, "client");
    this.promptTemplate = Objects.requireNonNull(promptTemplate, "promptTemplate");
    if (topK < 1) {
      throw new IllegalArgumentException("topK must be positive");
    }
    this.topK = topK;
  }

  @Override
  public RagRun run(RagCase testCase, int maxTokens) throws Exception {
    long totalStart = System.nanoTime();
    long retrievalStart = System.nanoTime();
    List<RetrievedDocument> hits = retriever.retrieve(testCase.question(), topK);
    double retrievalMillis = elapsedMillis(retrievalStart);
    String prompt = RagPromptRenderer.render(testCase.question(), hits, promptTemplate);
    GenerationResult generated = client.generate(prompt, maxTokens);
    double endToEndMillis = elapsedMillis(totalStart);
    return RagRunFactory.create(
        "plain-java", client, testCase, hits, prompt, retrievalMillis, endToEndMillis, generated);
  }

  private static double elapsedMillis(long start) {
    return (System.nanoTime() - start) / 1_000_000.0;
  }
}
