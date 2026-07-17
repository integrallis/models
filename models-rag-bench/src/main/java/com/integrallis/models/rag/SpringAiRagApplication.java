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
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.rag.generation.augmentation.QueryAugmenter;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

/** Spring AI modular-RAG application over the controlled workload. */
public final class SpringAiRagApplication implements RagApplication {
  private final GenerationClient client;
  private final GenerationSession generation;
  private final RetrievalTrace retrieval;
  private final ChatClient chatClient;
  private final ExecutorService advisorExecutor;

  public SpringAiRagApplication(RagRetriever retriever, GenerationClient client, int topK) {
    this(retriever, client, topK, RagPromptTemplate.RAW);
  }

  public SpringAiRagApplication(
      RagRetriever retriever, GenerationClient client, int topK, RagPromptTemplate promptTemplate) {
    this(retriever, client, topK, promptTemplate, Executors.newVirtualThreadPerTaskExecutor());
  }

  SpringAiRagApplication(
      RagRetriever retriever,
      GenerationClient client,
      int topK,
      RagPromptTemplate promptTemplate,
      ExecutorService advisorExecutor) {
    this.client = Objects.requireNonNull(client, "client");
    this.generation = new GenerationSession(client);
    this.retrieval = new RetrievalTrace(Objects.requireNonNull(retriever, "retriever"), topK);
    this.advisorExecutor = Objects.requireNonNull(advisorExecutor, "advisorExecutor");

    DocumentRetriever documentRetriever =
        query ->
            retrieval.retrieve(query.text()).stream()
                .map(SpringAiRagApplication::toDocument)
                .toList();
    QueryAugmenter queryAugmenter =
        (query, documents) ->
            new Query(
                RagPromptRenderer.render(query.text(), retrieval.hits(), promptTemplate),
                query.history(),
                query.context());
    RetrievalAugmentationAdvisor advisor =
        RetrievalAugmentationAdvisor.builder()
            .documentRetriever(documentRetriever)
            .queryAugmenter(queryAugmenter)
            .taskExecutor(advisorExecutor::execute)
            .build();
    this.chatClient =
        ChatClient.builder(new MeasuredChatModel(generation)).defaultAdvisors(advisor).build();
  }

  @Override
  public RagRun run(RagCase testCase, int maxTokens) {
    retrieval.clear();
    generation.begin(maxTokens);
    long start = System.nanoTime();
    String answer = chatClient.prompt().user(testCase.question()).call().content();
    double endToEndMillis = (System.nanoTime() - start) / 1_000_000.0;
    GenerationResult result = generation.result();
    if (!answer.equals(result.text())) {
      throw new IllegalStateException("Spring AI changed the generated answer");
    }
    return RagRunFactory.create(
        "spring-ai",
        client,
        testCase,
        retrieval.hits(),
        generation.prompt(),
        retrieval.elapsedMillis(),
        endToEndMillis,
        result);
  }

  @Override
  public void close() {
    advisorExecutor.close();
  }

  private static Document toDocument(RetrievedDocument hit) {
    RagDocument document = hit.document();
    return new Document(
        document.id(),
        document.text(),
        Map.of("title", document.title(), "rank", hit.rank(), "score", hit.score()));
  }

  private static final class MeasuredChatModel implements ChatModel {
    private final GenerationSession generation;

    private MeasuredChatModel(GenerationSession generation) {
      this.generation = generation;
    }

    @Override
    public ChatResponse call(Prompt prompt) {
      GenerationResult result = generation.generate(prompt.getContents());
      return new ChatResponse(List.of(new Generation(new AssistantMessage(result.text()))));
    }
  }
}
