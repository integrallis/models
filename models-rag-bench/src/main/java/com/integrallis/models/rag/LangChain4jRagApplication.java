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

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.injector.ContentInjector;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.service.AiServices;
import java.util.Objects;

/** LangChain4j advanced-RAG application over the controlled workload. */
public final class LangChain4jRagApplication implements RagApplication {
  private final GenerationClient client;
  private final GenerationSession generation;
  private final RetrievalTrace retrieval;
  private final Assistant assistant;

  public LangChain4jRagApplication(RagRetriever retriever, GenerationClient client, int topK) {
    this.client = Objects.requireNonNull(client, "client");
    this.generation = new GenerationSession(client);
    this.retrieval = new RetrievalTrace(Objects.requireNonNull(retriever, "retriever"), topK);

    ContentRetriever contentRetriever =
        query ->
            retrieval.retrieve(query.text()).stream()
                .map(LangChain4jRagApplication::toContent)
                .toList();
    ContentInjector contentInjector =
        (contents, chatMessage) ->
            UserMessage.from(RagPromptRenderer.render(question(chatMessage), retrieval.hits()));
    RetrievalAugmentor augmentor =
        DefaultRetrievalAugmentor.builder()
            .contentRetriever(contentRetriever)
            .contentInjector(contentInjector)
            .build();
    this.assistant =
        AiServices.builder(Assistant.class)
            .chatModel(new MeasuredChatModel(generation, client.model()))
            .retrievalAugmentor(augmentor)
            .build();
  }

  @Override
  public RagRun run(RagCase testCase, int maxTokens) {
    retrieval.clear();
    generation.begin(maxTokens);
    long start = System.nanoTime();
    String answer = assistant.chat(testCase.question());
    double endToEndMillis = (System.nanoTime() - start) / 1_000_000.0;
    GenerationResult result = generation.result();
    if (!answer.equals(result.text())) {
      throw new IllegalStateException("LangChain4j changed the generated answer");
    }
    return RagRunFactory.create(
        "langchain4j",
        client,
        testCase,
        retrieval.hits(),
        generation.prompt(),
        retrieval.elapsedMillis(),
        endToEndMillis,
        result);
  }

  private static Content toContent(RetrievedDocument hit) {
    RagDocument document = hit.document();
    Metadata metadata =
        new Metadata()
            .put("id", document.id())
            .put("title", document.title())
            .put("score", hit.score())
            .put("rank", hit.rank());
    return Content.from(TextSegment.from(document.text(), metadata));
  }

  private static String question(ChatMessage message) {
    if (message instanceof UserMessage userMessage && userMessage.hasSingleText()) {
      return userMessage.singleText();
    }
    throw new IllegalArgumentException("RAG requires one textual user message");
  }

  private interface Assistant {
    String chat(String question);
  }

  private static final class MeasuredChatModel implements ChatModel {
    private final GenerationSession generation;
    private final String model;

    private MeasuredChatModel(GenerationSession generation, String model) {
      this.generation = generation;
      this.model = model;
    }

    @Override
    public ChatResponse doChat(ChatRequest request) {
      String prompt =
          request.messages().stream()
              .filter(UserMessage.class::isInstance)
              .map(UserMessage.class::cast)
              .filter(UserMessage::hasSingleText)
              .map(UserMessage::singleText)
              .reduce((first, second) -> second)
              .orElseThrow(() -> new IllegalArgumentException("missing textual user message"));
      GenerationResult result = generation.generate(prompt);
      return ChatResponse.builder()
          .aiMessage(AiMessage.from(result.text()))
          .modelName(model)
          .build();
    }
  }
}
