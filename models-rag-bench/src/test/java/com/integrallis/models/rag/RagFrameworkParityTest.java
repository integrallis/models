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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

class RagFrameworkParityTest {

  @Test
  void plainJavaLangChain4jAndSpringAiUseIdenticalRetrievalAndPrompt() throws Exception {
    RagCorpus corpus = RagCorpus.loadDefault();
    RagCase testCase = corpus.cases().getFirst();
    List<RagRun> runs = new ArrayList<>();
    List<RecordingGenerationClient> clients = new ArrayList<>();

    for (String framework : List.of("plain-java", "langchain4j", "spring-ai")) {
      RecordingGenerationClient client = new RecordingGenerationClient();
      clients.add(client);
      try (LuceneRagRetriever retriever = new LuceneRagRetriever(corpus.documents());
          RagApplication application = application(framework, retriever, client)) {
        runs.add(application.run(testCase, 32));
      }
    }

    assertThat(runs)
        .extracting(RagRun::framework)
        .containsExactly("plain-java", "langchain4j", "spring-ai");
    assertThat(runs).extracting(RagRun::promptSha256).containsOnly(runs.getFirst().promptSha256());
    assertThat(runs.getFirst().promptSha256())
        .isEqualTo("26fccba3e100ea107b382b25a9404ba8fb68bf9bd8c2003878ad6b6cefd87841");
    assertThat(runs)
        .allSatisfy(
            run ->
                assertThat(run.retrieved())
                    .extracting(hit -> hit.document().id())
                    .containsExactlyElementsOf(
                        runs.getFirst().retrieved().stream()
                            .map(hit -> hit.document().id())
                            .toList()));
    assertThat(clients)
        .extracting(RecordingGenerationClient::lastPrompt)
        .containsOnly(clients.getFirst().lastPrompt());
    assertThat(runs).allSatisfy(run -> assertThat(run.evaluation().correct()).isTrue());
  }

  @Test
  void frameworksUseTheSameChatmlEnvelope() throws Exception {
    RagCorpus corpus = RagCorpus.loadDefault();
    RagCase testCase = corpus.cases().getFirst();
    List<RagRun> runs = new ArrayList<>();
    List<RecordingGenerationClient> clients = new ArrayList<>();

    for (String framework : List.of("plain-java", "langchain4j", "spring-ai")) {
      RecordingGenerationClient client = new RecordingGenerationClient();
      clients.add(client);
      try (LuceneRagRetriever retriever = new LuceneRagRetriever(corpus.documents());
          RagApplication application =
              application(framework, retriever, client, RagPromptTemplate.CHATML)) {
        runs.add(application.run(testCase, 32));
      }
    }

    assertThat(runs)
        .extracting(RagRun::promptSha256)
        .containsOnly("adeb518c096c5bbe124b3c8d60ab2001e6754d192938465f6bedea5ca5a62bad");
    assertThat(clients)
        .extracting(RecordingGenerationClient::lastPrompt)
        .allSatisfy(
            prompt ->
                assertThat(prompt)
                    .startsWith("<|im_start|>user\n")
                    .endsWith("<|im_end|>\n<|im_start|>assistant\n"));
  }

  @Test
  void springAiApplicationClosesItsAdvisorExecutor() throws Exception {
    RagCorpus corpus = RagCorpus.loadDefault();
    ExecutorService executor = Executors.newSingleThreadExecutor();

    try (LuceneRagRetriever retriever = new LuceneRagRetriever(corpus.documents())) {
      SpringAiRagApplication application =
          new SpringAiRagApplication(
              retriever, new RecordingGenerationClient(), 1, RagPromptTemplate.RAW, executor);

      application.close();
    }

    assertThat(executor.isShutdown()).isTrue();
  }

  private static RagApplication application(
      String framework, RagRetriever retriever, GenerationClient client) {
    return application(framework, retriever, client, RagPromptTemplate.RAW);
  }

  private static RagApplication application(
      String framework,
      RagRetriever retriever,
      GenerationClient client,
      RagPromptTemplate promptTemplate) {
    return switch (framework) {
      case "plain-java" -> new PlainJavaRagApplication(retriever, client, 1, promptTemplate);
      case "langchain4j" -> new LangChain4jRagApplication(retriever, client, 1, promptTemplate);
      case "spring-ai" -> new SpringAiRagApplication(retriever, client, 1, promptTemplate);
      default -> throw new IllegalArgumentException(framework);
    };
  }

  private static final class RecordingGenerationClient implements GenerationClient {
    private String lastPrompt;

    @Override
    public String backend() {
      return "recording";
    }

    @Override
    public String model() {
      return "recording-model";
    }

    @Override
    public GenerationResult generate(String prompt, int maxTokens) {
      lastPrompt = prompt;
      return new GenerationResult(
          "The report deadline is 30 calendar days and the deductible is 75 dollars "
              + "[claims-auto-glass].",
          100,
          18,
          5,
          25,
          1_000,
          0);
    }

    @Override
    public void close() {}

    String lastPrompt() {
      return lastPrompt;
    }
  }
}
