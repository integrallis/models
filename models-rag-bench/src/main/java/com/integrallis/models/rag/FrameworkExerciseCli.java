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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.langchain4j.ModelsChatModel;
import com.integrallis.models.runtime.RuntimeTextGenerationModel;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.spring.ai.ModelsSpringAiChatModel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Exercises one model the way an application would: ordinary prompts, through all three surfaces.
 *
 * <p>Distinct from the RAG harness on purpose. The qualification gate measures grounded answering
 * over a pinned corpus, which is the right question for a retrieval application and says nothing
 * about whether a model is coherent when someone simply asks it something. This sends the same
 * plain prompts through the plain Java runtime, the LangChain4j adapter and the Spring AI adapter,
 * records what each returns, and reports whether the three agree.
 *
 * <p>Greedy sampling, so a disagreement between surfaces is a real difference rather than sampling
 * noise: all three receive the same backend, the same chat template and the same options, and
 * differ only in the adapter, so identical output is the expected result and anything else is a
 * defect in an adapter.
 *
 * <p>It reports; it does not gate. A model that reads poorly here keeps the qualification it earned
 * against the gate it was measured on -- the finding is recorded and investigated, not used to
 * retract evidence.
 */
public final class FrameworkExerciseCli {
  private static final ObjectMapper JSON = new ObjectMapper();

  /** Ordinary requests, spanning the shapes an application actually sends. */
  private static final List<Map<String, String>> PROMPTS =
      List.of(
          Map.of(
              "id",
              "factual",
              "text",
              "What is the capital of France? Answer with only the city name."),
          Map.of(
              "id",
              "summarize",
              "text",
              "Summarise this in one sentence: The service rejected writes for 38 minutes after a"
                  + " schema migration added a non-null column with no default, and recovery came"
                  + " from rolling the migration back."),
          Map.of(
              "id",
              "code",
              "text",
              "Write a Java method that returns true when a string is a palindrome."),
          Map.of("id", "explain", "text", "Explain what a hash map is, in two sentences."),
          Map.of(
              "id",
              "instruction",
              "text",
              "List exactly three colours, one per line, and nothing else."));

  private FrameworkExerciseCli() {}

  public static void main(String[] args) throws Exception {
    Path model = null;
    Path output = null;
    String modelId = null;
    String chatTemplate = "chatml";
    int maxTokens = 96;
    int context = 2048;
    String backendId = "rust-ffm";
    for (int index = 0; index < args.length; index++) {
      switch (args[index]) {
        case "--model" -> model = Path.of(args[++index]);
        case "--model-id" -> modelId = args[++index];
        case "--chat-template" -> chatTemplate = args[++index];
        case "--max-tokens" -> maxTokens = Integer.parseInt(args[++index]);
        case "--context" -> context = Integer.parseInt(args[++index]);
        case "--backend" -> backendId = args[++index];
        case "--output" -> output = Path.of(args[++index]);
        default -> throw new IllegalArgumentException("unknown argument: " + args[index]);
      }
    }
    if (model == null || modelId == null) {
      System.err.println(
          "usage: framework-exercise --model <artifact.gguf> --model-id <id>"
              + " [--chat-template <id>] [--backend rust-ffm|pure-java] [--max-tokens n]"
              + " [--output <out.json>]");
      System.exit(2);
      return;
    }

    ChatTemplate template = ChatTemplate.parse(chatTemplate);
    SamplingOptions options =
        SamplingOptions.builder().temperature(0.0f).topP(1.0f).maxTokens(maxTokens).build();

    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(context));
    System.setProperty(RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY, "true");

    Map<String, Object> report = new LinkedHashMap<>();
    report.put("schemaVersion", 1);
    report.put("modelId", modelId);
    report.put("chatTemplate", template.id());
    report.put("maxOutputTokens", maxTokens);
    report.put("sampling", "greedy");
    report.put("backend", backendId);
    List<Map<String, Object>> results = new ArrayList<>();
    int agreeing = 0;

    // Named, never inferred: a report has to say which backend produced it, and a silent fallback
    // would
    // let evidence claim the native path while the pure-Java one ran.
    try (InferenceBackend backend = openBackend(backendId, model)) {
      RuntimeTextGenerationModel plain = new RuntimeTextGenerationModel(backend);
      ModelsChatModel langchain = new ModelsChatModel(backend, template, options);
      ModelsSpringAiChatModel spring = new ModelsSpringAiChatModel(backend, template, options);
      for (Map<String, String> prompt : PROMPTS) {
        String text = prompt.get("text");
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("prompt", prompt.get("id"));
        row.put("request", text);
        // Each surface is reset between prompts so one answer cannot condition the next; a
        // difference
        // between adapters must come from the adapter, not from carried state.
        backend.reset();
        String plainOut =
            trimmed(plain.generate(template.render(List.of(ChatMessage.user(text))), options));
        backend.reset();
        String langchainOut = trimmed(langchain.chat(text));
        backend.reset();
        String springOut = trimmed(spring.call(text));
        row.put("plainJava", plainOut);
        row.put("langchain4j", langchainOut);
        row.put("springAi", springOut);
        boolean identical = plainOut.equals(langchainOut) && plainOut.equals(springOut);
        row.put("identicalAcrossSurfaces", identical);
        row.put("empty", plainOut.isBlank());
        if (identical) {
          agreeing++;
        }
        results.add(row);
      }
    }
    report.put("prompts", results);
    report.put("promptsAgreeingAcrossSurfaces", agreeing);
    report.put("promptCount", PROMPTS.size());
    report.put("allSurfacesAgree", agreeing == PROMPTS.size());
    report.put("anyEmptyAnswer", results.stream().anyMatch(row -> (Boolean) row.get("empty")));

    String rendered = JSON.writerWithDefaultPrettyPrinter().writeValueAsString(report);
    if (output != null) {
      java.nio.file.Files.writeString(output, rendered + System.lineSeparator());
    }
    System.out.println(rendered);
  }

  private static InferenceBackend openBackend(String backendId, Path model) {
    return switch (backendId) {
      case "rust-ffm" -> RustFfmBackend.load(model);
      case "pure-java" -> PureJavaBackend.load(model);
      default ->
          throw new IllegalArgumentException("backend must be rust-ffm or pure-java: " + backendId);
    };
  }

  private static String trimmed(String value) {
    return value == null ? "" : value.strip();
  }
}
