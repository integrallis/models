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
package com.integrallis.models.bench;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ModelPrompt;
import java.io.IOException;
import java.util.List;

/**
 * Renders the Granite 4.x documents prompt exactly as the published chat template renders a
 * documents-only request.
 *
 * <p>The two {@code <documents>} markers are special tokens in the Granite 4.1 tokenizer and are
 * emitted as control segments. Every instruction sentence, document body, and conversation turn is
 * a text segment, so caller data can never be interpreted as a control token even when it spells
 * one. Documents render through the same JSON shape Transformers' {@code tojson} filter produces
 * for a {@code {doc_id, text}} mapping.
 */
final class GraniteDocumentsPrompt {
  static final String ASSISTANT_MARKER = "<|start_of_role|>assistant<|end_of_role|>";
  static final String DOCUMENTS_OPEN = "<documents>";
  static final String DOCUMENTS_CLOSE = "</documents>";
  static final String INTRO =
      "You are a helpful assistant with access to the following documents. You may use one or "
          + "more documents to assist with the user query.\n\n"
          + "You are given a list of documents within ";
  static final String TAGS_SUFFIX = " XML tags:\n";
  static final String OUTRO =
      "\n\nWrite the response to the user's input by strictly aligning with the facts in the "
          + "provided documents. If the information needed to answer the question is not "
          + "available in the documents, inform the user that the question cannot be answered "
          + "based on the available data.";
  private static final ObjectMapper JSON = new ObjectMapper();

  private GraniteDocumentsPrompt() {}

  /** Renders one document exactly as Transformers' {@code tojson} filter renders the mapping. */
  static String documentJson(int docId, String text) {
    try {
      return "{\"doc_id\": " + docId + ", \"text\": " + JSON.writeValueAsString(text) + "}";
    } catch (IOException failure) {
      throw new IllegalStateException(failure);
    }
  }

  /** The block rendered between the two markers; document ids count from one in list order. */
  static String documentsBlock(List<String> documentTexts) {
    StringBuilder block = new StringBuilder();
    for (int index = 0; index < documentTexts.size(); index++) {
      block.append('\n').append(documentJson(index + 1, documentTexts.get(index)));
    }
    return block.append('\n').toString();
  }

  /**
   * Appends the documents system message. A leading instruction, when given, precedes the template
   * text separated by a blank line, exactly as the template joins a caller system message.
   */
  static ModelPrompt.Builder appendSystem(
      ModelPrompt.Builder prompt, List<String> documentTexts, String leadingInstruction) {
    prompt.control("<|start_of_role|>system<|end_of_role|>");
    String intro = leadingInstruction == null ? INTRO : leadingInstruction + "\n\n" + INTRO;
    return prompt
        .text(intro)
        .control(DOCUMENTS_OPEN)
        .control(DOCUMENTS_CLOSE)
        .text(TAGS_SUFFIX)
        .control(DOCUMENTS_OPEN)
        .text(documentsBlock(documentTexts))
        .control(DOCUMENTS_CLOSE)
        .text(OUTRO)
        .control("<|end_of_text|>\n");
  }

  /** Appends one user or assistant turn. */
  static ModelPrompt.Builder appendTurn(ModelPrompt.Builder prompt, String role, String text) {
    if (!("user".equals(role) || "assistant".equals(role))) {
      throw new IllegalArgumentException("unsupported role: " + role);
    }
    return prompt
        .control("<|start_of_role|>" + role + "<|end_of_role|>")
        .text(text)
        .control("<|end_of_text|>\n");
  }

  /**
   * Closes the prompt with the assistant marker, which is also the activated adapters' boundary.
   */
  static ModelPrompt finish(ModelPrompt.Builder prompt) {
    return prompt.control(ASSISTANT_MARKER).build();
  }
}
