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

import java.util.Locale;

/** Explicit prompt envelopes used to keep native and pure-Java requests byte-identical. */
public enum RagPromptTemplate {
  RAW("raw"),
  CHATML("chatml"),
  CHATML_NO_THINK("chatml-no-think");

  private final String id;

  RagPromptTemplate(String id) {
    this.id = id;
  }

  /** Stable report and CLI identifier. */
  public String id() {
    return id;
  }

  /** Applies this model-facing envelope to the canonical RAG prompt. */
  public String apply(String prompt) {
    return switch (this) {
      case RAW -> prompt;
      case CHATML -> "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
      case CHATML_NO_THINK ->
          "<|im_start|>user\n"
              + prompt
              + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    };
  }

  /** Resolves a CLI identifier. */
  public static RagPromptTemplate parse(String value) {
    for (RagPromptTemplate template : values()) {
      if (template.id.equals(value.toLowerCase(Locale.ROOT))) {
        return template;
      }
    }
    throw new IllegalArgumentException(
        "prompt-template must be one of raw, chatml, chatml-no-think");
  }
}
