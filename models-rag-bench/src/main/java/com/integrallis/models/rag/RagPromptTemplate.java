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

import com.integrallis.models.api.ModelPrompt;
import java.util.Locale;

/** Explicit prompt envelopes used to keep native and pure-Java requests byte-identical. */
public enum RagPromptTemplate {
  RAW("raw"),
  CHATML("chatml"),
  CHATML_DIRECT("chatml-direct"),
  CHATML_ANSWER("chatml-answer"),
  CHATML_NO_THINK("chatml-no-think"),
  ZEPHYR("zephyr"),
  LLAMA3("llama3"),
  GEMMA("gemma"),
  GEMMA4("gemma4"),
  PHI3("phi3"),
  DEEPSEEK("deepseek"),
  H2O("h2o"),
  H2O_DIRECT("h2o-direct"),
  MINICPM5_NO_THINK("minicpm5-no-think");

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
    return applyPrompt(prompt).text();
  }

  /** Applies a single-turn envelope while preserving trusted template-token boundaries. */
  public ModelPrompt applyPrompt(String prompt) {
    ModelPrompt.Builder result = ModelPrompt.builder();
    return switch (this) {
      case RAW -> result.text(prompt).build();
      case CHATML ->
          result
              .control("<|im_start|>user\n")
              .text(prompt)
              .control("<|im_end|>\n<|im_start|>assistant\n")
              .build();
      case CHATML_DIRECT ->
          result
              .control("<|im_start|>user\n")
              .text(prompt)
              .control("<|im_end|>\n<|im_start|>assistant\n")
              .text("The context states that ")
              .build();
      case CHATML_ANSWER ->
          result
              .control("<|im_start|>user\n")
              .text(prompt)
              .control("<|im_end|>\n<|im_start|>assistant\n")
              .text("Answer: ")
              .build();
      case CHATML_NO_THINK ->
          result
              .control("<|im_start|>user\n")
              .text(prompt)
              .control("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
              .build();
      case ZEPHYR ->
          result.control("<|user|>\n").text(prompt).control("</s>\n<|assistant|>").build();
      case LLAMA3 ->
          result
              .control("<|start_header_id|>user<|end_header_id|>\n\n")
              .text(prompt.strip())
              .control("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
              .build();
      case GEMMA ->
          result
              .control("<start_of_turn>user\n")
              .text(prompt.strip())
              .control("<end_of_turn>\n<start_of_turn>model\n")
              .build();
      case GEMMA4 ->
          result
              .control("<|turn>user\n")
              .text(prompt.strip())
              .control("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
              .build();
      case PHI3 ->
          result
              .control("<|user|>\n")
              .text(prompt.strip())
              .control("<|end|>\n<|assistant|>\n")
              .build();
      case DEEPSEEK ->
          result.control("### Instruction:\n").text(prompt).control("\n### Response:\n").build();
      case H2O ->
          result.control("<|prompt|>").text(prompt.strip()).control("</s><|answer|>").build();
      case H2O_DIRECT ->
          result
              .control("<|prompt|>")
              .text(prompt.strip())
              .control("</s><|answer|>")
              .text("The context states that ")
              .build();
      case MINICPM5_NO_THINK ->
          result
              .control("<s><|im_start|>user\n")
              .text(prompt)
              .control("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
              .build();
    };
  }

  /** Applies a role-aware envelope while retaining legacy prompt bytes for single-turn profiles. */
  public String apply(String systemPrompt, String userPrompt) {
    return applyPrompt(systemPrompt, userPrompt).text();
  }

  /** Applies a role-aware envelope while preserving trusted template-token boundaries. */
  public ModelPrompt applyPrompt(String systemPrompt, String userPrompt) {
    ModelPrompt.Builder result = ModelPrompt.builder();
    return switch (this) {
      case RAW -> result.text(systemPrompt).text(userPrompt).build();
      case CHATML, CHATML_DIRECT, CHATML_ANSWER, CHATML_NO_THINK -> {
        result
            .control("<|im_start|>system\n")
            .text(systemPrompt.stripTrailing())
            .control("<|im_end|>\n<|im_start|>user\n")
            .text(userPrompt)
            .control("<|im_end|>\n<|im_start|>assistant\n");
        if (this == CHATML_DIRECT) {
          result.text("The context states that ");
        } else if (this == CHATML_ANSWER) {
          result.text("Answer: ");
        } else if (this == CHATML_NO_THINK) {
          result.control("<think>\n\n</think>\n\n");
        }
        yield result.build();
      }
      case ZEPHYR ->
          result
              .control("<|system|>\n")
              .text(systemPrompt.stripTrailing())
              .control("</s>\n<|user|>\n")
              .text(userPrompt)
              .control("</s>\n<|assistant|>")
              .build();
      case LLAMA3 ->
          result
              .control("<|start_header_id|>system<|end_header_id|>\n\n")
              .text(systemPrompt.strip())
              .control("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n")
              .text(userPrompt.strip())
              .control("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
              .build();
      case GEMMA ->
          result
              .control("<start_of_turn>user\n")
              .text(systemPrompt.strip())
              .text("\n\n")
              .text(userPrompt.strip())
              .control("<end_of_turn>\n<start_of_turn>model\n")
              .build();
      case GEMMA4 ->
          result
              .control("<|turn>system\n")
              .text(systemPrompt.strip())
              .control("<turn|>\n<|turn>user\n")
              .text(userPrompt.strip())
              .control("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
              .build();
      case PHI3 ->
          result
              .control("<|system|>\n")
              .text(systemPrompt.strip())
              .control("<|end|>\n<|user|>\n")
              .text(userPrompt.strip())
              .control("<|end|>\n<|assistant|>\n")
              .build();
      case DEEPSEEK ->
          result
              .control("### Instruction:\n")
              .text(systemPrompt.stripTrailing())
              .text("\n\n")
              .text(userPrompt)
              .control("\n### Response:\n")
              .build();
      case H2O, H2O_DIRECT -> {
        result
            .control("<|prompt|>")
            .text(systemPrompt.strip())
            .text("\n\n")
            .text(userPrompt.strip())
            .control("</s><|answer|>");
        if (this == H2O_DIRECT) {
          result.text("The context states that ");
        }
        yield result.build();
      }
      case MINICPM5_NO_THINK ->
          result
              .control("<s><|im_start|>system\n")
              .text(systemPrompt.stripTrailing())
              .control("<|im_end|>\n<|im_start|>user\n")
              .text(userPrompt)
              .control("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
              .build();
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
        "prompt-template must be one of raw, chatml, chatml-direct, chatml-answer, "
            + "chatml-no-think, zephyr, llama3, gemma, gemma4, phi3, deepseek, h2o, h2o-direct, "
            + "minicpm5-no-think");
  }
}
