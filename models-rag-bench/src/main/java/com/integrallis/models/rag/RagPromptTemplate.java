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
  CHATML_DIRECT("chatml-direct"),
  CHATML_NO_THINK("chatml-no-think"),
  ZEPHYR("zephyr"),
  LLAMA3("llama3"),
  GEMMA("gemma"),
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
    return switch (this) {
      case RAW -> prompt;
      case CHATML -> "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
      case CHATML_DIRECT ->
          "<|im_start|>user\n"
              + prompt
              + "<|im_end|>\n<|im_start|>assistant\nThe context states that ";
      case CHATML_NO_THINK ->
          "<|im_start|>user\n"
              + prompt
              + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
      case ZEPHYR -> "<|user|>\n" + prompt + "</s>\n<|assistant|>";
      case LLAMA3 ->
          "<|start_header_id|>user<|end_header_id|>\n\n"
              + prompt.strip()
              + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
      case GEMMA ->
          "<start_of_turn>user\n" + prompt.strip() + "<end_of_turn>\n<start_of_turn>model\n";
      case PHI3 -> "<|user|>\n" + prompt.strip() + "<|end|>\n<|assistant|>\n";
      case DEEPSEEK -> "### Instruction:\n" + prompt + "\n### Response:\n";
      case H2O -> "<|prompt|>" + prompt.strip() + "</s><|answer|>";
      case H2O_DIRECT -> "<|prompt|>" + prompt.strip() + "</s><|answer|>The context states that ";
      case MINICPM5_NO_THINK ->
          "<s><|im_start|>user\n"
              + prompt
              + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    };
  }

  /** Applies a role-aware envelope while retaining legacy prompt bytes for single-turn profiles. */
  public String apply(String systemPrompt, String userPrompt) {
    return switch (this) {
      case RAW -> systemPrompt + userPrompt;
      case CHATML ->
          "<|im_start|>system\n"
              + systemPrompt.stripTrailing()
              + "<|im_end|>\n<|im_start|>user\n"
              + userPrompt
              + "<|im_end|>\n<|im_start|>assistant\n";
      case CHATML_DIRECT ->
          "<|im_start|>system\n"
              + systemPrompt.stripTrailing()
              + "<|im_end|>\n<|im_start|>user\n"
              + userPrompt
              + "<|im_end|>\n<|im_start|>assistant\nThe context states that ";
      case CHATML_NO_THINK ->
          "<|im_start|>system\n"
              + systemPrompt.stripTrailing()
              + "<|im_end|>\n<|im_start|>user\n"
              + userPrompt
              + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
      case ZEPHYR ->
          "<|system|>\n"
              + systemPrompt.stripTrailing()
              + "</s>\n<|user|>\n"
              + userPrompt
              + "</s>\n<|assistant|>";
      case LLAMA3 ->
          "<|start_header_id|>system<|end_header_id|>\n\n"
              + systemPrompt.strip()
              + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
              + userPrompt.strip()
              + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
      case GEMMA ->
          "<start_of_turn>user\n"
              + systemPrompt.strip()
              + "\n\n"
              + userPrompt.strip()
              + "<end_of_turn>\n<start_of_turn>model\n";
      case PHI3 ->
          "<|system|>\n"
              + systemPrompt.strip()
              + "<|end|>\n<|user|>\n"
              + userPrompt.strip()
              + "<|end|>\n<|assistant|>\n";
      case DEEPSEEK ->
          "### Instruction:\n"
              + systemPrompt.stripTrailing()
              + "\n\n"
              + userPrompt
              + "\n### Response:\n";
      case H2O ->
          "<|prompt|>" + systemPrompt.strip() + "\n\n" + userPrompt.strip() + "</s><|answer|>";
      case H2O_DIRECT ->
          "<|prompt|>"
              + systemPrompt.strip()
              + "\n\n"
              + userPrompt.strip()
              + "</s><|answer|>The context states that ";
      case MINICPM5_NO_THINK ->
          "<s><|im_start|>system\n"
              + systemPrompt.stripTrailing()
              + "<|im_end|>\n<|im_start|>user\n"
              + userPrompt
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
        "prompt-template must be one of raw, chatml, chatml-direct, chatml-no-think, zephyr, "
            + "llama3, gemma, phi3, deepseek, h2o, h2o-direct, minicpm5-no-think");
  }
}
