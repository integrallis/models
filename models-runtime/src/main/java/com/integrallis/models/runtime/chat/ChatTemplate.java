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
package com.integrallis.models.runtime.chat;

import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Model-facing chat envelopes used by qualified ModelJars artifacts.
 *
 * <p>Template selection is explicit because architecture names do not uniquely identify a model's
 * instruction format.
 */
public enum ChatTemplate {
  RAW("raw"),
  CHATML("chatml"),
  CHATML_DIRECT("chatml-direct"),
  CHATML_ANSWER("chatml-answer"),
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

  ChatTemplate(String id) {
    this.id = id;
  }

  /** Stable identifier recorded by ModelJars qualification metadata. */
  public String id() {
    return id;
  }

  /** Renders a complete conversation and opens the assistant turn to be generated. */
  public String render(List<ChatMessage> messages) {
    List<ChatMessage> conversation = validated(messages);
    return switch (this) {
      case RAW -> renderRaw(conversation);
      case CHATML -> renderChatMl(conversation, "", "");
      case CHATML_DIRECT -> renderChatMl(conversation, "", "The context states that ");
      case CHATML_ANSWER -> renderChatMl(conversation, "", "Answer: ");
      case CHATML_NO_THINK -> renderChatMl(conversation, "", "<think>\n\n</think>\n\n");
      case ZEPHYR -> renderZephyr(conversation);
      case LLAMA3 -> renderLlama3(conversation);
      case GEMMA -> renderGemma(conversation);
      case PHI3 -> renderPhi3(conversation);
      case DEEPSEEK -> renderDeepSeek(conversation);
      case H2O -> renderH2o(conversation, "");
      case H2O_DIRECT -> renderH2o(conversation, "The context states that ");
      case MINICPM5_NO_THINK -> renderChatMl(conversation, "<s>", "<think>\n\n</think>\n\n");
    };
  }

  /** Resolves the stable identifier recorded in ModelJars metadata. */
  public static ChatTemplate parse(String value) {
    if (value != null) {
      String normalized = value.trim().toLowerCase(Locale.ROOT);
      for (ChatTemplate template : values()) {
        if (template.id.equals(normalized)) {
          return template;
        }
      }
    }
    throw new IllegalArgumentException("Unknown chat template: " + value);
  }

  private static List<ChatMessage> validated(List<ChatMessage> messages) {
    Objects.requireNonNull(messages, "messages");
    if (messages.isEmpty()) {
      throw new IllegalArgumentException("messages must not be empty");
    }
    return List.copyOf(messages);
  }

  private static String renderRaw(List<ChatMessage> messages) {
    return String.join("\n", messages.stream().map(ChatMessage::text).toList());
  }

  private static String renderChatMl(
      List<ChatMessage> messages, String prefix, String assistantPrefix) {
    StringBuilder prompt = new StringBuilder(prefix);
    for (ChatMessage message : messages) {
      prompt
          .append("<|im_start|>")
          .append(message.role().templateName())
          .append('\n')
          .append(message.text())
          .append("<|im_end|>\n");
    }
    return prompt.append("<|im_start|>assistant\n").append(assistantPrefix).toString();
  }

  private static String renderZephyr(List<ChatMessage> messages) {
    StringBuilder prompt = new StringBuilder();
    for (ChatMessage message : messages) {
      prompt
          .append("<|")
          .append(message.role().templateName())
          .append("|>\n")
          .append(message.text())
          .append("</s>\n");
    }
    return prompt.append("<|assistant|>").toString();
  }

  private static String renderLlama3(List<ChatMessage> messages) {
    StringBuilder prompt = new StringBuilder();
    for (ChatMessage message : messages) {
      prompt
          .append("<|start_header_id|>")
          .append(message.role().templateName())
          .append("<|end_header_id|>\n\n")
          .append(message.text().strip())
          .append("<|eot_id|>");
    }
    return prompt.append("<|start_header_id|>assistant<|end_header_id|>\n\n").toString();
  }

  private static String renderGemma(List<ChatMessage> messages) {
    StringBuilder prompt = new StringBuilder();
    StringBuilder pendingUser = new StringBuilder();
    for (ChatMessage message : messages) {
      if (message.role() == ChatRole.ASSISTANT) {
        appendGemmaUser(prompt, pendingUser);
        prompt
            .append("<start_of_turn>model\n")
            .append(message.text().strip())
            .append("<end_of_turn>\n");
      } else {
        if (!pendingUser.isEmpty()) {
          pendingUser.append("\n\n");
        }
        pendingUser.append(message.text().strip());
      }
    }
    appendGemmaUser(prompt, pendingUser);
    return prompt.append("<start_of_turn>model\n").toString();
  }

  private static void appendGemmaUser(StringBuilder prompt, StringBuilder pendingUser) {
    if (!pendingUser.isEmpty()) {
      prompt.append("<start_of_turn>user\n").append(pendingUser).append("<end_of_turn>\n");
      pendingUser.setLength(0);
    }
  }

  private static String renderPhi3(List<ChatMessage> messages) {
    StringBuilder prompt = new StringBuilder();
    for (ChatMessage message : messages) {
      prompt
          .append("<|")
          .append(message.role().templateName())
          .append("|>\n")
          .append(message.text().strip())
          .append("<|end|>\n");
    }
    return prompt.append("<|assistant|>\n").toString();
  }

  private static String renderDeepSeek(List<ChatMessage> messages) {
    StringBuilder prompt = new StringBuilder();
    StringBuilder pendingInstruction = new StringBuilder();
    for (ChatMessage message : messages) {
      if (message.role() == ChatRole.ASSISTANT) {
        appendInstruction(prompt, pendingInstruction);
        prompt.append("### Response:\n").append(message.text().strip()).append('\n');
      } else {
        if (!pendingInstruction.isEmpty()) {
          pendingInstruction.append("\n\n");
        }
        pendingInstruction.append(message.text().strip());
      }
    }
    appendInstruction(prompt, pendingInstruction);
    return prompt.append("### Response:\n").toString();
  }

  private static void appendInstruction(StringBuilder prompt, StringBuilder instruction) {
    if (!instruction.isEmpty()) {
      prompt.append("### Instruction:\n").append(instruction).append('\n');
      instruction.setLength(0);
    }
  }

  private static String renderH2o(List<ChatMessage> messages, String assistantPrefix) {
    StringBuilder prompt = new StringBuilder();
    for (ChatMessage message : messages) {
      if (!prompt.isEmpty()) {
        prompt.append("\n\n");
      }
      if (message.role() == ChatRole.ASSISTANT) {
        prompt.append("Previous answer: ");
      }
      prompt.append(message.text().strip());
    }
    return "<|prompt|>" + prompt + "</s><|answer|>" + assistantPrefix;
  }
}
