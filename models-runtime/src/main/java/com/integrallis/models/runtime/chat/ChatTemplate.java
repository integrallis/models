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

import com.integrallis.models.api.ModelPrompt;
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

  private static final String DEEPSEEK_BOS = "<｜begin▁of▁sentence｜>";
  private static final String DEEPSEEK_DEFAULT_SYSTEM =
      "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by "
          + "Deepseek Company, and you only answer questions related to computer science. For "
          + "politically sensitive questions, security and privacy issues, and other "
          + "non-computer science questions, you will refuse to answer";

  private final String id;

  ChatTemplate(String id) {
    this.id = id;
  }

  /** Stable identifier recorded by ModelJars qualification metadata. */
  public String id() {
    return id;
  }

  /** Renders a conversation while keeping template controls separate from ordinary message text. */
  public ModelPrompt render(List<ChatMessage> messages) {
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

  private static ModelPrompt renderRaw(List<ChatMessage> messages) {
    ModelPrompt.Builder prompt = ModelPrompt.builder();
    for (int index = 0; index < messages.size(); index++) {
      if (index > 0) {
        prompt.control("\n");
      }
      prompt.text(messages.get(index).text());
    }
    return prompt.build();
  }

  private static ModelPrompt renderChatMl(
      List<ChatMessage> messages, String prefix, String assistantPrefix) {
    ModelPrompt.Builder prompt = ModelPrompt.builder().control(prefix);
    for (ChatMessage message : messages) {
      prompt
          .control("<|im_start|>" + message.role().templateName() + "\n")
          .text(message.text())
          .control("<|im_end|>\n");
    }
    return prompt.control("<|im_start|>assistant\n" + assistantPrefix).build();
  }

  private static ModelPrompt renderZephyr(List<ChatMessage> messages) {
    ModelPrompt.Builder prompt = ModelPrompt.builder();
    for (ChatMessage message : messages) {
      prompt
          .control("<|" + message.role().templateName() + "|>\n")
          .text(message.text())
          .control("</s>\n");
    }
    return prompt.control("<|assistant|>").build();
  }

  private static ModelPrompt renderLlama3(List<ChatMessage> messages) {
    ModelPrompt.Builder prompt = ModelPrompt.builder();
    for (ChatMessage message : messages) {
      prompt
          .control("<|start_header_id|>" + message.role().templateName() + "<|end_header_id|>\n\n")
          .text(message.text().strip())
          .control("<|eot_id|>");
    }
    return prompt.control("<|start_header_id|>assistant<|end_header_id|>\n\n").build();
  }

  private static ModelPrompt renderGemma(List<ChatMessage> messages) {
    ModelPrompt.Builder prompt = ModelPrompt.builder();
    boolean pendingUser = false;
    for (ChatMessage message : messages) {
      if (message.role() == ChatRole.ASSISTANT) {
        if (pendingUser) {
          prompt.control("<end_of_turn>\n");
          pendingUser = false;
        }
        prompt
            .control("<start_of_turn>model\n")
            .text(message.text().strip())
            .control("<end_of_turn>\n");
      } else {
        if (pendingUser) {
          prompt.control("\n\n");
        } else {
          prompt.control("<start_of_turn>user\n");
        }
        prompt.text(message.text().strip());
        pendingUser = true;
      }
    }
    if (pendingUser) {
      prompt.control("<end_of_turn>\n");
    }
    return prompt.control("<start_of_turn>model\n").build();
  }

  private static ModelPrompt renderPhi3(List<ChatMessage> messages) {
    ModelPrompt.Builder prompt = ModelPrompt.builder();
    for (ChatMessage message : messages) {
      prompt
          .control("<|" + message.role().templateName() + "|>\n")
          .text(message.text().strip())
          .control("<|end|>\n");
    }
    return prompt.control("<|assistant|>\n").build();
  }

  private static ModelPrompt renderDeepSeek(List<ChatMessage> messages) {
    ModelPrompt.Builder prompt = ModelPrompt.builder().control(DEEPSEEK_BOS);
    boolean hasSystemMessage =
        messages.stream().anyMatch(message -> message.role() == ChatRole.SYSTEM);
    if (!hasSystemMessage) {
      prompt.text(DEEPSEEK_DEFAULT_SYSTEM).control("\n");
    }

    for (ChatMessage message : messages) {
      switch (message.role()) {
        case SYSTEM -> prompt.text(message.text());
        case USER -> prompt.control("### Instruction:\n").text(message.text()).control("\n");
        case ASSISTANT ->
            prompt.control("### Response:\n").text(message.text()).control("\n<|EOT|>\n");
        case TOOL ->
            throw new IllegalArgumentException(
                "DeepSeek Coder chat template does not support role " + message.role());
      }
    }
    return prompt.control("### Response:\n").build();
  }

  private static ModelPrompt renderH2o(List<ChatMessage> messages, String assistantPrefix) {
    ModelPrompt.Builder prompt = ModelPrompt.builder();
    for (ChatMessage message : messages) {
      switch (message.role()) {
        case USER -> prompt.control("<|prompt|>").text(message.text()).control("</s>");
        case ASSISTANT -> prompt.control("<|answer|>").text(message.text()).control("</s>");
        case SYSTEM, TOOL ->
            throw new IllegalArgumentException(
                "H2O chat template does not support role " + message.role());
      }
    }
    return prompt.control("<|answer|>" + assistantPrefix).build();
  }
}
