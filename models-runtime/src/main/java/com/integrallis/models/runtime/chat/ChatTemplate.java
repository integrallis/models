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
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
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
  RAW("raw", ToolSyntax.NONE),
  CHATML("chatml", ToolSyntax.QWEN),
  CHATML_DIRECT("chatml-direct", ToolSyntax.QWEN),
  CHATML_ANSWER("chatml-answer", ToolSyntax.QWEN),
  CHATML_NO_THINK("chatml-no-think", ToolSyntax.QWEN),
  ZEPHYR("zephyr", ToolSyntax.NONE),
  LLAMA3("llama3", ToolSyntax.LLAMA3),
  GPT_OSS("gpt-oss", ToolSyntax.HARMONY),
  NEEDLE2("needle2", ToolSyntax.NEEDLE2),
  GEMMA("gemma", ToolSyntax.NONE),
  GEMMA4("gemma4", ToolSyntax.GEMMA4),
  PHI3("phi3", ToolSyntax.NONE),
  DEEPSEEK("deepseek", ToolSyntax.NONE),
  H2O("h2o", ToolSyntax.NONE),
  H2O_DIRECT("h2o-direct", ToolSyntax.NONE),
  MINICPM5_NO_THINK("minicpm5-no-think", ToolSyntax.MINICPM5);

  // Verbatim from Qwen's published chat template; the wording is part of what the model was
  // trained on, so it is reproduced exactly rather than paraphrased.
  private static final String QWEN_TOOLS_PREAMBLE =
      "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
          + "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
  private static final String QWEN_TOOLS_EPILOGUE =
      "\n</tools>\n\nFor each function call, return a json object with function name and arguments"
          + " within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>,"
          + " \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
  private static final String LLAMA3_TOOLS_PREAMBLE =
      "You have access to the following functions. To call a function, please respond with JSON"
          + " for a function call.\n\nRespond in the format {\"name\": function name,"
          + " \"parameters\": dictionary of argument name and its value}.\n\n";
  private static final String GPT_OSS_SYSTEM =
      "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
          + "Knowledge cutoff: 2024-06\n\nReasoning: medium\n\n"
          + "# Valid channels: analysis, commentary, final. Channel must be included for every"
          + " message.";

  private static final String DEEPSEEK_BOS = "<｜begin▁of▁sentence｜>";
  private static final String DEEPSEEK_DEFAULT_SYSTEM =
      "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by "
          + "Deepseek Company, and you only answer questions related to computer science. For "
          + "politically sensitive questions, security and privacy issues, and other "
          + "non-computer science questions, you will refuse to answer";

  private final String id;
  private final ToolSyntax toolSyntax;

  ChatTemplate(String id, ToolSyntax toolSyntax) {
    this.id = id;
    this.toolSyntax = toolSyntax;
  }

  /** Stable identifier recorded by ModelJars qualification metadata. */
  public String id() {
    return id;
  }

  /**
   * How this family expresses tool calls, in both directions.
   *
   * <p>Values are taken from each family's published chat template. Families with no trained format
   * map to {@link ToolSyntax#NONE}; emitting an invented format would produce output the model was
   * never trained on, which looks plausible and never parses.
   */
  public ToolSyntax toolSyntax() {
    return toolSyntax;
  }

  /** Whether this family has a trained tool-call format at all. */
  public boolean supportsTools() {
    return toolSyntax.supportsTools();
  }

  /**
   * Whether this runtime can recover tool calls from this family's output.
   *
   * <p>This is the check adapters should gate on. It is narrower than {@link #supportsTools()}:
   * Gemma 4 and MiniCPM5 have real tool formats, but both encode arguments as tagged pairs that
   * cannot be turned into JSON without the declared tool schemas.
   */
  public boolean canParseToolCalls() {
    return toolSyntax.parsable();
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
      case GPT_OSS -> renderGptOss(conversation, List.of());
      case NEEDLE2 -> renderNeedle2(conversation, List.of());
      case GEMMA -> renderGemma(conversation);
      case GEMMA4 -> renderGemma4(conversation);
      case PHI3 -> renderPhi3(conversation);
      case DEEPSEEK -> renderDeepSeek(conversation);
      case H2O -> renderH2o(conversation, "");
      case H2O_DIRECT -> renderH2o(conversation, "The context states that ");
      case MINICPM5_NO_THINK -> renderChatMl(conversation, "<s>", "<think>\n\n</think>\n\n");
    };
  }

  /**
   * Renders a conversation that declares tools.
   *
   * <p>Delegates to {@link #render(List)} when nothing tool-related is present, so callers can use
   * this overload unconditionally.
   *
   * <p>Trust boundary: template-owned delimiters are emitted as {@code CONTROL} so they tokenize to
   * the single ids the model was trained on, while everything supplied from outside — tool schemas,
   * call arguments, and above all tool results — is emitted as {@code TEXT}. Because the tokenizer
   * only recognises registered tokens inside {@code CONTROL}, a tool result that spells out control
   * markers cannot close a span or open a turn.
   *
   * @throws IllegalArgumentException if this family cannot express tool calls
   */
  public ModelPrompt render(List<ChatMessage> messages, List<ToolSpec> tools) {
    List<ChatMessage> conversation = validated(messages);
    List<ToolSpec> declared = tools == null ? List.of() : List.copyOf(tools);
    boolean conversationCallsTools = conversation.stream().anyMatch(ChatMessage::hasToolCalls);
    if (declared.isEmpty() && !conversationCallsTools) {
      return render(messages);
    }
    if (!canParseToolCalls()) {
      throw new IllegalArgumentException(
          "chat template " + id + " cannot express tool calls; check canParseToolCalls() first");
    }
    return switch (this) {
      case CHATML -> renderChatMlWithTools(conversation, "", "", declared);
      case CHATML_DIRECT ->
          renderChatMlWithTools(conversation, "", "The context states that ", declared);
      case CHATML_ANSWER -> renderChatMlWithTools(conversation, "", "Answer: ", declared);
      case CHATML_NO_THINK ->
          renderChatMlWithTools(conversation, "", "<think>\n\n</think>\n\n", declared);
      case LLAMA3 -> renderLlama3WithTools(conversation, declared);
      case GPT_OSS -> renderGptOss(conversation, declared);
      case NEEDLE2 -> renderNeedle2(conversation, declared);
      default ->
          throw new IllegalArgumentException(
              "chat template "
                  + id
                  + " cannot express tool calls; check canParseToolCalls() first");
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

  private static ModelPrompt renderChatMlWithTools(
      List<ChatMessage> messages, String prefix, String assistantPrefix, List<ToolSpec> tools) {
    ModelPrompt.Builder prompt = ModelPrompt.builder().control(prefix);

    int start = 0;
    if (!tools.isEmpty()) {
      // The tool declarations and any caller system prompt share a single system turn.
      prompt.control("<|im_start|>system\n");
      if (messages.get(0).role() == ChatRole.SYSTEM) {
        prompt.text(messages.get(0).text()).control("\n\n");
        start = 1;
      }
      prompt.control(QWEN_TOOLS_PREAMBLE);
      for (ToolSpec tool : tools) {
        prompt.control("\n");
        appendToolJson(prompt, tool);
      }
      prompt.control(QWEN_TOOLS_EPILOGUE);
    }

    for (int index = start; index < messages.size(); index++) {
      ChatMessage message = messages.get(index);
      if (message.role() == ChatRole.TOOL) {
        boolean firstOfRun = index == start || messages.get(index - 1).role() != ChatRole.TOOL;
        boolean lastOfRun =
            index == messages.size() - 1 || messages.get(index + 1).role() != ChatRole.TOOL;
        if (firstOfRun) {
          prompt.control("<|im_start|>user");
        }
        prompt.control("\n<tool_response>\n").text(message.text()).control("\n</tool_response>");
        if (lastOfRun) {
          prompt.control("<|im_end|>\n");
        }
        continue;
      }

      prompt.control("<|im_start|>" + message.role().templateName() + "\n");
      if (!message.text().isEmpty()) {
        prompt.text(message.text());
      }
      boolean firstCall = true;
      for (ToolCall call : message.toolCalls()) {
        if (!firstCall || !message.text().isEmpty()) {
          prompt.control("\n");
        }
        firstCall = false;
        prompt
            .control("<tool_call>\n{\"name\": \"")
            .text(call.name())
            .control("\", \"arguments\": ")
            .text(call.argumentsJson())
            .control("}\n</tool_call>");
      }
      prompt.control("<|im_end|>\n");
    }
    return prompt.control("<|im_start|>assistant\n" + assistantPrefix).build();
  }

  private static ModelPrompt renderLlama3WithTools(
      List<ChatMessage> messages, List<ToolSpec> tools) {
    ModelPrompt.Builder prompt = ModelPrompt.builder();

    int start = 0;
    if (messages.get(0).role() == ChatRole.SYSTEM) {
      prompt.control("<|start_header_id|>system<|end_header_id|>\n\n");
      if (!tools.isEmpty()) {
        prompt.control("Environment: ipython\n\n");
      }
      prompt.text(messages.get(0).text().strip()).control("<|eot_id|>");
      start = 1;
    } else if (!tools.isEmpty()) {
      prompt.control(
          "<|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython<|eot_id|>");
    }

    boolean toolsRendered = tools.isEmpty();
    for (int index = start; index < messages.size(); index++) {
      ChatMessage message = messages.get(index);
      if (message.role() == ChatRole.TOOL) {
        // Llama 3.x returns results under `ipython` rather than a `tool` role.
        prompt
            .control("<|start_header_id|>ipython<|end_header_id|>\n\n")
            .text(message.text().strip())
            .control("<|eot_id|>");
        continue;
      }

      prompt.control(
          "<|start_header_id|>" + message.role().templateName() + "<|end_header_id|>\n\n");
      if (!toolsRendered && message.role() == ChatRole.USER) {
        // Schemas ride in the first user turn, not the system prompt.
        prompt.control(LLAMA3_TOOLS_PREAMBLE);
        for (ToolSpec tool : tools) {
          appendToolJson(prompt, tool);
          prompt.control("\n\n");
        }
        toolsRendered = true;
      }
      if (!message.text().isEmpty()) {
        prompt.text(message.text().strip());
      }
      if (message.toolCalls().size() > 1) {
        throw new IllegalArgumentException(
            "chat template " + ChatTemplate.LLAMA3.id + " renders at most one tool call per turn");
      }
      for (ToolCall call : message.toolCalls()) {
        prompt
            .control("{\"name\": \"")
            .text(call.name())
            .control("\", \"parameters\": ")
            .text(call.argumentsJson())
            .control("}");
      }
      prompt.control("<|eot_id|>");
    }
    return prompt.control("<|start_header_id|>assistant<|end_header_id|>\n\n").build();
  }

  /** Renders the official GPT-OSS Harmony conversation protocol. */
  private static ModelPrompt renderGptOss(List<ChatMessage> messages, List<ToolSpec> tools) {
    ModelPrompt.Builder prompt = ModelPrompt.builder().control(GPT_OSS_SYSTEM);
    if (!tools.isEmpty()) {
      prompt.control("\nCalls to these tools must go to the commentary channel: 'functions'.");
    }
    prompt.control("<|end|>");

    int start = 0;
    String instructions = "";
    if (messages.getFirst().role() == ChatRole.SYSTEM) {
      instructions = messages.getFirst().text();
      start = 1;
    }
    if (!instructions.isEmpty() || !tools.isEmpty()) {
      prompt.control("<|start|>developer<|message|>");
      if (!instructions.isEmpty()) {
        prompt.control("# Instructions\n\n").text(instructions);
        if (!tools.isEmpty()) {
          prompt.control("\n\n");
        }
      }
      if (!tools.isEmpty()) {
        prompt.text(HarmonyToolRenderer.render(tools));
      }
      prompt.control("<|end|>");
    }

    for (int index = start; index < messages.size(); index++) {
      ChatMessage message = messages.get(index);
      switch (message.role()) {
        case SYSTEM ->
            throw new IllegalArgumentException(
                "GPT-OSS accepts a system instruction only at the start");
        case USER ->
            prompt.control("<|start|>user<|message|>").text(message.text()).control("<|end|>");
        case ASSISTANT -> {
          if (!message.text().isEmpty()) {
            String channel = message.hasToolCalls() ? "analysis" : "final";
            prompt
                .control("<|start|>assistant<|channel|>" + channel + "<|message|>")
                .text(message.text())
                .control("<|end|>");
          }
          for (ToolCall call : message.toolCalls()) {
            prompt
                .control("<|start|>assistant<|channel|>commentary to=functions.")
                .text(call.name())
                .control(" <|constrain|>json<|message|>")
                .text(call.argumentsJson())
                .control("<|call|>");
          }
        }
        case TOOL -> {
          if (message.name().isBlank()) {
            throw new IllegalArgumentException("GPT-OSS Harmony requires the tool-result name");
          }
          prompt
              .control("<|start|>functions.")
              .text(message.name())
              .control(" to=assistant<|channel|>commentary<|message|>")
              .text(message.text())
              .control("<|end|>");
        }
      }
    }
    return prompt.control("<|start|>assistant").build();
  }

  /** Renders the protocol used by the official Needle 2 training and inference code. */
  private static ModelPrompt renderNeedle2(List<ChatMessage> messages, List<ToolSpec> tools) {
    ModelPrompt.Builder prompt = ModelPrompt.builder();
    int start = 0;
    if (messages.getFirst().role() == ChatRole.SYSTEM) {
      prompt
          .control("<|im_start|>system\n")
          .text(messages.getFirst().text())
          .control("<|im_end|>\n");
      start = 1;
    }

    boolean toolsRendered = false;
    for (int index = start; index < messages.size(); index++) {
      ChatMessage message = messages.get(index);
      switch (message.role()) {
        case SYSTEM ->
            throw new IllegalArgumentException("needle2 accepts a system turn only at the start");
        case USER, TOOL -> {
          prompt.control("<|im_start|>user\n");
          if (!toolsRendered && message.role() == ChatRole.USER) {
            appendNeedle2Tools(prompt, tools);
            prompt.control("\n");
            toolsRendered = true;
          }
          prompt.text(message.text()).control("<|im_end|>\n");
        }
        case ASSISTANT -> {
          prompt.control("<|im_start|>assistant\n<think>");
          if (!message.text().isEmpty()) {
            prompt.text(message.text());
          }
          prompt.control("</think>\n<tool_call>[");
          for (int callIndex = 0; callIndex < message.toolCalls().size(); callIndex++) {
            if (callIndex > 0) {
              prompt.control(",");
            }
            ToolCall call = message.toolCalls().get(callIndex);
            prompt
                .control("{\"name\":\"")
                .text(escapeJson(call.name()))
                .control("\",\"arguments\":")
                .text(call.argumentsJson())
                .control("}");
          }
          prompt.control("]</tool_call><|im_end|>\n");
        }
      }
    }
    return prompt.control("<|im_start|>assistant\n").build();
  }

  private static void appendNeedle2Tools(ModelPrompt.Builder prompt, List<ToolSpec> tools) {
    prompt.control("<tools>[");
    for (int index = 0; index < tools.size(); index++) {
      if (index > 0) {
        prompt.control(",");
      }
      ToolSpec tool = tools.get(index);
      prompt
          .control("{\"name\":\"")
          .text(escapeJson(tool.name()))
          .control("\",\"description\":\"")
          .text(escapeJson(tool.description()))
          .control("\",\"parameters\":")
          .text(tool.inputSchema())
          .control("}");
    }
    prompt.control("]</tools>");
  }

  /**
   * Appends one tool declaration in the OpenAI shape both families render.
   *
   * <p>The wrapper is template-owned so it stays {@code CONTROL}; the caller's name, description
   * and schema are {@code TEXT}. The schema is copied verbatim because it is already JSON.
   */
  private static void appendToolJson(ModelPrompt.Builder prompt, ToolSpec tool) {
    prompt
        .control("{\"type\": \"function\", \"function\": {\"name\": \"")
        .text(escapeJson(tool.name()))
        .control("\", \"description\": \"")
        .text(escapeJson(tool.description()))
        .control("\", \"parameters\": ")
        .text(tool.inputSchema())
        .control("}}");
  }

  /** Escapes a value for inclusion in a JSON string literal. */
  private static String escapeJson(String value) {
    StringBuilder escaped = new StringBuilder(value.length());
    for (int index = 0; index < value.length(); index++) {
      char current = value.charAt(index);
      switch (current) {
        case '"' -> escaped.append("\\\"");
        case '\\' -> escaped.append("\\\\");
        case '\n' -> escaped.append("\\n");
        case '\r' -> escaped.append("\\r");
        case '\t' -> escaped.append("\\t");
        default -> {
          if (current < 0x20) {
            escaped.append(String.format("\\u%04x", (int) current));
          } else {
            escaped.append(current);
          }
        }
      }
    }
    return escaped.toString();
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

  private static ModelPrompt renderGemma4(List<ChatMessage> messages) {
    ModelPrompt.Builder prompt = ModelPrompt.builder();
    for (int index = 0; index < messages.size(); index++) {
      ChatMessage message = messages.get(index);
      String role =
          switch (message.role()) {
            case SYSTEM -> {
              if (index != 0) {
                throw new IllegalArgumentException("Gemma 4 system message must be first");
              }
              yield "system";
            }
            case USER -> "user";
            case ASSISTANT -> "model";
            case TOOL ->
                throw new IllegalArgumentException(
                    "Gemma 4 text chat template does not support role " + message.role());
          };
      prompt.control("<|turn>" + role + "\n").text(message.text().strip()).control("<turn|>\n");
    }
    return prompt.control("<|turn>model\n<|channel>thought\n<channel|>").build();
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
