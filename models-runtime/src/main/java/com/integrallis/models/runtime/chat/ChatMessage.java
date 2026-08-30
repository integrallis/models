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

import com.integrallis.models.api.ToolCall;
import java.util.List;
import java.util.Objects;

/** One role-aware message rendered into a model's required chat template. */
public record ChatMessage(ChatRole role, String text, List<ToolCall> toolCalls, String name) {

  public ChatMessage {
    role = Objects.requireNonNull(role, "role");
    toolCalls = toolCalls == null ? List.of() : List.copyOf(toolCalls);
    text = text == null ? "" : text;
    name = name == null ? "" : name;
    // An assistant turn may be nothing but a tool call, so blank text is valid there and
    // nowhere else.
    if (text.isBlank() && toolCalls.isEmpty()) {
      throw new IllegalArgumentException("text must not be blank");
    }
    if (role != ChatRole.TOOL && !name.isEmpty()) {
      throw new IllegalArgumentException("name is supported only for tool messages");
    }
  }

  public ChatMessage(ChatRole role, String text, List<ToolCall> toolCalls) {
    this(role, text, toolCalls, "");
  }

  public ChatMessage(ChatRole role, String text) {
    this(role, text, List.of(), "");
  }

  public static ChatMessage system(String text) {
    return new ChatMessage(ChatRole.SYSTEM, text);
  }

  public static ChatMessage user(String text) {
    return new ChatMessage(ChatRole.USER, text);
  }

  public static ChatMessage assistant(String text) {
    return new ChatMessage(ChatRole.ASSISTANT, text);
  }

  public static ChatMessage tool(String text) {
    return new ChatMessage(ChatRole.TOOL, text);
  }

  /** A named tool result for protocols, such as Harmony, that address functions explicitly. */
  public static ChatMessage tool(String name, String text) {
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("tool name must not be blank");
    }
    return new ChatMessage(ChatRole.TOOL, text, List.of(), name);
  }

  /** An assistant turn that invokes tools, optionally alongside prose. */
  public static ChatMessage assistantToolCalls(String text, List<ToolCall> toolCalls) {
    if (toolCalls == null || toolCalls.isEmpty()) {
      throw new IllegalArgumentException("toolCalls must not be empty");
    }
    return new ChatMessage(ChatRole.ASSISTANT, text, toolCalls);
  }

  /** Whether this message invokes any tools. */
  public boolean hasToolCalls() {
    return !toolCalls.isEmpty();
  }
}
