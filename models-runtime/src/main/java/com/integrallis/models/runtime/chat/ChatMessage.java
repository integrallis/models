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

import java.util.Objects;

/** One role-aware message rendered into a model's required chat template. */
public record ChatMessage(ChatRole role, String text) {

  public ChatMessage {
    role = Objects.requireNonNull(role, "role");
    if (text == null || text.isBlank()) {
      throw new IllegalArgumentException("text must not be blank");
    }
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
}
