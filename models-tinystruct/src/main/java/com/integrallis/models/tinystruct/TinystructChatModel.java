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
package com.integrallis.models.tinystruct;

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.chat.ChatMessage;
import java.util.List;
import java.util.function.Consumer;
import org.tinystruct.data.component.Builder;

/**
 * Contract for tinystruct applications that exchange role-aware chat turns, tool specifications,
 * and token streams as native {@link Builder} structures.
 */
public interface TinystructChatModel {

  /**
   * Generates a response for a single user prompt using the model's default sampling options.
   *
   * @param userMessage the user prompt text
   * @return a {@link Builder} containing {@code text}, {@code content}, {@code model}, {@code
   *     finish_reason}, {@code usage}, and {@code tool_calls}
   */
  default Builder chat(String userMessage) {
    return chat(List.of(ChatMessage.user(userMessage)));
  }

  /**
   * Generates a response for a role-aware conversation using the model's default sampling options.
   *
   * @param messages the conversation messages
   * @return a {@link Builder} containing {@code text}, {@code content}, {@code model}, {@code
   *     finish_reason}, {@code usage}, and {@code tool_calls}
   */
  default Builder chat(List<ChatMessage> messages) {
    return chat(messages, null, List.of());
  }

  /**
   * Generates a response for a role-aware conversation with optional sampling overrides and
   * declared tools.
   *
   * @param messages the conversation messages
   * @param options sampling options, or {@code null} to use the model's defaults
   * @param tools tool specifications available for this turn
   * @return a {@link Builder} containing {@code text}, {@code content}, {@code model}, {@code
   *     finish_reason}, {@code usage}, and {@code tool_calls}
   */
  Builder chat(List<ChatMessage> messages, SamplingOptions options, List<ToolSpec> tools);

  /**
   * Streams tokens for a role-aware conversation to {@code onToken} and returns the final assembled
   * {@link Builder} response.
   *
   * @param messages the conversation messages
   * @param options sampling options, or {@code null} to use the model's defaults
   * @param tools tool specifications available for this turn
   * @param onToken callback invoked for each generated token
   * @return a {@link Builder} containing {@code text}, {@code content}, {@code model}, {@code
   *     finish_reason}, {@code usage}, and {@code tool_calls}
   */
  Builder stream(
      List<ChatMessage> messages,
      SamplingOptions options,
      List<ToolSpec> tools,
      Consumer<String> onToken);
}
