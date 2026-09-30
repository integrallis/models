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
package com.integrallis.models.spring.ai;

import com.integrallis.models.router.RoutingRequest;
import com.integrallis.models.router.RoutingTokenBounds;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.function.ToIntFunction;
import org.springframework.ai.chat.messages.*;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.ai.content.MediaContent;
import org.springframework.ai.model.tool.ToolCallingChatOptions;

/** Complete standard-prompt accounting; vendor state/framing must be bounded by the application. */
public final class RoutingRequests {
  private RoutingRequests() {}

  /**
   * Includes all history/system text, tool definitions/calls/results, media and reserved output.
   * Text counters see standard tool descriptions; provider-specific serialization and unresolved
   * tool names must be included in {@code providerOverheadTokens}. No media is fetched or decoded.
   */
  public static RoutingTokenBounds account(
      Prompt prompt,
      ToIntFunction<String> textTokens,
      ToIntFunction<Media> mediaTokens,
      int providerOverheadTokens,
      int outputTokens) {
    Objects.requireNonNull(prompt, "prompt");
    Objects.requireNonNull(textTokens, "textTokens");
    Objects.requireNonNull(mediaTokens, "mediaTokens");
    int input = checked(providerOverheadTokens);
    for (Message message : prompt.getInstructions()) {
      if (message.getText() != null) input = add(input, textTokens.applyAsInt(message.getText()));
      if (message instanceof MediaContent content) {
        for (Media media : content.getMedia()) input = add(input, mediaTokens.applyAsInt(media));
      }
      if (message instanceof AssistantMessage assistant && assistant.hasToolCalls())
        input = add(input, textTokens.applyAsInt(assistant.getToolCalls().toString()));
      if (message instanceof ToolResponseMessage tool)
        input = add(input, textTokens.applyAsInt(tool.getResponses().toString()));
    }
    if (prompt.getOptions() instanceof ToolCallingChatOptions tools
        && tools.getToolCallbacks() != null) {
      for (var tool : tools.getToolCallbacks())
        input = add(input, textTokens.applyAsInt(tool.getToolDefinition().toString()));
    }
    return new RoutingTokenBounds(input, outputTokens);
  }

  /** Text-only context estimate. Spending controls require verified, model-specific bounds. */
  public static RoutingRequest estimate(Prompt prompt) {
    String query = "";
    for (Message message : prompt.getInstructions())
      if (message instanceof UserMessage user) query = user.getText();
    int output =
        prompt.getOptions() == null || prompt.getOptions().getMaxTokens() == null
            ? 0
            : prompt.getOptions().getMaxTokens();
    var estimate =
        account(
            prompt,
            text -> text.getBytes(StandardCharsets.UTF_8).length,
            media -> {
              throw new IllegalArgumentException(
                  "media requires an explicit request accounting factory");
            },
            Math.multiplyExact(32, prompt.getInstructions().size()),
            output);
    return RoutingRequest.builder(query == null ? "" : query)
        .estimatedTokens(estimate.contextTokens())
        .build();
  }

  private static int add(int total, int value) {
    return Math.addExact(total, checked(value));
  }

  private static int checked(int value) {
    if (value < 0) throw new IllegalArgumentException("token counts must not be negative");
    return value;
  }
}
