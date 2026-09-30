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
package com.integrallis.models.langchain4j;

import com.integrallis.models.router.RoutingRequest;
import com.integrallis.models.router.RoutingTokenBounds;
import dev.langchain4j.data.message.*;
import dev.langchain4j.model.chat.request.ChatRequest;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.function.ToIntFunction;

/** Complete standard-request accounting without loading media or changing provider options. */
public final class RoutingRequests {
  private RoutingRequests() {}

  /**
   * Accounts for every history/system message, tool definition, tool call/result, media item and
   * reserved output. Supply model-specific token counters and a bound for provider framing, hidden
   * state and any vendor-specific options. Returned bounds are only as sound as those supplied
   * counters; the router cannot derive billable image/audio tokens from an attachment's byte size.
   */
  public static RoutingTokenBounds account(
      ChatRequest request,
      ToIntFunction<String> textTokens,
      ToIntFunction<Content> mediaTokens,
      int providerOverheadTokens,
      int outputTokens) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(textTokens, "textTokens");
    Objects.requireNonNull(mediaTokens, "mediaTokens");
    int input = checked(providerOverheadTokens);
    for (ChatMessage message : request.messages()) {
      if (message instanceof UserMessage user) {
        if (user.name() != null) input = add(input, textTokens.applyAsInt(user.name()));
        for (Content content : user.contents()) {
          input =
              add(
                  input,
                  content instanceof TextContent text
                      ? textTokens.applyAsInt(text.text())
                      : mediaTokens.applyAsInt(content));
        }
      } else if (message instanceof SystemMessage system) {
        input = add(input, textTokens.applyAsInt(system.text()));
      } else if (message instanceof AiMessage assistant) {
        if (assistant.text() != null) input = add(input, textTokens.applyAsInt(assistant.text()));
        if (assistant.toolExecutionRequests() != null)
          input = add(input, textTokens.applyAsInt(assistant.toolExecutionRequests().toString()));
      } else if (message instanceof ToolExecutionResultMessage tool) {
        input = add(input, textTokens.applyAsInt(tool.toString()));
      } else
        throw new IllegalArgumentException(
            "custom message requires application-supplied complete accounting");
    }
    if (request.toolSpecifications() != null)
      input = add(input, textTokens.applyAsInt(request.toolSpecifications().toString()));
    if (request.responseFormat() != null)
      input = add(input, textTokens.applyAsInt(request.responseFormat().toString()));
    return new RoutingTokenBounds(input, outputTokens);
  }

  /** Text-only admission estimate, not a verified bound for spending controls. */
  public static RoutingRequest estimate(ChatRequest request) {
    String query = "";
    for (ChatMessage message : request.messages()) {
      if (message instanceof UserMessage user) {
        query =
            user.contents().stream()
                .filter(TextContent.class::isInstance)
                .map(TextContent.class::cast)
                .map(TextContent::text)
                .collect(java.util.stream.Collectors.joining("\n"));
      }
    }
    RoutingTokenBounds estimate =
        account(
            request,
            text -> text.getBytes(StandardCharsets.UTF_8).length,
            media -> {
              throw new IllegalArgumentException(
                  "media requires an explicit request accounting factory");
            },
            Math.multiplyExact(32, request.messages().size()),
            request.maxOutputTokens() == null ? 0 : request.maxOutputTokens());
    return RoutingRequest.builder(query).estimatedTokens(estimate.contextTokens()).build();
  }

  private static int add(int total, int value) {
    return Math.addExact(total, checked(value));
  }

  private static int checked(int value) {
    if (value < 0) throw new IllegalArgumentException("token counts must not be negative");
    return value;
  }
}
