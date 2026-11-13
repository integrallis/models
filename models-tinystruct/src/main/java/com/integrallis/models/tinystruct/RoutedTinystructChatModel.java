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
import com.integrallis.models.router.ModelFleet;
import com.integrallis.models.router.RoutingContinuity;
import com.integrallis.models.router.RoutingExecutionOptions;
import com.integrallis.models.router.RoutingRequest;
import com.integrallis.models.router.RoutingRequirements;
import com.integrallis.models.router.RoutingUsage;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatRole;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import org.tinystruct.data.component.Builder;

/**
 * A {@link TinystructChatModel} that routes chat and streaming turns across a {@link ModelFleet} of
 * local and hosted {@link TinystructChatModel} delegates.
 */
public final class RoutedTinystructChatModel implements TinystructChatModel {

  private final ModelFleet<TinystructChatModel> fleet;
  private final Function<List<ChatMessage>, RoutingRequest> requestFactory;
  private final Function<List<ChatMessage>, RoutingContinuity> continuityFactory;
  private final Function<List<ChatMessage>, RoutingRequirements> requirementsFactory;
  private final Function<List<ChatMessage>, RoutingExecutionOptions> executionFactory;

  /** Routes chat requests from the conversation messages with default token estimation. */
  public RoutedTinystructChatModel(ModelFleet<TinystructChatModel> fleet) {
    this(
        fleet,
        RoutedTinystructChatModel::defaultRequest,
        RoutedTinystructChatModel::defaultContinuity,
        ignored -> RoutingRequirements.none(),
        ignored -> RoutingExecutionOptions.unlimited());
  }

  /** Routes chat requests with an application-supplied request factory. */
  public RoutedTinystructChatModel(
      ModelFleet<TinystructChatModel> fleet,
      Function<List<ChatMessage>, RoutingRequest> requestFactory) {
    this(
        fleet,
        requestFactory,
        RoutedTinystructChatModel::defaultContinuity,
        ignored -> RoutingRequirements.none(),
        ignored -> RoutingExecutionOptions.unlimited());
  }

  /**
   * Routes chat requests with custom request, continuity, requirements, and execution options
   * extractors.
   */
  public RoutedTinystructChatModel(
      ModelFleet<TinystructChatModel> fleet,
      Function<List<ChatMessage>, RoutingRequest> requestFactory,
      Function<List<ChatMessage>, RoutingContinuity> continuityFactory,
      Function<List<ChatMessage>, RoutingRequirements> requirementsFactory,
      Function<List<ChatMessage>, RoutingExecutionOptions> executionFactory) {
    this.fleet = Objects.requireNonNull(fleet, "fleet");
    this.requestFactory = Objects.requireNonNull(requestFactory, "requestFactory");
    this.continuityFactory = Objects.requireNonNull(continuityFactory, "continuityFactory");
    this.requirementsFactory = Objects.requireNonNull(requirementsFactory, "requirementsFactory");
    this.executionFactory = Objects.requireNonNull(executionFactory, "executionFactory");
  }

  @Override
  public Builder chat(List<ChatMessage> messages, SamplingOptions options, List<ToolSpec> tools) {
    Objects.requireNonNull(messages, "messages");
    RoutingRequest request =
        Objects.requireNonNull(requestFactory.apply(messages), "routing request");
    RoutingRequirements requirements =
        Objects.requireNonNull(requirementsFactory.apply(messages), "routing requirements");
    RoutingContinuity continuity =
        Objects.requireNonNull(continuityFactory.apply(messages), "routing continuity");
    RoutingExecutionOptions executionOptions =
        Objects.requireNonNull(executionFactory.apply(messages), "routing execution options");

    return fleet
        .execute(
            request,
            requirements,
            continuity,
            executionOptions,
            delegate -> delegate.chat(messages, options, tools),
            RoutedTinystructChatModel::extractUsage)
        .value();
  }

  @Override
  public Builder stream(
      List<ChatMessage> messages,
      SamplingOptions options,
      List<ToolSpec> tools,
      Consumer<String> onToken) {
    Objects.requireNonNull(messages, "messages");
    Objects.requireNonNull(onToken, "onToken");
    RoutingRequest request =
        Objects.requireNonNull(requestFactory.apply(messages), "routing request");
    RoutingRequirements requirements =
        Objects.requireNonNull(requirementsFactory.apply(messages), "routing requirements");
    RoutingContinuity continuity =
        Objects.requireNonNull(continuityFactory.apply(messages), "routing continuity");
    RoutingExecutionOptions executionOptions =
        Objects.requireNonNull(executionFactory.apply(messages), "routing execution options");

    return fleet
        .execute(
            request,
            requirements,
            continuity,
            executionOptions,
            delegate -> delegate.stream(messages, options, tools, onToken),
            RoutedTinystructChatModel::extractUsage)
        .value();
  }

  /** Estimates a {@link RoutingRequest} from the conversation messages. */
  public static RoutingRequest defaultRequest(List<ChatMessage> messages) {
    Objects.requireNonNull(messages, "messages");
    StringBuilder combined = new StringBuilder();
    String latestUser = "";
    for (ChatMessage message : messages) {
      if (message != null && message.text() != null) {
        if (!combined.isEmpty()) {
          combined.append('\n');
        }
        combined.append(message.text());
        if (message.role() == ChatRole.USER) {
          latestUser = message.text();
        }
      }
    }
    String promptText = !latestUser.isEmpty() ? latestUser : combined.toString();
    int estimatedInput = Math.max(1, (combined.length() + 3) / 4);
    return RoutingRequest.builder(promptText).estimatedTokens(estimatedInput).build();
  }

  static RoutingContinuity defaultContinuity(List<ChatMessage> messages) {
    return RoutingContinuity.none();
  }

  static RoutingUsage extractUsage(Builder response) {
    if (response == null || !(response.get("usage") instanceof Builder usage)) {
      return null;
    }
    Object promptTokens = usage.get("prompt_tokens");
    Object completionTokens = usage.get("completion_tokens");
    if (!(promptTokens instanceof Number prompt)
        || !(completionTokens instanceof Number completion)) {
      return null;
    }
    if (prompt.intValue() == 0 && completion.intValue() == 0) {
      return null;
    }
    return new RoutingUsage(Math.max(0, prompt.intValue()), Math.max(0, completion.intValue()));
  }
}
