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

import com.integrallis.models.router.ModelFleet;
import com.integrallis.models.router.RoutingContinuity;
import com.integrallis.models.router.RoutingRequest;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/** LangChain4j model that routes unchanged requests across local and hosted chat clients. */
public final class RoutedChatModel implements ChatModel {
  private final ModelFleet<ChatModel> fleet;
  private final Function<ChatRequest, RoutingRequest> requestFactory;
  private final Function<ChatRequest, RoutingContinuity> continuityFactory;

  /** Routes from the latest user message, without implicit session affinity. */
  public RoutedChatModel(ModelFleet<ChatModel> fleet) {
    this(fleet, RoutedChatModel::defaultRequest, RoutedChatModel::defaultContinuity);
  }

  /** Routes with an application-supplied mapping for token estimates, task, and session id. */
  public RoutedChatModel(
      ModelFleet<ChatModel> fleet, Function<ChatRequest, RoutingRequest> requestFactory) {
    this(fleet, requestFactory, RoutedChatModel::defaultContinuity);
  }

  /** Routes with application-supplied request and continuity extraction. */
  public RoutedChatModel(
      ModelFleet<ChatModel> fleet,
      Function<ChatRequest, RoutingRequest> requestFactory,
      Function<ChatRequest, RoutingContinuity> continuityFactory) {
    this.fleet = Objects.requireNonNull(fleet, "fleet");
    this.requestFactory = Objects.requireNonNull(requestFactory, "requestFactory");
    this.continuityFactory = Objects.requireNonNull(continuityFactory, "continuityFactory");
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    Objects.requireNonNull(request, "request");
    RoutingRequest routingRequest =
        Objects.requireNonNull(requestFactory.apply(request), "routing request");
    RoutingContinuity continuity =
        Objects.requireNonNull(continuityFactory.apply(request), "routing continuity");
    return fleet.execute(routingRequest, continuity, model -> model.doChat(request)).value();
  }

  private static RoutingRequest defaultRequest(ChatRequest request) {
    List<ChatMessage> messages = request.messages();
    for (int index = messages.size() - 1; index >= 0; index--) {
      if (messages.get(index) instanceof UserMessage user) {
        return RoutingRequest.builder(user.singleText()).build();
      }
    }
    return RoutingRequest.builder("").build();
  }

  private static RoutingContinuity defaultContinuity(ChatRequest request) {
    List<ChatMessage> messages = request.messages();
    boolean toolLoop =
        !messages.isEmpty() && messages.getLast() instanceof ToolExecutionResultMessage;
    return RoutingContinuity.builder().activeToolLoop(toolLoop).build();
  }
}
