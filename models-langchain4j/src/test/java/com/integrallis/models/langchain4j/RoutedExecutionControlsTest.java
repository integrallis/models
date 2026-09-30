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

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.router.*;
import dev.langchain4j.data.message.*;
import dev.langchain4j.model.chat.*;
import dev.langchain4j.model.chat.request.*;
import dev.langchain4j.model.chat.response.*;
import java.lang.reflect.*;
import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.BiConsumer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RoutedExecutionControlsTest {
  private static final ChatRequest REQUEST =
      ChatRequest.builder()
          .messages(UserMessage.from("hello"))
          .parameters(DefaultChatRequestParameters.builder().maxOutputTokens(100).build())
          .build();

  private static StreamingChatModel model(
      BiConsumer<ChatRequest, StreamingChatResponseHandler> action) {
    return new StreamingChatModel() {
      @Override
      public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
        action.accept(request, handler);
      }
    };
  }

  private static ModelCandidate candidate(String id) {
    return ModelCandidate.builder(id).costPerMillionTokens(1000, 1000).build();
  }

  private static RoutedStreamingChatModel routed(
      ModelFleet<StreamingChatModel> fleet, RoutingExecutionOptions options) {
    return new RoutedStreamingChatModel(
        fleet,
        ignored -> RoutingRequest.builder("hello").build(),
        ignored -> RoutingContinuity.none(),
        ignored -> RoutingRequirements.none(),
        ignored -> options);
  }

  private static StreamingChatResponseHandler handler(
      AtomicReference<Throwable> error, AtomicInteger completions) {
    return new StreamingChatResponseHandler() {
      @Override
      public void onPartialResponse(String text) {}

      @Override
      public void onCompleteResponse(ChatResponse response) {
        completions.incrementAndGet();
      }

      @Override
      public void onError(Throwable failure) {
        error.set(failure);
      }
    };
  }

  @Test
  void forwardsEveryRuntimeCallbackWithOriginalArguments() throws Exception {
    List<Method> methods =
        Arrays.stream(StreamingChatResponseHandler.class.getMethods())
            .filter(
                method ->
                    method.getName().startsWith("on")
                        && !method.getName().equals("onError")
                        && !method.getName().equals("onCompleteResponse"))
            .toList();
    List<Object[]> sent = new ArrayList<>(), received = new ArrayList<>();
    List<Method> forwarded = new ArrayList<>();
    AtomicReference<Throwable> error = new AtomicReference<>();
    StreamingChatResponseHandler consumer =
        (StreamingChatResponseHandler)
            Proxy.newProxyInstance(
                StreamingChatResponseHandler.class.getClassLoader(),
                new Class<?>[] {StreamingChatResponseHandler.class},
                (proxy, method, args) -> {
                  if (method.getName().equals("onError")) error.set((Throwable) args[0]);
                  else {
                    forwarded.add(method);
                    received.add(args);
                  }
                  return null;
                });
    StreamingChatModel provider =
        model(
            (request, callback) -> {
              for (Method method : methods) {
                Object[] args =
                    Arrays.stream(method.getParameterTypes())
                        .map(RoutedExecutionControlsTest::argument)
                        .toArray();
                sent.add(args);
                try {
                  method.invoke(callback, args);
                } catch (ReflectiveOperationException failure) {
                  throw new IllegalStateException(failure);
                }
              }
            });
    var fleet = ModelFleet.<StreamingChatModel>builder().model(candidate("a"), provider).build();
    routed(fleet, RoutingExecutionOptions.unlimited()).doChat(REQUEST, consumer);
    assertThat(error.get()).isNull();
    assertThat(forwarded).containsExactlyElementsOf(methods);
    assertThat(received).hasSize(sent.size());
    for (int i = 0; i < sent.size(); i++)
      for (int j = 0; j < sent.get(i).length; j++)
        assertThat(received.get(i)[j]).isSameAs(sent.get(i)[j]);
  }

  private static Object argument(Class<?> type) {
    try {
      if (type == String.class) return "fragment";
      if (type == Object.class) return new Object();
      if (type.getSimpleName().endsWith("Context")) {
        Class<?> handleType = Class.forName("dev.langchain4j.model.chat.response.StreamingHandle");
        Object handle =
            Proxy.newProxyInstance(
                handleType.getClassLoader(),
                new Class<?>[] {handleType},
                (p, m, a) -> m.getName().equals("isCancelled") ? false : null);
        return type.getConstructor(handleType).newInstance(handle);
      }
      if (type.getSimpleName().equals("CompleteToolCall")) {
        return type.getConstructor(int.class, dev.langchain4j.agent.tool.ToolExecutionRequest.class)
            .newInstance(
                0,
                dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                    .id("1")
                    .name("tool")
                    .arguments("{}")
                    .build());
      }
      if (type.getSimpleName().equals("PartialToolCall")) {
        Object builder = type.getMethod("builder").invoke(null);
        builder.getClass().getMethod("index", int.class).invoke(builder, 0);
        builder.getClass().getMethod("id", String.class).invoke(builder, "1");
        builder.getClass().getMethod("name", String.class).invoke(builder, "tool");
        builder.getClass().getMethod("partialArguments", String.class).invoke(builder, "{}");
        return builder.getClass().getMethod("build").invoke(builder);
      }
      return type.getConstructor(String.class).newInstance("fragment");
    } catch (ReflectiveOperationException failure) {
      throw new IllegalStateException(failure);
    }
  }

  @Test
  void cancellingOriginalProviderContextStopsLateErrorsAndFallback() throws Exception {
    Optional<Method> rich =
        Arrays.stream(StreamingChatResponseHandler.class.getMethods())
            .filter(
                method ->
                    method.getName().equals("onPartialResponse") && method.getParameterCount() == 2)
            .findFirst();
    // Older supported SDKs do not expose cancellation contexts; forwarding is tested above there.
    if (rich.isEmpty()) return;
    Method method = rich.orElseThrow();
    Class<?> handleType = Class.forName("dev.langchain4j.model.chat.response.StreamingHandle");
    AtomicBoolean cancelled = new AtomicBoolean();
    Object handle =
        Proxy.newProxyInstance(
            handleType.getClassLoader(),
            new Class<?>[] {handleType},
            (p, m, a) -> {
              if (m.getName().equals("cancel")) cancelled.set(true);
              return m.getName().equals("isCancelled") ? cancelled.get() : null;
            });
    Object context = method.getParameterTypes()[1].getConstructor(handleType).newInstance(handle);
    AtomicInteger fallback = new AtomicInteger(), terminals = new AtomicInteger();
    AtomicReference<Throwable> error = new AtomicReference<>();
    var fleet =
        ModelFleet.<StreamingChatModel>builder()
            .model(
                candidate("a"),
                model(
                    (request, callback) -> {
                      try {
                        method.invoke(callback, argument(method.getParameterTypes()[0]), context);
                      } catch (ReflectiveOperationException failure) {
                        throw new IllegalStateException(failure);
                      }
                      callback.onError(new IllegalStateException("late provider failure"));
                      callback.onCompleteResponse(
                          ChatResponse.builder().aiMessage(AiMessage.from("late")).build());
                    }))
            .model(candidate("b"), model((request, callback) -> fallback.incrementAndGet()))
            .build();
    StreamingChatResponseHandler consumer =
        (StreamingChatResponseHandler)
            Proxy.newProxyInstance(
                StreamingChatResponseHandler.class.getClassLoader(),
                new Class<?>[] {StreamingChatResponseHandler.class},
                (p, m, args) -> {
                  if (m.equals(method)) {
                    assertThat(args[1]).isSameAs(context);
                    handleType.getMethod("cancel").invoke(handle);
                  }
                  if (m.getName().equals("onError")) {
                    terminals.incrementAndGet();
                    error.set((Throwable) args[0]);
                  }
                  if (m.getName().equals("onCompleteResponse")) terminals.incrementAndGet();
                  return null;
                });
    routed(fleet, RoutingExecutionOptions.unlimited()).doChat(REQUEST, consumer);
    assertThat(cancelled).isTrue();
    assertThat(error.get()).isInstanceOf(CancellationException.class);
    assertThat(terminals).hasValue(1);
    assertThat(fallback).hasValue(0);
    assertThat(fleet.router().status("a").orElseThrow().consecutiveFailures()).isZero();
  }

  @Test
  void cancellationBeforeFirstTokenChargesReservationAndIgnoresLateCompletion() {
    RoutingBudget budget = new RoutingBudget(BigDecimal.ONE);
    RoutingCancellationToken token = new RoutingCancellationToken();
    AtomicReference<StreamingChatResponseHandler> provider = new AtomicReference<>();
    AtomicInteger fallback = new AtomicInteger(), completions = new AtomicInteger();
    AtomicReference<Throwable> error = new AtomicReference<>();
    var fleet =
        ModelFleet.<StreamingChatModel>builder()
            .model(candidate("a"), model((r, h) -> provider.set(h)))
            .model(candidate("b"), model((r, h) -> fallback.incrementAndGet()))
            .build();
    var options =
        RoutingExecutionOptions.builder()
            .sharedBudget(budget)
            .tokenBounds(new RoutingTokenBounds(100, 100))
            .cancellation(token)
            .build();
    routed(fleet, options).doChat(REQUEST, handler(error, completions));
    assertThat(budget.snapshot().reserved()).isEqualByComparingTo("0.2");
    token.cancel();
    provider
        .get()
        .onCompleteResponse(ChatResponse.builder().aiMessage(AiMessage.from("late")).build());
    provider.get().onError(new IllegalStateException("late"));
    assertThat(error.get()).isInstanceOf(CancellationException.class);
    assertThat(completions).hasValue(0);
    assertThat(fallback).hasValue(0);
    assertThat(budget.snapshot().spent()).isEqualByComparingTo("0.2");
    assertThat(budget.snapshot().reserved()).isEqualByComparingTo("0");
  }

  @Test
  void missingOutputCapFailsBeforeProvider() {
    AtomicInteger calls = new AtomicInteger();
    var fleet =
        ModelFleet.<StreamingChatModel>builder()
            .model(candidate("a"), model((r, h) -> calls.incrementAndGet()))
            .build();
    var options =
        RoutingExecutionOptions.builder()
            .requestBudget(BigDecimal.ONE)
            .tokenBounds(new RoutingTokenBounds(100, 100))
            .build();
    assertThatThrownBy(
            () ->
                routed(fleet, options)
                    .doChat(
                        ChatRequest.builder().messages(UserMessage.from("hello")).build(),
                        handler(new AtomicReference<>(), new AtomicInteger())))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(calls).hasValue(0);
  }

  @Test
  void interruptedEntryDoesNotRunFactories() {
    AtomicInteger factories = new AtomicInteger();
    var fleet =
        ModelFleet.<StreamingChatModel>builder()
            .model(
                candidate("a"),
                model(
                    (r, h) -> {
                      throw new AssertionError();
                    }))
            .build();
    var routed =
        new RoutedStreamingChatModel(
            fleet,
            r -> {
              factories.incrementAndGet();
              return RoutingRequest.builder("hi").build();
            });
    AtomicReference<Throwable> error = new AtomicReference<>();
    Thread.currentThread().interrupt();
    try {
      routed.doChat(REQUEST, handler(error, new AtomicInteger()));
      assertThat(Thread.currentThread().isInterrupted()).isTrue();
    } finally {
      Thread.interrupted();
    }
    assertThat(error.get()).isInstanceOf(CancellationException.class);
    assertThat(factories).hasValue(0);
  }

  @Test
  void accountsForHistorySystemToolsMediaAndReservedOutput() {
    var image = ImageContent.from("https://example.invalid/image.png");
    var request =
        ChatRequest.builder()
            .messages(
                SystemMessage.from("system"),
                UserMessage.from("history"),
                AiMessage.from("previous"),
                UserMessage.from(TextContent.from("latest"), image))
            .toolSpecifications(
                dev.langchain4j.agent.tool.ToolSpecification.builder()
                    .name("lookup")
                    .description("Search")
                    .build())
            .build();
    AtomicInteger mediaCalls = new AtomicInteger();
    List<String> counted = new ArrayList<>();
    var bounds =
        RoutingRequests.account(
            request,
            text -> {
              counted.add(text);
              return text.length();
            },
            media -> {
              assertThat(media).isSameAs(image);
              mediaCalls.incrementAndGet();
              return 500;
            },
            20,
            100);
    assertThat(counted).contains("system", "history", "previous", "latest");
    assertThat(counted.stream().anyMatch(text -> text.contains("lookup"))).isTrue();
    assertThat(mediaCalls).hasValue(1);
    assertThat(bounds.contextTokens()).isGreaterThan(620);
    assertThatThrownBy(() -> RoutingRequests.estimate(request))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("media");
  }

  @Test
  void concurrentTerminalEventsSettleAndNotifyOnlyOnce() throws Exception {
    RoutingBudget budget = new RoutingBudget(BigDecimal.ONE);
    AtomicReference<StreamingChatResponseHandler> callback = new AtomicReference<>();
    AtomicInteger terminals = new AtomicInteger();
    var fleet =
        ModelFleet.<StreamingChatModel>builder()
            .model(candidate("a"), model((r, h) -> callback.set(h)))
            .build();
    var controls =
        RoutingExecutionOptions.builder()
            .tokenBounds(new RoutingTokenBounds(100, 100))
            .sharedBudget(budget)
            .build();
    routed(fleet, controls)
        .doChat(
            REQUEST,
            new StreamingChatResponseHandler() {
              @Override
              public void onPartialResponse(String text) {}

              @Override
              public void onCompleteResponse(ChatResponse response) {
                terminals.incrementAndGet();
              }

              @Override
              public void onError(Throwable failure) {
                terminals.incrementAndGet();
              }
            });
    CountDownLatch start = new CountDownLatch(1);
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var completion =
          executor.submit(
              () -> {
                start.await();
                callback
                    .get()
                    .onCompleteResponse(
                        ChatResponse.builder()
                            .aiMessage(AiMessage.from("answer"))
                            .tokenUsage(new dev.langchain4j.model.output.TokenUsage(10, 20))
                            .build());
                return null;
              });
      var failure =
          executor.submit(
              () -> {
                start.await();
                callback.get().onError(new IllegalStateException("provider failed"));
                return null;
              });
      start.countDown();
      completion.get(3, TimeUnit.SECONDS);
      failure.get(3, TimeUnit.SECONDS);
    }
    assertThat(terminals).hasValue(1);
    assertThat(budget.snapshot().reserved()).isEqualByComparingTo("0");
    assertThat(budget.snapshot().spent().doubleValue()).isIn(0.03, 0.2);
  }
}
