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

import com.integrallis.models.router.ModelCandidate;
import com.integrallis.models.router.ModelFleet;
import com.integrallis.models.router.RoutingBudgetExceededException;
import com.integrallis.models.router.RoutingCancellation;
import com.integrallis.models.router.RoutingContinuity;
import com.integrallis.models.router.RoutingDecision;
import com.integrallis.models.router.RoutingExecution;
import com.integrallis.models.router.RoutingExecutionOptions;
import com.integrallis.models.router.RoutingFeedback;
import com.integrallis.models.router.RoutingRequest;
import com.integrallis.models.router.RoutingRequirements;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

/** LangChain4j streaming model that routes unchanged requests across local and hosted clients. */
public final class RoutedStreamingChatModel implements StreamingChatModel {
  private final ModelFleet<StreamingChatModel> fleet;
  private final Function<ChatRequest, RoutingRequest> requestFactory;
  private final Function<ChatRequest, RoutingContinuity> continuityFactory;
  private final Function<ChatRequest, RoutingRequirements> requirementsFactory;
  private final Function<ChatRequest, RoutingExecutionOptions> executionFactory;

  /** Routes from the latest user message, without implicit session affinity. */
  public RoutedStreamingChatModel(ModelFleet<StreamingChatModel> fleet) {
    this(
        fleet,
        RoutedStreamingChatModel::defaultRequest,
        RoutedStreamingChatModel::defaultContinuity);
  }

  /** Routes with an application-supplied mapping for token estimates, task, and session id. */
  public RoutedStreamingChatModel(
      ModelFleet<StreamingChatModel> fleet, Function<ChatRequest, RoutingRequest> requestFactory) {
    this(fleet, requestFactory, RoutedStreamingChatModel::defaultContinuity);
  }

  /** Routes with application-supplied request and continuity extraction. */
  public RoutedStreamingChatModel(
      ModelFleet<StreamingChatModel> fleet,
      Function<ChatRequest, RoutingRequest> requestFactory,
      Function<ChatRequest, RoutingContinuity> continuityFactory) {
    this(fleet, requestFactory, continuityFactory, ignored -> RoutingRequirements.none());
  }

  /**
   * Routes with requirements extracted from the complete original request, including its history.
   *
   * <p>Requirements constrain the selected model and every fallback. The application supplies
   * capability and data-boundary policy; the adapter does not infer sensitive data. A null
   * requirement result fails before any client is invoked.
   */
  public RoutedStreamingChatModel(
      ModelFleet<StreamingChatModel> fleet,
      Function<ChatRequest, RoutingRequest> requestFactory,
      Function<ChatRequest, RoutingContinuity> continuityFactory,
      Function<ChatRequest, RoutingRequirements> requirementsFactory) {
    this(
        fleet,
        requestFactory,
        continuityFactory,
        requirementsFactory,
        ignored -> RoutingExecutionOptions.unlimited());
  }

  /** Routes with complete token bounds, shared spending controls and an end-to-end deadline. */
  public RoutedStreamingChatModel(
      ModelFleet<StreamingChatModel> fleet,
      Function<ChatRequest, RoutingRequest> requestFactory,
      Function<ChatRequest, RoutingContinuity> continuityFactory,
      Function<ChatRequest, RoutingRequirements> requirementsFactory,
      Function<ChatRequest, RoutingExecutionOptions> executionFactory) {
    this.executionFactory = Objects.requireNonNull(executionFactory, "executionFactory");
    this.fleet = Objects.requireNonNull(fleet, "fleet");
    this.requestFactory = Objects.requireNonNull(requestFactory, "requestFactory");
    this.continuityFactory = Objects.requireNonNull(continuityFactory, "continuityFactory");
    this.requirementsFactory = Objects.requireNonNull(requirementsFactory, "requirementsFactory");
  }

  @Override
  public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(handler, "handler");
    if (Thread.currentThread().isInterrupted()) {
      handler.onError(new CancellationException("routed stream interrupted"));
      return;
    }
    RoutingExecutionOptions options =
        Objects.requireNonNull(executionFactory.apply(request), "routing execution options");
    RoutingAccounting.validate(request, options);
    RoutingExecution execution = new RoutingExecution(options);
    Stream stream = new Stream(request, handler, execution);
    execution.onCancel(stream::cancel);
    try {
      execution.checkActive();
      stream.routingRequest =
          Objects.requireNonNull(requestFactory.apply(request), "routing request");
      RoutingDecision decision =
          execution.call(
              () ->
                  fleet.decide(
                      stream.routingRequest,
                      Objects.requireNonNull(
                          requirementsFactory.apply(request), "routing requirements"),
                      Objects.requireNonNull(
                          continuityFactory.apply(request), "routing continuity"),
                      execution));
      stream.taskType = decision.taskType();
      stream.order.add(decision.selected());
      stream.order.addAll(decision.fallbacks());
      stream.start(0);
    } catch (RuntimeException | Error failure) {
      execution.close();
      if (RoutingCancellation.isCancellation(failure)) stream.terminalError(failure);
      else throw failure;
    } catch (Exception failure) {
      execution.close();
      throw new IllegalStateException(failure);
    }
  }

  private final class Stream {
    private final ChatRequest request;
    private final StreamingChatResponseHandler handler;
    private final RoutingExecution execution;
    private final List<ModelCandidate> order = new ArrayList<>();
    private final Object delivery = new Object();
    private final List<ProviderHandle> handles = new ArrayList<>();
    private boolean terminal;
    private RoutingRequest routingRequest;
    private String taskType;

    Stream(ChatRequest request, StreamingChatResponseHandler handler, RoutingExecution execution) {
      this.request = request;
      this.handler = handler;
      this.execution = execution;
    }

    void cancel() {
      List<ProviderHandle> copy;
      synchronized (handles) {
        copy = List.copyOf(handles);
      }
      for (ProviderHandle handle : copy)
        Thread.ofVirtual().name("models-provider-cancel").start(handle::cancel);
      terminalError(new CancellationException("routed stream cancelled or deadline exceeded"));
    }

    void terminalError(Throwable failure) {
      synchronized (delivery) {
        if (terminal) return;
        terminal = true;
        execution.close();
        handler.onError(failure);
      }
    }

    void checkHandles() {
      List<ProviderHandle> copy;
      synchronized (handles) {
        copy = List.copyOf(handles);
      }
      for (ProviderHandle handle : copy) {
        if (handle.isCancelled()) {
          execution.cancel();
          break;
        }
      }
      execution.checkActive();
    }

    void observeContexts(Object[] args) throws ReflectiveOperationException {
      if (args == null) return;
      for (Object arg : args) {
        if (arg == null
            || !arg.getClass().getName().startsWith("dev.langchain4j.model.chat.response."))
          continue;
        java.lang.reflect.Method accessor;
        try {
          accessor = arg.getClass().getMethod("streamingHandle");
        } catch (NoSuchMethodException absent) {
          continue;
        }
        Object value = accessor.invoke(arg);
        if (value == null) continue;
        synchronized (handles) {
          if (handles.stream().noneMatch(handle -> handle.value == value)) {
            handles.add(new ProviderHandle(value, accessor.getReturnType()));
          }
        }
      }
    }

    void start(int index) {
      try {
        checkHandles();
      } catch (CancellationException cancelled) {
        terminalError(cancelled);
        return;
      }
      ModelCandidate candidate = order.get(index);
      RoutingExecution.Attempt reservation;
      try {
        reservation = execution.beginAttempt(candidate);
      } catch (RoutingBudgetExceededException denied) {
        if (index + 1 < order.size()) start(index + 1);
        else terminalError(denied);
        return;
      }
      AtomicBoolean emitted = new AtomicBoolean();
      AtomicBoolean finished = new AtomicBoolean();
      AtomicLong firstToken = new AtomicLong(-1);
      long started = System.nanoTime();
      java.util.function.Consumer<Throwable> failed =
          failure -> {
            boolean retry;
            synchronized (delivery) {
              if (terminal || !finished.compareAndSet(false, true)) return;
              reservation.close();
              try {
                checkHandles();
              } catch (CancellationException cancelled) {
                terminalError(cancelled);
                return;
              }
              if (RoutingCancellation.isCancellation(failure)
                  || failure instanceof RoutingBudgetExceededException) {
                terminalError(failure);
                return;
              }
              record(routingRequest, taskType, candidate.id(), false, -1, started);
              retry = !emitted.get() && index + 1 < order.size();
              if (!retry) terminalError(failure);
            }
            // Do not hold the delivery lock while a provider starts: deadlines must still deliver.
            if (retry) start(index + 1);
          };
      // Forward the runtime interface, including callbacks absent from older supported SDKs.
      StreamingChatResponseHandler forwarding =
          (StreamingChatResponseHandler)
              Proxy.newProxyInstance(
                  StreamingChatResponseHandler.class.getClassLoader(),
                  new Class<?>[] {StreamingChatResponseHandler.class},
                  (proxy, method, args) -> {
                    if (method.getDeclaringClass() == Object.class) {
                      return switch (method.getName()) {
                        case "toString" -> "RoutedStreamingChatResponseHandler";
                        case "hashCode" -> System.identityHashCode(proxy);
                        case "equals" -> proxy == args[0];
                        default -> throw new UnsupportedOperationException(method.getName());
                      };
                    }
                    if (method.getName().equals("onError")) {
                      failed.accept((Throwable) args[0]);
                      return null;
                    }
                    synchronized (delivery) {
                      if (terminal || finished.get()) return null;
                      try {
                        observeContexts(args);
                        checkHandles();
                        if (method.getName().equals("onCompleteResponse")) {
                          finished.set(true);
                          reservation.settle(RoutingAccounting.usage((ChatResponse) args[0]));
                          execution.checkActive();
                          terminal = true;
                          execution.close();
                          record(
                              routingRequest,
                              taskType,
                              candidate.id(),
                              true,
                              firstToken.get(),
                              started);
                        } else {
                          // Raw events can contain non-portable provider state too.
                          emitted.set(true);
                          if (!method.getName().equals("onUnmappedRawEvent"))
                            firstToken.compareAndSet(-1, System.nanoTime());
                        }
                        try {
                          method.invoke(handler, args);
                        } catch (InvocationTargetException failure) {
                          // Application callbacks are not provider failures and must never trigger
                          // fallback.
                          execution.cancel();
                          throw failure.getCause();
                        }
                        if (!terminal) checkHandles();
                      } catch (CancellationException | RoutingBudgetExceededException control) {
                        finished.set(true);
                        reservation.close();
                        terminalError(control);
                      }
                    }
                    return null;
                  });
      try {
        execution.call(
            () -> {
              execution.checkActive();
              fleet.model(candidate.id()).client().chat(request, forwarding);
              return null;
            });
      } catch (Exception | Error failure) {
        failed.accept(failure);
      }
    }
  }

  private static final class ProviderHandle {
    private final Object value;
    private final java.lang.reflect.Method cancel;
    private final java.lang.reflect.Method isCancelled;

    ProviderHandle(Object value, Class<?> type) throws NoSuchMethodException {
      this.value = value;
      cancel = type.getMethod("cancel");
      isCancelled = type.getMethod("isCancelled");
    }

    void cancel() {
      try {
        cancel.invoke(value);
      } catch (ReflectiveOperationException ignored) {
        /* best effort */
      }
    }

    boolean isCancelled() {
      try {
        return Boolean.TRUE.equals(isCancelled.invoke(value));
      } catch (ReflectiveOperationException failure) {
        throw new IllegalStateException("cannot inspect provider cancellation", failure);
      }
    }
  }

  private void record(
      RoutingRequest request,
      String taskType,
      String modelId,
      boolean success,
      long firstTokenNanos,
      long started) {
    RoutingFeedback.Builder feedback =
        success ? RoutingFeedback.success(modelId) : RoutingFeedback.failure(modelId);
    request.session().ifPresent(feedback::sessionId);
    if (taskType != null) {
      feedback.taskType(taskType);
    }
    if (success && firstTokenNanos >= started) {
      feedback.timeToFirstTokenMillis((firstTokenNanos - started) / 1_000_000);
    }
    fleet.router().record(feedback.build());
  }

  private static RoutingRequest defaultRequest(ChatRequest request) {
    return RoutingRequests.estimate(request);
  }

  private static RoutingContinuity defaultContinuity(ChatRequest request) {
    List<ChatMessage> messages = request.messages();
    boolean toolLoop =
        !messages.isEmpty() && messages.getLast() instanceof ToolExecutionResultMessage;
    return RoutingContinuity.builder().activeToolLoop(toolLoop).build();
  }
}
