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

import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Proxy;
import java.util.Objects;

/**
 * Delivers streamed fragments to a LangChain4j handler with cancellation where the API exists.
 *
 * <p>LangChain4j added {@code PartialResponse}, {@code PartialResponseContext} and {@code
 * StreamingHandle} after 1.0.0. The adapter supports every version in the framework-compat matrix,
 * so those types are never referenced at compile time: when they are present at runtime, each
 * fragment is delivered through {@code onPartialResponse(PartialResponse, PartialResponseContext)}
 * with a handle whose {@code cancel()} this class observes; otherwise fragments go through {@code
 * onPartialResponse(String)} and generation cannot be cancelled from the handler.
 */
final class LangChain4jPartialResponses {

  private static final Api API = Api.probe();

  private LangChain4jPartialResponses() {}

  /** Returns whether the running LangChain4j exposes a cancellable streaming handle. */
  static boolean cancellationSupported() {
    return API != null;
  }

  /** Creates the per-request delivery channel for {@code handler}. */
  static Channel open(StreamingChatResponseHandler handler) {
    Objects.requireNonNull(handler, "handler");
    return API == null ? new Channel(handler, null, null) : API.open(handler);
  }

  /** One streaming request's fragment delivery and cancellation state. */
  static final class Channel {
    private final StreamingChatResponseHandler handler;
    private final Object context;
    private final CancellationState state;

    private Channel(StreamingChatResponseHandler handler, Object context, CancellationState state) {
      this.handler = handler;
      this.context = context;
      this.state = state;
    }

    /** Delivers one fragment to the handler. */
    void emit(String token) {
      if (context == null) {
        handler.onPartialResponse(token);
        return;
      }
      try {
        API.onPartialResponse.invoke(handler, API.partialResponse.invoke(token), context);
      } catch (RuntimeException | Error failure) {
        throw failure;
      } catch (Throwable failure) {
        throw new IllegalStateException("cannot deliver LangChain4j partial response", failure);
      }
    }

    /** Returns whether the handler cancelled through its streaming handle. */
    boolean isCancelled() {
      return state != null && state.cancelled;
    }
  }

  private static final class CancellationState {
    private volatile boolean cancelled;
  }

  private record Api(
      Class<?> handleType,
      MethodHandle partialResponse,
      MethodHandle partialResponseContext,
      MethodHandle onPartialResponse) {

    private static Api probe() {
      ClassLoader loader = StreamingChatResponseHandler.class.getClassLoader();
      String base = "dev.langchain4j.model.chat.response.";
      try {
        Class<?> responseType = Class.forName(base + "PartialResponse", false, loader);
        Class<?> contextType = Class.forName(base + "PartialResponseContext", false, loader);
        Class<?> handleType = Class.forName(base + "StreamingHandle", false, loader);
        MethodHandles.Lookup lookup = MethodHandles.publicLookup();
        return new Api(
            handleType,
            lookup.findConstructor(responseType, MethodType.methodType(void.class, String.class)),
            lookup.findConstructor(contextType, MethodType.methodType(void.class, handleType)),
            lookup.findVirtual(
                StreamingChatResponseHandler.class,
                "onPartialResponse",
                MethodType.methodType(void.class, responseType, contextType)));
      } catch (ClassNotFoundException | NoSuchMethodException | IllegalAccessException absent) {
        return null;
      }
    }

    private Channel open(StreamingChatResponseHandler handler) {
      CancellationState state = new CancellationState();
      Object handle =
          Proxy.newProxyInstance(
              handleType.getClassLoader(),
              new Class<?>[] {handleType},
              (proxy, method, args) ->
                  switch (method.getName()) {
                    case "cancel" -> {
                      state.cancelled = true;
                      yield null;
                    }
                    case "isCancelled" -> state.cancelled;
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    case "toString" -> "ModelsStreamingHandle[cancelled=" + state.cancelled + "]";
                    default -> throw new UnsupportedOperationException(method.getName());
                  });
      try {
        return new Channel(handler, partialResponseContext.invoke(handle), state);
      } catch (RuntimeException | Error failure) {
        throw failure;
      } catch (Throwable failure) {
        throw new IllegalStateException("cannot create LangChain4j streaming context", failure);
      }
    }
  }
}
