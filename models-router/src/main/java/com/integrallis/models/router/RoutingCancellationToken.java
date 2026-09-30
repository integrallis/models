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
package com.integrallis.models.router;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Explicit request cancellation, including before the first provider callback. */
public final class RoutingCancellationToken {
  private boolean cancelled;
  private final List<Runnable> listeners = new ArrayList<>();

  public synchronized boolean isCancelled() {
    return cancelled;
  }

  public void cancel() {
    List<Runnable> callbacks;
    synchronized (this) {
      if (cancelled) return;
      cancelled = true;
      callbacks = List.copyOf(listeners);
      listeners.clear();
    }
    callbacks.forEach(RoutingCancellationToken::notifyListener);
  }

  public AutoCloseable onCancel(Runnable listener) {
    Objects.requireNonNull(listener, "listener");
    synchronized (this) {
      if (!cancelled) {
        listeners.add(listener);
        return () -> {
          synchronized (this) {
            listeners.remove(listener);
          }
        };
      }
    }
    notifyListener(listener);
    return () -> {};
  }

  private static void notifyListener(Runnable listener) {
    try {
      listener.run();
    } catch (RuntimeException ignored) {
      // One provider's cancellation hook must not prevent the remaining hooks from running.
    }
  }
}
