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
package com.integrallis.models.api;

/**
 * Push callback for one generation.
 *
 * <p>The producer invokes callbacks serially and in token order. It invokes exactly one terminal
 * callback, either {@link #onComplete()}, {@link #onComplete(GenerationUsage)}, {@link
 * #onComplete(GenerationUsage, StopReason)}, or {@link #onError(Throwable)}, and invokes nothing
 * afterward. The callback thread is implementation-specific; consumers should return promptly and
 * must not reenter the same non-thread-safe model.
 */
public interface TokenStream {

  /** Called with a non-null decoded token fragment, which may be empty. */
  void onToken(String token);

  /** Called when generation is complete. */
  void onComplete();

  /**
   * Called when generation is complete with exact token counts.
   *
   * <p>The default preserves compatibility with consumers that only need the original terminal
   * callback. Producers that can measure usage should invoke this method instead of {@link
   * #onComplete()}.
   */
  default void onComplete(GenerationUsage usage) {
    onComplete();
  }

  /**
   * Called when generation is complete with exact token counts and the reason it ended.
   *
   * <p>The default forwards to {@link #onComplete(GenerationUsage)}, so consumers written before
   * stop reasons existed keep receiving usage. Producers that know why generation ended should
   * invoke this method; this is the only completion callback they invoke.
   *
   * @param usage exact token counts for the completed generation
   * @param stopReason why generation ended
   */
  default void onComplete(GenerationUsage usage, StopReason stopReason) {
    onComplete(usage);
  }

  /**
   * Returns whether the consumer wants generation to stop.
   *
   * <p>Producers that support cancellation poll this after each emitted token and then finish
   * normally with {@link StopReason#CANCELLED}. The default never cancels. Implementations must be
   * cheap and safe to call from the producer's thread.
   *
   * @return {@code true} to request that generation stop
   */
  default boolean isCancelled() {
    return false;
  }

  /** Called with the non-null failure that ended generation. */
  void onError(Throwable failure);
}
