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

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/** High-level text generation contract shared by in-process and local-engine backends. */
public interface TextGenerationModel extends AutoCloseable {

  /** Human-readable model identifier reported to framework adapters. */
  String modelName();

  /** Returns the backend execution plan and environment. */
  BackendDiagnostics diagnostics();

  /** Generates text and returns it after the stream completes. */
  default String generate(String prompt, SamplingOptions options) {
    Objects.requireNonNull(prompt, "prompt");
    Objects.requireNonNull(options, "options");
    StringBuilder output = new StringBuilder();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    generate(
        prompt,
        options,
        new TokenStream() {
          @Override
          public void onToken(String token) {
            output.append(token);
          }

          @Override
          public void onComplete() {}

          @Override
          public void onError(Throwable throwable) {
            failure.compareAndSet(null, throwable);
          }
        });
    if (failure.get() != null) {
      throw new IllegalStateException("text generation failed", failure.get());
    }
    return output.toString();
  }

  /** Generates text incrementally through the supplied stream callback. */
  void generate(String prompt, SamplingOptions options, TokenStream stream);

  /** Releases backend-owned resources. */
  @Override
  default void close() {}
}
