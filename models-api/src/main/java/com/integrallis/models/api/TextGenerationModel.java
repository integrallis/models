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
import java.util.function.Consumer;

/**
 * High-level text generation contract shared by in-process and local-engine backends.
 *
 * <p>Implementations may own mutable inference state and are not required to be thread-safe.
 * Callers must serialize generation unless an implementation explicitly documents concurrent use.
 * Prompts, options, token streams, model names, and diagnostics must be non-null. A streaming
 * generation invokes its terminal callback before returning, which allows the default blocking
 * methods to collect the result without a second synchronization contract.
 */
public interface TextGenerationModel extends AutoCloseable {

  /** Human-readable model identifier reported to framework adapters. */
  String modelName();

  /** Returns the backend execution plan and environment. */
  BackendDiagnostics diagnostics();

  /** Generates text and returns it after the stream completes. */
  default String generate(String prompt, SamplingOptions options) {
    Objects.requireNonNull(prompt, "prompt");
    Objects.requireNonNull(options, "options");
    return collect(stream -> generate(prompt, options, stream));
  }

  /**
   * Generates text from a prompt that separates template controls from ordinary message text.
   *
   * <p>Implementations without segmented-tokenization support receive the complete prompt text.
   */
  default String generate(ModelPrompt prompt, SamplingOptions options) {
    Objects.requireNonNull(prompt, "prompt");
    Objects.requireNonNull(options, "options");
    return collect(stream -> generate(prompt, options, stream));
  }

  /**
   * Generates text incrementally through the supplied stream callback.
   *
   * <p>The implementation follows the ordering and terminal-signal contract defined by {@link
   * TokenStream}. It must report an inference failure to {@code stream.onError} rather than also
   * throwing that same failure.
   */
  void generate(String prompt, SamplingOptions options, TokenStream stream);

  /**
   * Generates text incrementally from a segmented prompt.
   *
   * <p>The default supports engines that accept text rather than tokenizer segments.
   */
  default void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
    Objects.requireNonNull(prompt, "prompt");
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(stream, "stream");
    generate(prompt.text(), options, stream);
  }

  /** Releases backend-owned resources. */
  @Override
  default void close() {}

  private static String collect(Consumer<TokenStream> generation) {
    StringBuilder output = new StringBuilder();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    generation.accept(
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
    Throwable generationFailure = failure.get();
    if (generationFailure instanceof RuntimeException runtimeFailure) {
      throw runtimeFailure;
    }
    if (generationFailure instanceof Error error) {
      throw error;
    }
    if (generationFailure != null) {
      throw new ModelGenerationException("text generation failed", generationFailure);
    }
    return output.toString();
  }
}
