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

/** High-level contract for in-process speech synthesis models. */
public interface TextToSpeechModel extends AutoCloseable {

  /** Human-readable model identifier reported to framework adapters. */
  String modelName();

  /** Returns the backend execution plan and environment. */
  BackendDiagnostics diagnostics();

  /** Synthesizes speech with model-specific defaults. */
  default PcmAudio synthesize(String text) {
    return synthesize(text, SpeechSynthesisOptions.builder().build());
  }

  /** Synthesizes speech as normalized floating-point PCM. */
  PcmAudio synthesize(String text, SpeechSynthesisOptions options);

  /**
   * Streams speech through a callback. Implementations may override this to emit chunks as they
   * become available; the default adapts the blocking operation into one chunk.
   */
  default void synthesize(String text, SpeechSynthesisOptions options, AudioStream stream) {
    Objects.requireNonNull(text, "text");
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(stream, "stream");
    try {
      stream.onAudio(synthesize(text, options));
    } catch (RuntimeException | Error failure) {
      stream.onError(failure);
      return;
    }
    // Completion is already the terminal signal. If the consumer throws from it, propagate that
    // failure instead of violating the callback contract by following it with onError.
    stream.onComplete();
  }

  /** Releases model-owned resources. */
  @Override
  default void close() {}
}
