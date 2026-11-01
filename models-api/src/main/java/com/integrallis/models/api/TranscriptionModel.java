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

/**
 * High-level contract for in-process speech recognition models.
 *
 * <p>Mirrors {@link TextToSpeechModel} in shape on purpose — identity, diagnostics, one blocking
 * operation, one streaming default, {@code close} — because that interface is already served by a
 * shipped model and adapters know it. A transcription model is not a decoder: a {@code parakeet}
 * artifact carries <b>zero</b> {@code blk.*} tensors and its roots are {@code
 * preprocessor/encoder/decoder/joint}, so none of {@code DecoderArchitecture}, {@code LlamaConfig}
 * or the execution planner applies to it. That is why this is a separate contract rather than a
 * capability on an existing one.
 *
 * <p>Takes {@link PcmAudio} rather than a file path or an encoded buffer. Decoding containers and
 * resampling belong to a caller or a codec component, not to a model: a model that accepted MP3
 * would own an audio-decoding dependency on the inference path, and a model that resampled silently
 * would make its own word error rate depend on a resampler nobody declared.
 */
public interface TranscriptionModel extends AutoCloseable {

  /** Human-readable model identifier reported to framework adapters. */
  String modelName();

  /** Returns the backend execution plan and environment. */
  BackendDiagnostics diagnostics();

  /**
   * The sample rate this model requires, in hertz.
   *
   * <p>Exposed rather than assumed. {@code parakeet} declares {@code preprocessor.sample_rate =
   * 16000}, and feeding 44.1 kHz audio to a 16 kHz mel front end produces a transcript that is
   * wrong in a way no exception reports — so a caller can resample before calling, and an adapter
   * can refuse mismatched audio instead of transcribing noise.
   */
  int requiredSampleRate();

  /** Transcribes with model-specific defaults. */
  default Transcript transcribe(PcmAudio audio) {
    return transcribe(audio, TranscriptionOptions.builder().build());
  }

  /** Transcribes normalized floating-point PCM at {@link #requiredSampleRate()}. */
  Transcript transcribe(PcmAudio audio, TranscriptionOptions options);

  /**
   * Streams a transcript through a callback. Implementations may override this to emit partial
   * results as the encoder advances; the default adapts the blocking operation into one result.
   */
  default void transcribe(PcmAudio audio, TranscriptionOptions options, TranscriptStream stream) {
    Objects.requireNonNull(audio, "audio");
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(stream, "stream");
    try {
      stream.onTranscript(transcribe(audio, options));
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
