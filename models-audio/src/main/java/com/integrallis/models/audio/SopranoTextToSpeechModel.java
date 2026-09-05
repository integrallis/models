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
package com.integrallis.models.audio;

import com.integrallis.models.api.AudioStream;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.PcmAudio;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SpeechSynthesisOptions;
import com.integrallis.models.api.TextToSpeechModel;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.runtime.Sampler;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/** Java-native Soprano 1.1 text-to-speech runtime. */
public final class SopranoTextToSpeechModel implements TextToSpeechModel {

  private static final int DEFAULT_MAX_TOKENS = 512;
  private static final int DEFAULT_TEXT_CHUNK_CODEPOINTS = 200;
  private static final int DECODER_RECEPTIVE_FIELD = 4;
  private static final int STREAM_CHUNK_TOKENS = 12;
  private static final int MAX_NATIVE_THREADS = 6;

  private final SopranoEngine engine;
  private boolean closed;

  SopranoTextToSpeechModel(SopranoEngine engine) {
    this.engine = Objects.requireNonNull(engine, "engine");
  }

  /** Opens a standalone Soprano GGUF artifact without an external inference process. */
  public static SopranoTextToSpeechModel load(Path modelPath) throws IOException {
    return new SopranoTextToSpeechModel(SopranoBackendEngine.load(modelPath));
  }

  /** Opens Soprano with the integrity-checked Models native projection kernels. */
  public static SopranoTextToSpeechModel loadNative(Path modelPath) throws IOException {
    return new SopranoTextToSpeechModel(
        SopranoBackendEngine.load(
            modelPath,
            RustGgufBatchedMatrixKernel.openBundled(
                nativeThreadRecommendation(Runtime.getRuntime().availableProcessors()))));
  }

  /** Opens Soprano with an explicit Models native projection library. */
  public static SopranoTextToSpeechModel loadNative(Path modelPath, Path nativeLibrary)
      throws IOException {
    Objects.requireNonNull(nativeLibrary, "nativeLibrary");
    return new SopranoTextToSpeechModel(
        SopranoBackendEngine.load(
            modelPath,
            RustGgufBatchedMatrixKernel.open(
                nativeLibrary,
                nativeThreadRecommendation(Runtime.getRuntime().availableProcessors()))));
  }

  static int nativeThreadRecommendation(int availableProcessors) {
    return Math.max(1, Math.min(MAX_NATIVE_THREADS, availableProcessors));
  }

  @Override
  public String modelName() {
    return engine.modelName();
  }

  @Override
  public BackendDiagnostics diagnostics() {
    return engine.diagnostics();
  }

  @Override
  public synchronized PcmAudio synthesize(String text, SpeechSynthesisOptions options) {
    requireOpen();
    Objects.requireNonNull(options, "options");
    validateControls(options);

    List<String> chunks = textChunks(text);
    List<float[]> samples = new ArrayList<>(chunks.size());
    int sampleCount = 0;
    for (String chunk : chunks) {
      PcmAudio audio = synthesizeChunk(chunk, options);
      float[] chunkSamples = audio.samples();
      samples.add(chunkSamples);
      sampleCount = Math.addExact(sampleCount, chunkSamples.length);
    }

    float[] merged = new float[sampleCount];
    int offset = 0;
    for (float[] chunk : samples) {
      System.arraycopy(chunk, 0, merged, offset, chunk.length);
      offset += chunk.length;
    }
    return new PcmAudio(engine.sampleRate(), 1, merged);
  }

  /** Emits one PCM chunk per completed reference-sized text segment. */
  @Override
  public synchronized void synthesize(
      String text, SpeechSynthesisOptions options, AudioStream stream) {
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(stream, "stream");
    try {
      requireOpen();
      validateControls(options);
      for (String chunk : textChunks(text)) {
        synthesizeChunkStreaming(chunk, options, stream);
      }
    } catch (RuntimeException | Error failure) {
      stream.onError(failure);
      return;
    }
    stream.onComplete();
  }

  private void synthesizeChunkStreaming(
      String text, SpeechSynthesisOptions options, AudioStream stream) {
    int[] prompt = engine.encodePrompt(text);
    SopranoEngine.Step current = engine.begin(prompt);
    SamplingOptions sampling =
        options.sampling() != null ? options.sampling() : defaultSamplingOptions();
    int maxTokens = Math.min(sampling.maxTokens(), engine.contextLength() - engine.checkpoint());
    if (maxTokens <= 0) {
      throw new IllegalArgumentException("Soprano prompt leaves no context for speech frames");
    }

    Sampler sampler = new Sampler(sampling);
    List<Integer> history = new ArrayList<>(prompt.length + maxTokens);
    for (int token : prompt) {
      history.add(token);
    }
    Deque<float[]> featureWindow =
        new ArrayDeque<>(2 * DECODER_RECEPTIVE_FIELD + STREAM_CHUNK_TOKENS);
    int generated = 0;
    int emittedStableFrames = 0;
    for (int step = 0; step < maxTokens; step++) {
      int token = sampler.sample(current.logits(), history);
      history.add(token);
      if (token == engine.eosToken()) {
        break;
      }

      featureWindow.addLast(requireFrame(current.hiddenState()));
      generated++;
      while (featureWindow.size() > 2 * DECODER_RECEPTIVE_FIELD + STREAM_CHUNK_TOKENS) {
        featureWindow.removeFirst();
      }
      int stableFrames = generated - DECODER_RECEPTIVE_FIELD;
      if (stableFrames - emittedStableFrames >= STREAM_CHUNK_TOKENS) {
        emitStreamingCenter(featureWindow, STREAM_CHUNK_TOKENS, stream);
        emittedStableFrames += STREAM_CHUNK_TOKENS;
      }
      if (step + 1 < maxTokens) {
        current = engine.advance(token);
      }
    }
    if (generated == 0) {
      throw new IllegalStateException("Soprano produced no audio frames");
    }
    emitStreamingFinal(
        featureWindow,
        Math.max(0, generated - DECODER_RECEPTIVE_FIELD - emittedStableFrames),
        stream);
  }

  private void emitStreamingCenter(
      Deque<float[]> featureWindow, int stableFrames, AudioStream stream) {
    float[] audio = decodeWindow(featureWindow);
    int samplesPerToken = engine.samplesPerToken();
    int start =
        Math.max(0, audio.length - (DECODER_RECEPTIVE_FIELD + stableFrames - 1) * samplesPerToken);
    int end = Math.max(start, audio.length - (DECODER_RECEPTIVE_FIELD - 1) * samplesPerToken);
    emit(audio, start, end, stream);
  }

  private void emitStreamingFinal(
      Deque<float[]> featureWindow, int remainingStableFrames, AudioStream stream) {
    float[] audio = decodeWindow(featureWindow);
    if (remainingStableFrames > 0) {
      int samplesPerToken = engine.samplesPerToken();
      int start =
          Math.max(
              0,
              audio.length
                  - (DECODER_RECEPTIVE_FIELD + remainingStableFrames - 1) * samplesPerToken);
      int end = Math.max(start, audio.length - (DECODER_RECEPTIVE_FIELD - 1) * samplesPerToken);
      emit(audio, start, end, stream);
    }
    int tailSamples = (DECODER_RECEPTIVE_FIELD - 1) * engine.samplesPerToken();
    emit(audio, Math.max(0, audio.length - tailSamples), audio.length, stream);
  }

  private float[] decodeWindow(Deque<float[]> featureWindow) {
    float[] features = new float[Math.multiplyExact(featureWindow.size(), engine.hiddenSize())];
    int offset = 0;
    for (float[] frame : featureWindow) {
      System.arraycopy(frame, 0, features, offset, frame.length);
      offset += frame.length;
    }
    return engine.decode(features, featureWindow.size());
  }

  private void emit(float[] audio, int start, int end, AudioStream stream) {
    if (end > start) {
      stream.onAudio(new PcmAudio(engine.sampleRate(), 1, Arrays.copyOfRange(audio, start, end)));
    }
  }

  private PcmAudio synthesizeChunk(String text, SpeechSynthesisOptions options) {

    int[] prompt = engine.encodePrompt(text);
    SopranoEngine.Step current = engine.begin(prompt);
    SamplingOptions sampling =
        options.sampling() != null ? options.sampling() : defaultSamplingOptions();
    int maxTokens = Math.min(sampling.maxTokens(), engine.contextLength() - engine.checkpoint());
    if (maxTokens <= 0) {
      throw new IllegalArgumentException("Soprano prompt leaves no context for speech frames");
    }

    Sampler sampler = new Sampler(sampling);
    List<Integer> history = new ArrayList<>(prompt.length + maxTokens);
    for (int token : prompt) {
      history.add(token);
    }
    float[] features = new float[Math.multiplyExact(maxTokens, engine.hiddenSize())];
    int frames = 0;
    int generated = 0;
    for (int step = 0; step < maxTokens; step++) {
      int token = sampler.sample(current.logits(), history);
      history.add(token);
      if (token == engine.eosToken()) {
        break;
      }
      frames = appendFrame(features, frames, current.hiddenState(), engine.hiddenSize());
      generated++;
      if (step + 1 < maxTokens) {
        current = engine.advance(token);
      }
    }
    if (generated == 0) {
      throw new IllegalStateException("Soprano produced no audio frames");
    }

    float[] exactFeatures = Arrays.copyOf(features, frames * engine.hiddenSize());
    return new PcmAudio(engine.sampleRate(), 1, engine.decode(exactFeatures, frames));
  }

  private static List<String> textChunks(String text) {
    String normalized = SopranoTextNormalizer.normalize(text);
    List<String> chunks = SopranoTextChunker.split(normalized, DEFAULT_TEXT_CHUNK_CODEPOINTS);
    if (chunks.isEmpty()) {
      throw new IllegalArgumentException("Soprano text must not be blank");
    }
    return chunks;
  }

  @Override
  public synchronized void close() {
    if (!closed) {
      closed = true;
      engine.close();
    }
  }

  private SamplingOptions defaultSamplingOptions() {
    return SamplingOptions.builder()
        .temperature(0.3f)
        .topP(0.95f)
        .topK(engine.vocabularySize())
        .maxTokens(DEFAULT_MAX_TOKENS)
        .repetitionPenalty(1.2f)
        .build();
  }

  private static int appendFrame(
      float[] destination, int frame, float[] hidden, int expectedHiddenSize) {
    if (hidden.length != expectedHiddenSize) {
      throw new IllegalStateException(
          "Soprano acoustic frame hidden size is "
              + hidden.length
              + "; expected "
              + expectedHiddenSize);
    }
    int offset = Math.multiplyExact(frame, expectedHiddenSize);
    System.arraycopy(hidden, 0, destination, offset, hidden.length);
    return frame + 1;
  }

  private float[] requireFrame(float[] hidden) {
    if (hidden.length != engine.hiddenSize()) {
      throw new IllegalStateException(
          "Soprano acoustic frame hidden size is "
              + hidden.length
              + "; expected "
              + engine.hiddenSize());
    }
    return hidden.clone();
  }

  private static void validateControls(SpeechSynthesisOptions options) {
    if (options.language() != null
        && !"en".equals(options.language().toLowerCase(Locale.ROOT))
        && !"english".equals(options.language().toLowerCase(Locale.ROOT))) {
      throw new IllegalArgumentException("Soprano is an English-only model");
    }
    if (options.voice() != null) {
      throw new IllegalArgumentException("Soprano has one fixed voice and does not accept voice");
    }
    if (Float.compare(options.speed(), 1.0f) != 0) {
      throw new IllegalArgumentException("Soprano does not support speech speed adjustment");
    }
    if (options.referenceAudio() != null || options.referenceText() != null) {
      throw new IllegalArgumentException("Soprano does not support reference voice cloning");
    }
    if (options.sampling() != null && !options.sampling().stopSequences().isEmpty()) {
      throw new IllegalArgumentException(
          "Soprano acoustic generation does not support stop sequences");
    }
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("Soprano model is closed");
    }
  }
}
