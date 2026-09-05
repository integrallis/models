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
package com.integrallis.models.backend.purejava.soprano;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufEmbeddedFiles;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.llama.DecoderArchitecture;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaForwardPass;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.plan.ExecutionPlanner;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import com.integrallis.models.backend.purejava.plan.PureJavaExecutionPlan;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Mapped pure-Java execution engine for a standalone Soprano GGUF artifact. */
public final class SopranoBackend implements AutoCloseable {

  /** Stable vocabulary logits and normalized acoustic feature frame from one LM step. */
  public record Step(float[] logits, float[] hiddenState) {
    public Step {
      Objects.requireNonNull(logits, "logits");
      Objects.requireNonNull(hiddenState, "hiddenState");
    }
  }

  private final Arena arena;
  private final SopranoConfig config;
  private final SopranoTokenizer tokenizer;
  private final LlamaForwardPass languageModel;
  private final SopranoVocoder vocoder;
  private final GgufBatchedMatrixKernel matrixKernel;
  private final BackendDiagnostics diagnostics;
  private boolean closed;

  private SopranoBackend(
      Arena arena,
      SopranoConfig config,
      SopranoTokenizer tokenizer,
      LlamaForwardPass languageModel,
      SopranoVocoder vocoder,
      GgufBatchedMatrixKernel matrixKernel,
      BackendDiagnostics diagnostics) {
    this.arena = arena;
    this.config = config;
    this.tokenizer = tokenizer;
    this.languageModel = languageModel;
    this.vocoder = vocoder;
    this.matrixKernel = Objects.requireNonNull(matrixKernel, "matrixKernel");
    this.diagnostics = Objects.requireNonNull(diagnostics, "diagnostics");
  }

  /** Memory-maps and validates a self-contained Soprano audio.cpp GGUF package. */
  public static SopranoBackend load(Path modelPath) throws IOException {
    return load(modelPath, GgufBatchedMatrixKernel.none());
  }

  /**
   * Memory-maps Soprano and injects an optional projection kernel into its Java language-model
   * graph. The returned backend owns and closes the supplied kernel.
   */
  public static SopranoBackend load(Path modelPath, GgufBatchedMatrixKernel matrixKernel)
      throws IOException {
    Objects.requireNonNull(modelPath, "modelPath");
    Objects.requireNonNull(matrixKernel, "matrixKernel");
    Arena arena = Arena.ofShared();
    try {
      GgufFile file = GgufParser.parse(modelPath, arena);
      requireMetadata(file, "general.architecture", "audiocpp");
      requireMetadata(file, "audiocpp.model_spec.family", "soprano_tts");
      GgufEmbeddedFiles embedded = GgufEmbeddedFiles.from(file.metadata());
      SopranoConfig config = SopranoConfig.fromJson(embedded.readUtf8("config.json"));
      SopranoTokenizer tokenizer = SopranoTokenizer.fromJson(embedded.readUtf8("tokenizer.json"));
      validateTokenContract(config, tokenizer);
      LlamaConfig languageConfig = languageConfig(config);
      LlamaWeights languageWeights = LlamaWeights.fromHuggingFaceNamedGguf(file, languageConfig);
      ModelTopology topology =
          ModelTopology.from(
              languageConfig.architecture().metadataId(), languageConfig, languageWeights);
      PureJavaExecutionPlan executionPlan =
          ExecutionPlanner.plan(
              RuntimeFingerprint.capture(),
              topology,
              PureJavaPlanConfiguration.fromSystemProperties(matrixKernel.planRecommendations()),
              matrixKernel);
      LlamaForwardPass languageModel =
          new LlamaForwardPass(
              languageConfig,
              languageWeights,
              new KvCache(
                  languageConfig.numLayers(),
                  languageConfig.contextLength(),
                  languageConfig.keyDim(),
                  languageConfig.valueDim()),
              executionPlan,
              matrixKernel);
      SopranoVocoderWeights vocoderWeights = SopranoVocoderWeights.load(file, config);
      SopranoVocoder vocoder = new SopranoVocoder(config, vocoderWeights);
      return new SopranoBackend(
          arena,
          config,
          tokenizer,
          languageModel,
          vocoder,
          matrixKernel,
          withVocoderDiagnostics(executionPlan.diagnostics(), vocoderWeights, matrixKernel));
    } catch (IOException | RuntimeException | Error failure) {
      closeAfterFailure(arena, matrixKernel, failure);
      throw failure;
    }
  }

  /** Starts a new utterance and returns the final prompt position's sampling outputs. */
  public synchronized Step begin(String text) {
    return begin(encodePrompt(text));
  }

  /** Encodes text using the tokenizer embedded in the standalone artifact. */
  public synchronized int[] encodePrompt(String text) {
    requireOpen();
    if (text == null || text.isBlank()) {
      throw new IllegalArgumentException("Soprano text must not be blank");
    }
    return tokenizer.encodePrompt(text);
  }

  /** Starts a new utterance from an already encoded Soprano prompt. */
  public synchronized Step begin(int[] prompt) {
    requireOpen();
    Objects.requireNonNull(prompt, "prompt");
    if (prompt.length == 0) {
      throw new IllegalArgumentException("Soprano prompt must not be empty");
    }
    if (prompt.length >= config.contextLength()) {
      throw new IllegalArgumentException(
          "Soprano prompt uses "
              + prompt.length
              + " tokens but its context holds "
              + config.contextLength());
    }
    for (int token : prompt) {
      if (token < 0 || token >= config.vocabSize()) {
        throw new IllegalArgumentException(
            "Soprano prompt token is outside its vocabulary: " + token);
      }
    }
    languageModel.reset();
    return step(languageModel.prefillWithHidden(prompt.clone(), 0));
  }

  /** Advances the current utterance by one sampled acoustic token. */
  public synchronized Step advance(int token) {
    requireOpen();
    if (token < 0 || token >= config.vocabSize()) {
      throw new IllegalArgumentException("Soprano token is outside its vocabulary: " + token);
    }
    if (languageModel.checkpoint() == 0) {
      throw new IllegalStateException("begin(text) must be called before advance(token)");
    }
    if (languageModel.checkpoint() >= config.contextLength()) {
      throw new IllegalStateException("Soprano context is full");
    }
    return step(languageModel.forwardWithHidden(token, languageModel.checkpoint()));
  }

  /** Decodes frame-major acoustic features into normalized 32 kHz mono PCM. */
  public synchronized float[] decode(float[] features, int frames) {
    requireOpen();
    return vocoder.decode(features, frames);
  }

  public synchronized int checkpoint() {
    requireOpen();
    return languageModel.checkpoint();
  }

  public int eosToken() {
    return tokenizer.eosToken();
  }

  public int vocabularySize() {
    return config.vocabSize();
  }

  public int hiddenSize() {
    return config.hiddenSize();
  }

  public int contextLength() {
    return config.contextLength();
  }

  public int sampleRate() {
    return config.sampleRate();
  }

  /** PCM samples produced by one acoustic-token interval. */
  public int samplesPerToken() {
    return Math.multiplyExact(config.upscale(), config.hopLength());
  }

  /** Returns the selected Java execution plan and runtime environment. */
  public BackendDiagnostics diagnostics() {
    return diagnostics;
  }

  @Override
  public synchronized void close() {
    if (!closed) {
      closed = true;
      RuntimeException closeFailure = null;
      try {
        matrixKernel.close();
      } catch (RuntimeException failure) {
        closeFailure = failure;
      }
      try {
        arena.close();
      } catch (RuntimeException failure) {
        if (closeFailure == null) {
          closeFailure = failure;
        } else {
          closeFailure.addSuppressed(failure);
        }
      }
      if (closeFailure != null) {
        throw closeFailure;
      }
    }
  }

  private static Step step(LlamaForwardPass.StepOutput output) {
    return new Step(output.logits(), output.hiddenState());
  }

  private static LlamaConfig languageConfig(SopranoConfig config) {
    return new LlamaConfig(
        DecoderArchitecture.QWEN3,
        config.hiddenSize(),
        config.layers(),
        config.attentionHeads(),
        config.kvHeads(),
        config.headDim(),
        config.headDim(),
        config.vocabSize(),
        config.contextLength(),
        config.intermediateSize(),
        config.ropeTheta(),
        1.0f,
        config.ropeTheta(),
        config.rmsNormEpsilon(),
        0,
        1,
        0.0f);
  }

  private static BackendDiagnostics withVocoderDiagnostics(
      BackendDiagnostics diagnostics,
      SopranoVocoderWeights weights,
      GgufBatchedMatrixKernel matrixKernel) {
    Map<String, String> environment = new LinkedHashMap<>(diagnostics.environment());
    environment.put("matrix-kernel", matrixKernel.implementation());
    ArrayList<OptimizationDecision> optimizations = new ArrayList<>(diagnostics.optimizations());
    optimizations.add(
        new OptimizationDecision(
            "soprano.vocoder.prepared-f32",
            OptimizationStatus.ENABLED,
            "Q8_0 vocoder projections are expanded once for faster SIMD F32 batch execution",
            Map.of(
                "serializedBytes",
                Long.toString(weights.preparedSerializedBytes()),
                "expandedBytes",
                Long.toString(weights.preparedExpandedBytes()))));
    return new BackendDiagnostics(
        diagnostics.backend(), diagnostics.planVersion(), environment, optimizations);
  }

  private static void closeAfterFailure(
      Arena arena, GgufBatchedMatrixKernel matrixKernel, Throwable failure) {
    try {
      matrixKernel.close();
    } catch (RuntimeException closeFailure) {
      failure.addSuppressed(closeFailure);
    }
    try {
      arena.close();
    } catch (RuntimeException closeFailure) {
      failure.addSuppressed(closeFailure);
    }
  }

  private static void requireMetadata(GgufFile file, String key, String expected) {
    String actual = file.metadata().getString(key).orElse(null);
    if (!expected.equals(actual)) {
      throw new IllegalArgumentException(
          "Soprano GGUF " + key + " must be " + expected + "; got " + actual);
    }
  }

  static void validateTokenContract(SopranoConfig config, SopranoTokenizer tokenizer) {
    if (config.eosToken() != tokenizer.eosToken()) {
      throw new IllegalArgumentException(
          "Soprano config eos_token_id "
              + config.eosToken()
              + " does not match tokenizer [STOP] token "
              + tokenizer.eosToken());
    }
    if (config.bosToken() != tokenizer.eosToken()) {
      throw new IllegalArgumentException(
          "Soprano config bos_token_id "
              + config.bosToken()
              + " does not match tokenizer [STOP] prompt token "
              + tokenizer.eosToken());
    }
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("Soprano backend is closed");
    }
  }
}
