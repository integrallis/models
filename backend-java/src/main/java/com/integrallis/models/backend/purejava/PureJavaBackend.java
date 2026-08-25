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
package com.integrallis.models.backend.purejava;

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.cact.CactFile;
import com.integrallis.models.backend.purejava.cact.CactHeader;
import com.integrallis.models.backend.purejava.cact.CactNeedle2Layout;
import com.integrallis.models.backend.purejava.cact.CactParser;
import com.integrallis.models.backend.purejava.cact.CactTokenizer;
import com.integrallis.models.backend.purejava.cact.Needle2Weights;
import com.integrallis.models.backend.purejava.gemma4.Gemma4Config;
import com.integrallis.models.backend.purejava.gemma4.Gemma4Decoder;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.huggingface.Qwen2HuggingFaceConfig;
import com.integrallis.models.backend.purejava.llama.DenseProjectionHead;
import com.integrallis.models.backend.purejava.llama.EncoderForwardPass;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaForwardPass;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.plan.ExecutionPlanner;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import com.integrallis.models.backend.purejava.plan.PureJavaExecutionPlan;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import com.integrallis.models.backend.purejava.tokenizer.HuggingFaceTokenizer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Pure Java inference backend that loads a GGUF model and runs Llama-family forward passes without
 * any native dependencies.
 */
public final class PureJavaBackend implements SpeculativeInferenceBackend, BatchInferenceBackend {

  public static final String MAX_CONTEXT_LENGTH_PROPERTY = "models.purejava.maxContextLength";

  private final Arena arena;
  private final Tokenizer tokenizer;
  private final PureJavaDecoder decoder;
  private final ModelMetadata modelMetadata;
  private final int contextCapacity;
  private final PureJavaExecutionPlan executionPlan;
  private final BackendDiagnostics diagnostics;
  private final GgufBatchedMatrixKernel batchedMatrixKernel;
  private PureJavaDecoder.Session[] sessionBatch = new PureJavaDecoder.Session[0];
  private boolean closed;

  private record LoadedDecoder(
      PureJavaDecoder decoder,
      ModelMetadata metadata,
      int contextCapacity,
      PureJavaExecutionPlan executionPlan) {}

  private static final class PureJavaInferenceSession implements InferenceSession {
    private final PureJavaBackend owner;
    private final PureJavaDecoder.Session delegate;
    private boolean closed;

    private PureJavaInferenceSession(PureJavaBackend owner, PureJavaDecoder.Session delegate) {
      this.owner = owner;
      this.delegate = delegate;
    }

    @Override
    public int checkpoint() {
      owner.requireOpen(this);
      return delegate.checkpoint();
    }

    @Override
    public boolean isClosed() {
      return closed || owner.closed;
    }

    @Override
    public void close() {
      owner.closeSession(this);
    }
  }

  private PureJavaBackend(
      Arena arena,
      Tokenizer tokenizer,
      PureJavaDecoder decoder,
      ModelMetadata modelMetadata,
      int contextCapacity,
      PureJavaExecutionPlan executionPlan,
      BackendDiagnostics diagnostics,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    this.arena = arena;
    this.tokenizer = tokenizer;
    this.decoder = decoder;
    this.modelMetadata = modelMetadata;
    this.contextCapacity = contextCapacity;
    this.executionPlan = executionPlan;
    this.diagnostics = diagnostics;
    this.batchedMatrixKernel = batchedMatrixKernel;
  }

  /** Loads a GGUF model file and returns a ready-to-use backend. */
  public static PureJavaBackend load(Path modelPath) {
    return load(
        modelPath, Arena.ofShared(), BackendConfiguration.empty(), GgufBatchedMatrixKernel.none());
  }

  /** Loads a GGUF model with registry-neutral, artifact-qualified recommendations. */
  public static PureJavaBackend load(Path modelPath, BackendConfiguration backendConfiguration) {
    return load(
        modelPath,
        Arena.ofShared(),
        Objects.requireNonNull(backendConfiguration, "backendConfiguration"),
        GgufBatchedMatrixKernel.none());
  }

  /**
   * Loads a GGUF model with an injected batched projection implementation.
   *
   * <p>The returned backend owns and closes {@code batchedMatrixKernel}.
   */
  public static PureJavaBackend load(Path modelPath, GgufBatchedMatrixKernel batchedMatrixKernel) {
    return load(
        modelPath,
        Arena.ofShared(),
        BackendConfiguration.empty(),
        Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel"));
  }

  /**
   * Loads a GGUF model with an injected projection kernel and registry-neutral recommendations.
   *
   * <p>The returned backend owns and closes {@code batchedMatrixKernel}.
   */
  public static PureJavaBackend load(
      Path modelPath,
      BackendConfiguration backendConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    return load(
        modelPath,
        Arena.ofShared(),
        Objects.requireNonNull(backendConfiguration, "backendConfiguration"),
        Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel"));
  }

  static PureJavaBackend load(Path modelPath, Arena arena) {
    return load(modelPath, arena, BackendConfiguration.empty(), GgufBatchedMatrixKernel.none());
  }

  private static PureJavaBackend load(
      Path modelPath,
      Arena arena,
      BackendConfiguration backendConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    Objects.requireNonNull(arena, "arena");
    Objects.requireNonNull(backendConfiguration, "backendConfiguration");
    Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel");
    LoadedDecoder loaded = null;
    try {
      Objects.requireNonNull(modelPath, "modelPath");
      RuntimeFingerprint runtime = RuntimeFingerprint.capture();
      Map<String, String> recommendations =
          recommendations(backendConfiguration, batchedMatrixKernel);
      PureJavaPlanConfiguration planConfiguration =
          PureJavaPlanConfiguration.fromSystemProperties(recommendations);
      Tokenizer tokenizer;
      if (Files.isDirectory(modelPath)) {
        Qwen2HuggingFaceConfig config =
            Qwen2HuggingFaceConfig.parse(modelPath.resolve("config.json"));
        tokenizer =
            HuggingFaceTokenizer.fromQwen2(
                modelPath.resolve("tokenizer.json"),
                modelPath.resolve("tokenizer_config.json"),
                config);
        loaded =
            loadHuggingFaceQwen2(
                modelPath,
                config,
                new SafetensorsTensorSource(SafetensorsBundle.open(modelPath, arena)),
                runtime,
                planConfiguration,
                batchedMatrixKernel);
      } else if (CactParser.matches(modelPath)) {
        CactFile file = CactParser.parse(modelPath, arena);
        CactNeedle2Layout layout = CactNeedle2Layout.from(file);
        tokenizer = CactTokenizer.from(file);
        loaded = loadNeedle2(modelPath, layout, runtime, planConfiguration, batchedMatrixKernel);
      } else {
        GgufFile file = GgufParser.parse(modelPath, arena);
        tokenizer = GgufTokenizer.fromMetadata(file.metadata());
        String modelFamily = file.metadata().getString("general.architecture").orElse("llama");
        loaded =
            "gemma4".equals(modelFamily)
                ? loadGemma4(
                    modelPath,
                    file,
                    modelFamily,
                    runtime,
                    planConfiguration,
                    backendConfiguration,
                    batchedMatrixKernel)
                : loadLlama(
                    modelPath, file, modelFamily, runtime, planConfiguration, batchedMatrixKernel);
      }

      BackendDiagnostics diagnostics =
          backendConfiguration.enrich(
              architectureDiagnostics(loaded.executionPlan().diagnostics(), loaded.decoder()));

      return new PureJavaBackend(
          arena,
          tokenizer,
          loaded.decoder(),
          loaded.metadata(),
          loaded.contextCapacity(),
          loaded.executionPlan(),
          diagnostics,
          batchedMatrixKernel);
    } catch (IOException e) {
      closeAfterFailure(null, arena, batchedMatrixKernel, e);
      throw new UncheckedIOException("Failed to load model: " + modelPath, e);
    } catch (RuntimeException | Error e) {
      closeAfterFailure(loaded == null ? null : loaded.decoder(), arena, batchedMatrixKernel, e);
      throw e;
    }
  }

  private static LoadedDecoder loadNeedle2(
      Path modelPath,
      CactNeedle2Layout layout,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    CactHeader config = layout.header();
    int queryWidth = Math.multiplyExact(config.queryHeadCount(), config.headWidth());
    int kvWidth = Math.multiplyExact(config.kvHeadCount(), config.headWidth());
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(
            runtime,
            ModelTopology.mappedArchitecture(
                "needle2", queryWidth, kvWidth, kvWidth, config.layerCount()),
            planConfiguration,
            batchedMatrixKernel);
    int contextCapacity = runtimeContextLength(config.maximumSequenceLength());
    ModelMetadata metadata =
        new ModelMetadata(
            "needle2",
            modelPath.getFileName().toString(),
            config.maximumSequenceLength(),
            config.vocabularySize(),
            config.modelWidth(),
            config.layerCount(),
            config.queryHeadCount(),
            config.kvHeadCount());
    return new LoadedDecoder(
        new Needle2DecoderAdapter(Needle2Weights.load(layout), contextCapacity),
        metadata,
        contextCapacity,
        executionPlan);
  }

  private static LoadedDecoder loadHuggingFaceQwen2(
      Path modelPath,
      Qwen2HuggingFaceConfig huggingFaceConfig,
      SafetensorsTensorSource tensors,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    LlamaConfig config = huggingFaceConfig.model();
    LlamaWeights weights = LlamaWeights.fromQwen2Safetensors(tensors, huggingFaceConfig);
    String modelFamily = "qwen2";
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(
            runtime,
            ModelTopology.from(modelFamily, config, weights),
            planConfiguration,
            batchedMatrixKernel);
    int contextCapacity = runtimeContextLength(config.contextLength());
    KvCache cache =
        new KvCache(config.numLayers(), contextCapacity, config.keyDim(), config.valueDim());
    PureJavaDecoder decoder =
        new LlamaDecoder(
            new LlamaForwardPass(config, weights, cache, executionPlan, batchedMatrixKernel));
    ModelMetadata metadata =
        new ModelMetadata(
            modelFamily,
            modelPath.getFileName().toString(),
            config.contextLength(),
            config.vocabSize(),
            config.embeddingDim(),
            config.numLayers(),
            config.numHeads(),
            config.numKvHeads());
    return new LoadedDecoder(decoder, metadata, contextCapacity, executionPlan);
  }

  private static LoadedDecoder loadLlama(
      Path modelPath,
      GgufFile file,
      String modelFamily,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
    LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(
            runtime,
            ModelTopology.from(modelFamily, config, weights),
            planConfiguration,
            batchedMatrixKernel);
    int contextCapacity = runtimeContextLength(config.contextLength());
    PureJavaDecoder decoder;
    if (config.usesBidirectionalAttention()) {
      // An encoder holds no KV cache: every position depends on every other, so there is nothing
      // from a previous step to reuse.
      decoder =
          new EncoderDecoderAdapter(
              new EncoderForwardPass(
                  config,
                  weights,
                  DenseProjectionHead.load(file, modelFamily, config.embeddingDim()).orElse(null)));
    } else {
      KvCache cache =
          new KvCache(config.numLayers(), contextCapacity, config.keyDim(), config.valueDim());
      decoder =
          new LlamaDecoder(
              new LlamaForwardPass(config, weights, cache, executionPlan, batchedMatrixKernel));
    }
    ModelMetadata metadata =
        new ModelMetadata(
            modelFamily,
            modelName(modelPath, file),
            config.contextLength(),
            config.vocabSize(),
            config.embeddingDim(),
            config.numLayers(),
            config.numHeads(),
            config.numKvHeads());
    return new LoadedDecoder(decoder, metadata, contextCapacity, executionPlan);
  }

  private static LoadedDecoder loadGemma4(
      Path modelPath,
      GgufFile file,
      String modelFamily,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      BackendConfiguration backendConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel)
      throws IOException {
    Gemma4Config config = Gemma4Config.fromMetadata(file.metadata());
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(
            runtime, gemma4Topology(file, config), planConfiguration, batchedMatrixKernel);
    ModelMetadata metadata =
        new ModelMetadata(
            modelFamily,
            modelName(modelPath, file),
            config.contextLength(),
            config.vocabSize(),
            config.embeddingDim(),
            config.numLayers(),
            config.numHeads(),
            config.numKvHeads(0));
    int contextCapacity = runtimeContextLength(config.contextLength());
    Gemma4Decoder decoder = Gemma4Decoder.load(file, contextCapacity, batchedMatrixKernel);
    return new LoadedDecoder(
        new Gemma4DecoderAdapter(decoder), metadata, contextCapacity, executionPlan);
  }

  @Override
  public String name() {
    return "pure-java";
  }

  @Override
  public ModelMetadata metadata() {
    return modelMetadata;
  }

  @Override
  public int contextCapacity() {
    return contextCapacity;
  }

  /** Returns the immutable execution plan selected while loading this model. */
  public PureJavaExecutionPlan executionPlan() {
    return executionPlan;
  }

  @Override
  public BackendDiagnostics diagnostics() {
    return diagnostics;
  }

  @Override
  public Tokenizer tokenizer() {
    return tokenizer;
  }

  @Override
  public int maxBatchSize() {
    checkOpen();
    return decoder.maxBatchSize();
  }

  @Override
  public float[] forward(int token, int position) {
    checkOpen();
    return decoder.forward(token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    checkOpen();
    return decoder.forwardTransient(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    checkOpen();
    return decoder.prefill(tokens, startPosition);
  }

  /**
   * Runs the stack over a sequence and returns the final position's hidden state.
   *
   * <p>Skips the vocabulary projection, which is the widest matmul in the pass — so producing an
   * embedding costs less per token than generating one.
   *
   * <p>The array is backend-owned scratch, valid until the next call. Copy it to keep it.
   */
  public float[] prefillHiddenState(int[] tokens, int startPosition) {
    checkOpen();
    return decoder.prefillHiddenState(tokens, startPosition);
  }

  /**
   * Runs one step and returns its hidden state instead of logits.
   *
   * <p>Needed for mean pooling, which reduces over every position rather than only the last.
   *
   * <p>The array is backend-owned scratch, valid until the next call.
   */
  public float[] hiddenState(int token, int position) {
    checkOpen();
    return decoder.hiddenState(token, position);
  }

  /** Whether this model's architecture exposes hidden states for embedding. */
  public boolean supportsHiddenState() {
    return decoder.supportsHiddenState();
  }

  /**
   * Encodes a whole sequence into one vector using the model's own pooling and projection.
   *
   * <p>Available only where the model owns its embedding pipeline; check {@link
   * #supportsSequenceEmbedding()} first.
   *
   * @param tokens the tokenized text
   * @return a newly allocated pooled embedding, not normalized
   */
  public float[] embedSequence(int[] tokens) {
    checkOpen();
    return decoder.embedSequence(tokens);
  }

  /** Whether this model encodes whole sequences rather than exposing per-position states. */
  public boolean supportsSequenceEmbedding() {
    return decoder.supportsSequenceEmbedding();
  }

  @Override
  public InferenceSession openSession() {
    checkOpen();
    return new PureJavaInferenceSession(this, decoder.openSession());
  }

  @Override
  public float[] forward(InferenceSession session, int token, int position) {
    return decoder.forward(requireOpen(session).delegate, token, position);
  }

  @Override
  public float[] forwardTransient(InferenceSession session, int token, int position) {
    return decoder.forwardTransient(requireOpen(session).delegate, token, position);
  }

  @Override
  public float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
    return decoder.prefill(requireOpen(session).delegate, tokens, startPosition);
  }

  @Override
  public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
    return decoder.forwardBatch(unwrapSessions(sessions), tokens);
  }

  @Override
  public LogitBatch forwardBatchTransient(InferenceSession[] sessions, int[] tokens) {
    return decoder.forwardBatchTransient(unwrapSessions(sessions), tokens);
  }

  @Override
  public void rewind(InferenceSession session, int checkpoint) {
    decoder.rewind(requireOpen(session).delegate, checkpoint);
  }

  @Override
  public void reset(InferenceSession session) {
    decoder.reset(requireOpen(session).delegate);
  }

  @Override
  public int checkpoint() {
    checkOpen();
    return decoder.checkpoint();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    checkOpen();
    return decoder.verify(tokens, startPosition);
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    checkOpen();
    return decoder.verifyTransient(tokens, startPosition);
  }

  @Override
  public void rewind(int checkpoint) {
    checkOpen();
    decoder.rewind(checkpoint);
  }

  @Override
  public void reset() {
    checkOpen();
    decoder.reset();
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    RuntimeException closeFailure = null;
    try {
      decoder.close();
    } catch (IOException failure) {
      closeFailure = new UncheckedIOException("Failed to close decoder resources", failure);
    } catch (RuntimeException failure) {
      closeFailure = failure;
    }
    try {
      batchedMatrixKernel.close();
    } catch (RuntimeException failure) {
      closeFailure = combineCloseFailures(closeFailure, failure);
    }
    try {
      arena.close();
    } catch (RuntimeException failure) {
      closeFailure = combineCloseFailures(closeFailure, failure);
    }
    if (closeFailure != null) {
      throw closeFailure;
    }
  }

  private PureJavaDecoder.Session[] unwrapSessions(InferenceSession[] sessions) {
    Objects.requireNonNull(sessions, "sessions");
    if (sessionBatch.length != sessions.length) {
      sessionBatch = new PureJavaDecoder.Session[sessions.length];
    }
    for (int index = 0; index < sessions.length; index++) {
      sessionBatch[index] = requireOpen(sessions[index]).delegate;
    }
    return sessionBatch;
  }

  private PureJavaInferenceSession requireOpen(InferenceSession session) {
    checkOpen();
    Objects.requireNonNull(session, "session");
    if (!(session instanceof PureJavaInferenceSession pureJavaSession)
        || pureJavaSession.owner != this) {
      throw new IllegalArgumentException("session belongs to a different backend");
    }
    return requireOpen(pureJavaSession);
  }

  private PureJavaInferenceSession requireOpen(PureJavaInferenceSession session) {
    checkOpen();
    if (session.closed) {
      throw new IllegalStateException("session is closed");
    }
    return session;
  }

  private void closeSession(PureJavaInferenceSession session) {
    if (session.closed) {
      return;
    }
    session.closed = true;
    if (!closed) {
      decoder.reset(session.delegate);
    }
  }

  private void checkOpen() {
    if (closed) {
      throw new IllegalStateException("backend is closed");
    }
  }

  private static int runtimeContextLength(int modelContextLength) {
    String value = System.getProperty(MAX_CONTEXT_LENGTH_PROPERTY);
    if (value == null || value.isBlank()) {
      return modelContextLength;
    }
    int maxContextLength;
    try {
      maxContextLength = Integer.parseInt(value);
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException(
          MAX_CONTEXT_LENGTH_PROPERTY + " must be a positive integer: " + value, e);
    }
    if (maxContextLength <= 0) {
      throw new IllegalArgumentException(
          MAX_CONTEXT_LENGTH_PROPERTY + " must be a positive integer: " + value);
    }
    return Math.min(modelContextLength, maxContextLength);
  }

  private static ModelTopology gemma4Topology(GgufFile file, Gemma4Config config) {
    List<ModelTopology.LayerTopology> layers = new ArrayList<>(config.numLayers());
    int queryRows = 0;
    int keyRows = 0;
    int valueRows = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      String prefix = "blk." + layer + ".";
      GgufTensorType key = tensorType(file, prefix + "attn_k.weight");
      layers.add(
          new ModelTopology.LayerTopology(
              tensorType(file, prefix + "attn_q.weight"),
              key,
              config.usesSlidingWindow(layer) ? tensorType(file, prefix + "attn_v.weight") : key,
              tensorType(file, prefix + "attn_output.weight"),
              tensorType(file, prefix + "ffn_gate.weight"),
              tensorType(file, prefix + "ffn_up.weight"),
              tensorType(file, prefix + "ffn_down.weight")));
      queryRows = Math.max(queryRows, config.queryDim(layer));
      keyRows = Math.max(keyRows, config.keyDim(layer));
      valueRows = Math.max(valueRows, config.valueDim(layer));
    }
    return new ModelTopology("gemma4", queryRows, keyRows, valueRows, layers, true);
  }

  private static BackendDiagnostics architectureDiagnostics(
      BackendDiagnostics diagnostics, PureJavaDecoder decoder) {
    if (decoder instanceof Needle2DecoderAdapter needle2) {
      CactHeader header = needle2.header();
      Map<String, String> environment = new LinkedHashMap<>(diagnostics.environment());
      environment.put("artifact-format", "cact");
      environment.put("weight-encoding", "rotated-codebook");
      environment.put("kv-bits", Integer.toString(header.kvBits()));
      List<OptimizationDecision> optimizations = new ArrayList<>(diagnostics.optimizations());
      optimizations.add(
          new OptimizationDecision(
              "cact-rotated-codebook",
              OptimizationStatus.ENABLED,
              "packed CQ weights execute directly from the mapped artifact",
              Map.of("codebook-values", Integer.toString(header.codebookLength()))));
      return new BackendDiagnostics(
          diagnostics.backend(), diagnostics.planVersion(), environment, optimizations);
    }
    if (!(decoder instanceof Gemma4DecoderAdapter gemma4)) {
      return diagnostics;
    }
    int batchSize = gemma4.prefillBatchSize();
    Map<String, String> environment = new LinkedHashMap<>(diagnostics.environment());
    environment.put("architecture-prefill-batch-size", Integer.toString(batchSize));
    List<OptimizationDecision> optimizations = new ArrayList<>(diagnostics.optimizations());
    optimizations.add(
        new OptimizationDecision(
            "gemma4-batched-prefill",
            batchSize > 1 ? OptimizationStatus.ENABLED : OptimizationStatus.UNSUPPORTED,
            batchSize > 1
                ? "Gemma 4 attention, shared FFN, router, and routed experts use retained batched kernels"
                : "at least one Gemma 4 projection or routed-expert tensor lacks a retained batched kernel",
            Map.of("batch-size", Integer.toString(batchSize))));
    return new BackendDiagnostics(
        diagnostics.backend(), diagnostics.planVersion(), environment, optimizations);
  }

  private static GgufTensorType tensorType(GgufFile file, String name) {
    return file.getTensor(name).type();
  }

  private static String modelName(Path modelPath, GgufFile file) {
    return file.metadata().getString("general.name").orElse(modelPath.getFileName().toString());
  }

  private static Map<String, String> recommendations(
      BackendConfiguration backendConfiguration, GgufBatchedMatrixKernel batchedMatrixKernel) {
    Map<String, String> combined = new LinkedHashMap<>(backendConfiguration.recommendations());
    combined.putAll(batchedMatrixKernel.planRecommendations());
    return Map.copyOf(combined);
  }

  private static void closeAfterFailure(
      PureJavaDecoder decoder,
      Arena arena,
      GgufBatchedMatrixKernel batchedMatrixKernel,
      Throwable failure) {
    if (decoder != null) {
      try {
        decoder.close();
      } catch (IOException | RuntimeException closeFailure) {
        failure.addSuppressed(closeFailure);
      }
    }
    try {
      batchedMatrixKernel.close();
    } catch (RuntimeException closeFailure) {
      failure.addSuppressed(closeFailure);
    }
    try {
      arena.close();
    } catch (RuntimeException closeFailure) {
      failure.addSuppressed(closeFailure);
    }
  }

  private static RuntimeException combineCloseFailures(
      RuntimeException current, RuntimeException next) {
    if (current == null) {
      return next;
    }
    current.addSuppressed(next);
    return current;
  }
}
