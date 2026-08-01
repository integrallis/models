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
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaForwardPass;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.plan.ExecutionPlanner;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import com.integrallis.models.backend.purejava.plan.PureJavaExecutionPlan;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Pure Java inference backend that loads a GGUF model and runs Llama-family forward passes without
 * any native dependencies.
 */
public final class PureJavaBackend implements SpeculativeInferenceBackend, BatchInferenceBackend {

  public static final String MAX_CONTEXT_LENGTH_PROPERTY = "models.purejava.maxContextLength";

  private final Arena arena;
  private final LlamaConfig config;
  private final GgufTokenizer tokenizer;
  private final LlamaForwardPass forwardPass;
  private final ModelMetadata modelMetadata;
  private final PureJavaExecutionPlan executionPlan;
  private final BackendDiagnostics diagnostics;
  private final GgufBatchedMatrixKernel batchedMatrixKernel;
  private LlamaForwardPass.Session[] sessionBatch = new LlamaForwardPass.Session[0];
  private boolean closed;

  private static final class PureJavaInferenceSession implements InferenceSession {
    private final PureJavaBackend owner;
    private final LlamaForwardPass.Session delegate;
    private boolean closed;

    private PureJavaInferenceSession(PureJavaBackend owner, LlamaForwardPass.Session delegate) {
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
      LlamaConfig config,
      GgufTokenizer tokenizer,
      LlamaForwardPass forwardPass,
      ModelMetadata modelMetadata,
      PureJavaExecutionPlan executionPlan,
      BackendDiagnostics diagnostics,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    this.arena = arena;
    this.config = config;
    this.tokenizer = tokenizer;
    this.forwardPass = forwardPass;
    this.modelMetadata = modelMetadata;
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
    try {
      Objects.requireNonNull(modelPath, "modelPath");
      GgufFile file = GgufParser.parse(modelPath, arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      String modelFamily = file.metadata().getString("general.architecture").orElse("llama");
      RuntimeFingerprint runtime = RuntimeFingerprint.capture();
      PureJavaExecutionPlan executionPlan =
          ExecutionPlanner.plan(
              runtime,
              ModelTopology.from(modelFamily, config, weights),
              PureJavaPlanConfiguration.fromSystemProperties(
                  recommendations(backendConfiguration, batchedMatrixKernel)),
              batchedMatrixKernel);
      KvCache cache =
          new KvCache(
              config.numLayers(), runtimeContextLength(config), config.keyDim(), config.valueDim());
      LlamaForwardPass forwardPass =
          new LlamaForwardPass(config, weights, cache, executionPlan, batchedMatrixKernel);

      String modelName =
          file.metadata().getString("general.name").orElse(modelPath.getFileName().toString());
      ModelMetadata metadata =
          new ModelMetadata(
              modelFamily,
              modelName,
              config.contextLength(),
              config.vocabSize(),
              config.embeddingDim(),
              config.numLayers(),
              config.numHeads(),
              config.numKvHeads());

      BackendDiagnostics diagnostics = backendConfiguration.enrich(executionPlan.diagnostics());

      return new PureJavaBackend(
          arena,
          config,
          tokenizer,
          forwardPass,
          metadata,
          executionPlan,
          diagnostics,
          batchedMatrixKernel);
    } catch (IOException e) {
      closeAfterFailure(arena, batchedMatrixKernel, e);
      throw new UncheckedIOException("Failed to load model: " + modelPath, e);
    } catch (RuntimeException | Error e) {
      closeAfterFailure(arena, batchedMatrixKernel, e);
      throw e;
    }
  }

  @Override
  public String name() {
    return "pure-java";
  }

  @Override
  public ModelMetadata metadata() {
    return modelMetadata;
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
    return forwardPass.maxSessionBatchSize();
  }

  @Override
  public float[] forward(int token, int position) {
    return forwardPass.forward(token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    return forwardPass.forwardTransient(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return forwardPass.prefill(tokens, startPosition);
  }

  @Override
  public InferenceSession openSession() {
    checkOpen();
    return new PureJavaInferenceSession(this, forwardPass.openSession());
  }

  @Override
  public float[] forward(InferenceSession session, int token, int position) {
    return forwardPass.forward(requireOpen(session).delegate, token, position);
  }

  @Override
  public float[] forwardTransient(InferenceSession session, int token, int position) {
    return forwardPass.forwardTransient(requireOpen(session).delegate, token, position);
  }

  @Override
  public float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
    return forwardPass.prefill(requireOpen(session).delegate, tokens, startPosition);
  }

  @Override
  public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
    return forwardPass.forwardBatch(unwrapSessions(sessions), tokens);
  }

  @Override
  public LogitBatch forwardBatchTransient(InferenceSession[] sessions, int[] tokens) {
    return forwardPass.forwardBatchTransient(unwrapSessions(sessions), tokens);
  }

  @Override
  public void rewind(InferenceSession session, int checkpoint) {
    forwardPass.rewind(requireOpen(session).delegate, checkpoint);
  }

  @Override
  public void reset(InferenceSession session) {
    forwardPass.reset(requireOpen(session).delegate);
  }

  @Override
  public int checkpoint() {
    return forwardPass.checkpoint();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    return forwardPass.verify(tokens, startPosition);
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    return forwardPass.verifyTransient(tokens, startPosition);
  }

  @Override
  public void rewind(int checkpoint) {
    forwardPass.rewind(checkpoint);
  }

  @Override
  public void reset() {
    forwardPass.reset();
  }

  @Override
  public void close() {
    RuntimeException closeFailure = null;
    try {
      batchedMatrixKernel.close();
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
    closed = true;
    if (closeFailure != null) {
      throw closeFailure;
    }
  }

  private LlamaForwardPass.Session[] unwrapSessions(InferenceSession[] sessions) {
    Objects.requireNonNull(sessions, "sessions");
    if (sessionBatch.length != sessions.length) {
      sessionBatch = new LlamaForwardPass.Session[sessions.length];
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
      forwardPass.reset(session.delegate);
    }
  }

  private void checkOpen() {
    if (closed) {
      throw new IllegalStateException("backend is closed");
    }
  }

  private static int runtimeContextLength(LlamaConfig config) {
    String value = System.getProperty(MAX_CONTEXT_LENGTH_PROPERTY);
    if (value == null || value.isBlank()) {
      return config.contextLength();
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
    return Math.min(config.contextLength(), maxContextLength);
  }

  private static Map<String, String> recommendations(
      BackendConfiguration backendConfiguration, GgufBatchedMatrixKernel batchedMatrixKernel) {
    Map<String, String> combined = new LinkedHashMap<>(backendConfiguration.recommendations());
    combined.putAll(batchedMatrixKernel.planRecommendations());
    return Map.copyOf(combined);
  }

  private static void closeAfterFailure(
      Arena arena, GgufBatchedMatrixKernel batchedMatrixKernel, Throwable failure) {
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
}
