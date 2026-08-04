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
package com.integrallis.models.backend.nativekernel;

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.plan.PureJavaExecutionPlan;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * GGUF backend that keeps transformer execution in Java and delegates qualified matrix kernels to
 * Models-owned Rust code through FFM.
 */
public final class RustFfmBackend implements SpeculativeInferenceBackend {
  public static final String LIBRARY_PATH_PROPERTY = "models.native.kernels.library";
  public static final String LIBRARY_PATH_ENV = "MODELS_NATIVE_KERNELS_LIBRARY";
  public static final String LOAD_WARMUP_PROPERTY = "models.native.loadWarmup";
  public static final String PLAN_VERSION = "rust-ffm-v12";

  private final PureJavaBackend delegate;
  private final BackendDiagnostics diagnostics;

  RustFfmBackend(PureJavaBackend delegate, BackendDiagnostics diagnostics) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.diagnostics = Objects.requireNonNull(diagnostics, "diagnostics");
  }

  /** Loads a model using the native library configured by a system property or environment. */
  public static RustFfmBackend load(Path modelPath) {
    return load(modelPath, BackendConfiguration.empty());
  }

  /** Loads a model with registry-neutral, artifact-qualified recommendations. */
  public static RustFfmBackend load(Path modelPath, BackendConfiguration backendConfiguration) {
    String configured = System.getProperty(LIBRARY_PATH_PROPERTY);
    if (configured == null || configured.isBlank()) {
      configured = System.getenv(LIBRARY_PATH_ENV);
    }
    if (configured == null || configured.isBlank()) {
      return load(modelPath, BundledNativeKernelLibrary.resolve(), backendConfiguration);
    }
    return load(modelPath, Path.of(configured), backendConfiguration);
  }

  /** Loads a model with an explicit Models native-kernel platform library. */
  public static RustFfmBackend load(Path modelPath, Path libraryPath) {
    return load(modelPath, libraryPath, BackendConfiguration.empty());
  }

  /** Loads a model with an explicit native library and artifact-qualified recommendations. */
  public static RustFfmBackend load(
      Path modelPath, Path libraryPath, BackendConfiguration backendConfiguration) {
    Objects.requireNonNull(modelPath, "modelPath");
    Objects.requireNonNull(libraryPath, "libraryPath");
    Objects.requireNonNull(backendConfiguration, "backendConfiguration");
    NativeKernelSettings settings =
        NativeKernelSettings.fromSystemProperties(backendConfiguration.recommendations());
    RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath, settings);
    PureJavaBackend engine = PureJavaBackend.load(modelPath, backendConfiguration, kernel);
    try {
      if (settings.loadWarmup()) {
        warmup(engine);
      }
      return new RustFfmBackend(
          engine, diagnostics(engine.diagnostics(), kernel, settings.loadWarmup()));
    } catch (RuntimeException | Error failure) {
      try {
        engine.close();
      } catch (RuntimeException | Error closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  static void warmup(PureJavaBackend delegate) {
    Objects.requireNonNull(delegate, "delegate");
    Tokenizer tokenizer = delegate.tokenizer();
    int[] tokens = tokenizer.encode(ModelPrompt.text("Compile the in-process inference path."));
    if (tokens.length < 2) {
      tokens = new int[] {tokenizer.bosToken(), tokenizer.bosToken()};
    }
    try {
      delegate.prefill(tokens, 0);
    } finally {
      delegate.reset();
    }
  }

  @Override
  public String name() {
    return "rust-ffm";
  }

  @Override
  public ModelMetadata metadata() {
    return delegate.metadata();
  }

  @Override
  public int contextCapacity() {
    return delegate.contextCapacity();
  }

  /** Returns the Java transformer execution plan surrounding the native kernels. */
  public PureJavaExecutionPlan executionPlan() {
    return delegate.executionPlan();
  }

  @Override
  public BackendDiagnostics diagnostics() {
    return diagnostics;
  }

  @Override
  public Tokenizer tokenizer() {
    return delegate.tokenizer();
  }

  @Override
  public float[] forward(int token, int position) {
    return delegate.forward(token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    return delegate.forwardTransient(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return delegate.prefill(tokens, startPosition);
  }

  @Override
  public int checkpoint() {
    return delegate.checkpoint();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    return delegate.verify(tokens, startPosition);
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    return delegate.verifyTransient(tokens, startPosition);
  }

  @Override
  public void rewind(int checkpoint) {
    delegate.rewind(checkpoint);
  }

  @Override
  public void reset() {
    delegate.reset();
  }

  @Override
  public void close() {
    delegate.close();
  }

  static BackendDiagnostics diagnostics(
      BackendDiagnostics javaDiagnostics, RustGgufBatchedMatrixKernel kernel) {
    return diagnostics(javaDiagnostics, kernel, false);
  }

  private static BackendDiagnostics diagnostics(
      BackendDiagnostics javaDiagnostics, RustGgufBatchedMatrixKernel kernel, boolean loadWarmup) {
    Map<String, String> environment = new LinkedHashMap<>(javaDiagnostics.environment());
    environment.put("transformer-runtime", "java");
    environment.put("kernel-runtime", "rust-ffm");
    environment.put("kernel-implementation", kernel.implementation());
    environment.put("native-kernel-abi", Integer.toString(NativeKernelLibrary.ABI_VERSION));
    environment.put("native-kernel-threads", Integer.toString(kernel.threadCount()));
    environment.put("native-quantized-decode", Boolean.toString(kernel.nativeDecodeEnabled()));
    environment.put("native-q5-0-grouped", Boolean.toString(kernel.q5_0GroupedEnabled()));
    environment.put("native-load-warmup", Boolean.toString(loadWarmup));
    List<OptimizationDecision> optimizations = new ArrayList<>(javaDiagnostics.optimizations());
    optimizations.add(
        new OptimizationDecision(
            "load-warmup",
            loadWarmup ? OptimizationStatus.ENABLED : OptimizationStatus.DISABLED,
            loadWarmup
                ? "one resettable prefill executes during model loading so the first request does not pay JIT compilation cost"
                : "disabled by " + LOAD_WARMUP_PROPERTY,
            Map.of("property", LOAD_WARMUP_PROPERTY, "sequence-state", "reset-after-warmup")));
    optimizations.add(
        nativeQuantizedDecision(
            "rust-q4-0-batched-matmul",
            "eligible Q4_0 batched and grouped projections execute in the Models Rust kernel"));
    optimizations.add(
        nativeQuantizedDecision(
            "rust-q5-0-batched-matmul",
            "eligible Q5_0 batched projections execute in the Models Rust kernel"));
    optimizations.add(
        nativeQuantizedDecision(
            "rust-q8-0-batched-matmul",
            "eligible Q8_0 batched and grouped projections execute in the Models Rust kernel"));
    optimizations.add(
        nativeQuantizedDecision(
            "rust-q4-k-batched-matmul",
            "eligible Q4_K batched and grouped projections execute in the Models Rust kernel"));
    optimizations.add(
        nativeQuantizedDecision(
            "rust-q5-k-batched-matmul",
            "eligible Q5_K batched and grouped projections execute in the Models Rust kernel"));
    optimizations.add(
        nativeQuantizedDecision(
            "rust-q6-k-batched-matmul",
            "eligible Q6_K batched and grouped projections execute in the Models Rust kernel"));
    optimizations.add(
        nativeQuantizedDecision(
            "rust-mixed-k-grouped-matmul",
            "mixed Q4_K, Q5_K, and Q6_K projections share one Q8_K activation quantization"));
    optimizations.add(
        new OptimizationDecision(
            "rust-q5-0-grouped-matmul",
            kernel.q5_0GroupedEnabled() ? OptimizationStatus.ENABLED : OptimizationStatus.DISABLED,
            kernel.q5_0GroupedEnabled()
                ? "explicitly enabled for profile qualification"
                : "controlled Qwen2.5-0.5B profiling favored independent Q5_0 projections",
            Map.of(
                "property",
                RustGgufBatchedMatrixKernel.Q5_0_GROUPED_PROPERTY,
                "qualification",
                "model-and-platform-specific")));
    optimizations.add(
        new OptimizationDecision(
            "rust-quantized-decode",
            kernel.nativeDecodeEnabled() ? OptimizationStatus.ENABLED : OptimizationStatus.DISABLED,
            kernel.nativeDecodeEnabled()
                ? "single-token Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, and Q6_K projections execute in the Models Rust kernel"
                : "disabled by " + RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY,
            Map.of(
                "property",
                RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY,
                "transformer",
                "java")));
    return new BackendDiagnostics("rust-ffm", PLAN_VERSION, environment, optimizations);
  }

  private static OptimizationDecision nativeQuantizedDecision(String id, String reason) {
    return new OptimizationDecision(
        id,
        OptimizationStatus.ENABLED,
        reason,
        Map.of(
            "abi", Integer.toString(NativeKernelLibrary.ABI_VERSION),
            "boundary", "panama-ffm",
            "transformer", "java"));
  }
}
