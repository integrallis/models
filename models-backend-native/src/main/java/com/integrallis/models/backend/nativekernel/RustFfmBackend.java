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

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
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
  public static final String PLAN_VERSION = "rust-ffm-v1";

  private final PureJavaBackend delegate;
  private final BackendDiagnostics diagnostics;

  private RustFfmBackend(PureJavaBackend delegate, BackendDiagnostics diagnostics) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.diagnostics = Objects.requireNonNull(diagnostics, "diagnostics");
  }

  /** Loads a model using the native library configured by a system property or environment. */
  public static RustFfmBackend load(Path modelPath) {
    String configured = System.getProperty(LIBRARY_PATH_PROPERTY);
    if (configured == null || configured.isBlank()) {
      configured = System.getenv(LIBRARY_PATH_ENV);
    }
    if (configured == null || configured.isBlank()) {
      return load(modelPath, BundledNativeKernelLibrary.resolve());
    }
    return load(modelPath, Path.of(configured));
  }

  /** Loads a model with an explicit Models native-kernel platform library. */
  public static RustFfmBackend load(Path modelPath, Path libraryPath) {
    Objects.requireNonNull(modelPath, "modelPath");
    Objects.requireNonNull(libraryPath, "libraryPath");
    RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath);
    PureJavaBackend engine = PureJavaBackend.load(modelPath, kernel);
    return new RustFfmBackend(engine, diagnostics(engine.diagnostics(), kernel.implementation()));
  }

  @Override
  public String name() {
    return "rust-ffm";
  }

  @Override
  public ModelMetadata metadata() {
    return delegate.metadata();
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

  private static BackendDiagnostics diagnostics(
      BackendDiagnostics javaDiagnostics, String kernelImplementation) {
    Map<String, String> environment = new LinkedHashMap<>(javaDiagnostics.environment());
    environment.put("transformer-runtime", "java");
    environment.put("kernel-runtime", "rust-ffm");
    environment.put("kernel-implementation", kernelImplementation);
    environment.put("native-kernel-abi", Integer.toString(NativeKernelLibrary.ABI_VERSION));
    List<OptimizationDecision> optimizations = new ArrayList<>(javaDiagnostics.optimizations());
    optimizations.add(
        new OptimizationDecision(
            "rust-q4-0-batched-matmul",
            OptimizationStatus.ENABLED,
            "eligible Q4_0 batched projections execute in the Models Rust kernel",
            Map.of(
                "abi", Integer.toString(NativeKernelLibrary.ABI_VERSION),
                "boundary", "panama-ffm",
                "transformer", "java")));
    return new BackendDiagnostics("rust-ffm", PLAN_VERSION, environment, optimizations);
  }
}
