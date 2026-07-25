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
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarException;

/**
 * GGUF backend that keeps transformer execution in Java and delegates qualified matrix kernels to
 * Models-owned Rust code through FFM.
 */
public final class RustFfmBackend implements SpeculativeInferenceBackend {
  public static final String LIBRARY_PATH_PROPERTY = "models.native.kernels.library";
  public static final String LIBRARY_PATH_ENV = "MODELS_NATIVE_KERNELS_LIBRARY";
  public static final String PLAN_VERSION = "rust-ffm-v5";

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
    return new RustFfmBackend(engine, diagnostics(engine.diagnostics(), kernel));
  }

  /** Loads a ModelJar descriptor using its exact Rust/FFM performance profile. */
  public static RustFfmBackend load(ModelJarDescriptor descriptor) {
    String configured = System.getProperty(LIBRARY_PATH_PROPERTY);
    if (configured == null || configured.isBlank()) {
      configured = System.getenv(LIBRARY_PATH_ENV);
    }
    if (configured == null || configured.isBlank()) {
      return load(descriptor, BundledNativeKernelLibrary.resolve());
    }
    return load(descriptor, Path.of(configured));
  }

  /** Loads a ModelJar descriptor with an explicit Models native-kernel platform library. */
  public static RustFfmBackend load(ModelJarDescriptor descriptor, Path libraryPath) {
    Objects.requireNonNull(descriptor, "descriptor");
    Objects.requireNonNull(libraryPath, "libraryPath");
    if (!descriptor.supportsBackend("rust-ffm")) {
      throw new ModelJarException(
          "ModelJars descriptor does not support rust-ffm backend: " + descriptor.alias());
    }
    RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(libraryPath);
    PureJavaBackend engine = PureJavaBackend.load(descriptor, "rust-ffm", kernel);
    return new RustFfmBackend(engine, diagnostics(engine.diagnostics(), kernel));
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
      BackendDiagnostics javaDiagnostics, RustGgufBatchedMatrixKernel kernel) {
    Map<String, String> environment = new LinkedHashMap<>(javaDiagnostics.environment());
    environment.put("transformer-runtime", "java");
    environment.put("kernel-runtime", "rust-ffm");
    environment.put("kernel-implementation", kernel.implementation());
    environment.put("native-kernel-abi", Integer.toString(NativeKernelLibrary.ABI_VERSION));
    environment.put("native-quantized-decode", Boolean.toString(kernel.nativeDecodeEnabled()));
    List<OptimizationDecision> optimizations = new ArrayList<>(javaDiagnostics.optimizations());
    optimizations.add(
        nativeQuantizedDecision(
            "rust-q4-0-batched-matmul",
            "eligible Q4_0 batched and grouped projections execute in the Models Rust kernel"));
    optimizations.add(
        nativeQuantizedDecision(
            "rust-q5-0-batched-matmul",
            "eligible Q5_0 batched and grouped projections execute in the Models Rust kernel"));
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
