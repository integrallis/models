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
package com.integrallis.models.backend.tornado;

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/** Loads the qualified Java/Tornado Q4 backend when possible and otherwise uses the Vector API. */
public final class TornadoBackend {
  private static final System.Logger LOGGER = System.getLogger(TornadoBackend.class.getName());

  private TornadoBackend() {}

  /** Opens a model with automatic device selection, eager readiness, and safe CPU fallback. */
  public static TornadoBackendRuntime open(Path modelPath) {
    return open(modelPath, BackendConfiguration.empty(), TornadoBackendOptions.defaults());
  }

  /** Opens a model using explicit accelerator readiness and fallback controls. */
  public static TornadoBackendRuntime open(Path modelPath, TornadoBackendOptions options) {
    return open(modelPath, BackendConfiguration.empty(), options);
  }

  /** Opens a model using its qualified backend configuration and accelerator controls. */
  public static TornadoBackendRuntime open(
      Path modelPath, BackendConfiguration backendConfiguration, TornadoBackendOptions options) {
    Objects.requireNonNull(modelPath, "modelPath");
    Objects.requireNonNull(backendConfiguration, "backendConfiguration");
    Objects.requireNonNull(options, "options");
    Path model = modelPath.toAbsolutePath().normalize();
    if (!Files.isRegularFile(model)) {
      throw new IllegalArgumentException("modelPath must be a regular model file: " + model);
    }
    long modelBytes = size(model);
    List<AcceleratorEligibility.DeviceCapabilities> devices;
    try {
      devices = TornadoRuntimeDevices.discover();
    } catch (LinkageError | RuntimeException failure) {
      return fallback(
          model,
          backendConfiguration,
          options,
          "TornadoVM runtime unavailable: " + failure.getClass().getSimpleName(),
          0);
    }
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(devices, modelBytes, options.accelerateDecode());
    if (!decision.eligible()) {
      return fallback(
          model, backendConfiguration, options, decision.reason(), decision.requiredBytes());
    }

    TornadoGgufBatchedMatrixKernel kernel =
        new TornadoGgufBatchedMatrixKernel(
            options.executionBatchSize(), options.accelerateDecode());
    PureJavaBackend backend = null;
    try {
      backend = PureJavaBackend.load(model, backendConfiguration, kernel);
      Duration readiness = options.eagerReadiness() ? prepare(backend, options) : Duration.ZERO;
      if (options.eagerReadiness() && kernel.calls() == 0) {
        backend.close();
        return fallback(
            model,
            backendConfiguration,
            options,
            "model has no eligible Q4_0 projections",
            decision.requiredBytes());
      }
      String device = decision.device().name();
      LOGGER.log(
          System.Logger.Level.INFO,
          "Models accelerator selected {0}; readiness={1} ms plans={2}",
          device,
          readiness.toMillis(),
          kernel.projectionPlanCount());
      return new TornadoBackendRuntime(
          backend,
          new TornadoBackendStatus(true, device, "eligible", decision.requiredBytes(), readiness));
    } catch (LinkageError | RuntimeException failure) {
      if (backend != null) {
        try {
          backend.close();
        } catch (RuntimeException closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      if (options.requireAccelerator()) {
        throw new IllegalStateException("qualified accelerator initialization failed", failure);
      }
      return fallback(
          model,
          backendConfiguration,
          options,
          "accelerator initialization failed: " + failure.getClass().getSimpleName(),
          decision.requiredBytes());
    }
  }

  private static Duration prepare(PureJavaBackend backend, TornadoBackendOptions options) {
    long started = System.nanoTime();
    int token = readinessToken(backend);
    int[] tokens = new int[options.executionBatchSize()];
    Arrays.fill(tokens, token);
    backend.prefill(tokens, 0);
    if (options.accelerateDecode()) {
      backend.forward(token, tokens.length);
    }
    backend.reset();
    return Duration.ofNanos(System.nanoTime() - started);
  }

  private static int readinessToken(PureJavaBackend backend) {
    int token = backend.tokenizer().bosToken();
    return token >= 0 && token < backend.metadata().vocabSize() ? token : 0;
  }

  private static TornadoBackendRuntime fallback(
      Path model,
      BackendConfiguration backendConfiguration,
      TornadoBackendOptions options,
      String reason,
      long requiredBytes) {
    if (options.requireAccelerator()) {
      throw new IllegalStateException("accelerator required but unavailable: " + reason);
    }
    LOGGER.log(
        System.Logger.Level.INFO, "Models accelerator unavailable; using Vector API: {0}", reason);
    return new TornadoBackendRuntime(
        PureJavaBackend.load(model, backendConfiguration),
        new TornadoBackendStatus(false, "Vector API", reason, requiredBytes, Duration.ZERO));
  }

  private static long size(Path model) {
    try {
      return Files.size(model);
    } catch (IOException exception) {
      throw new UncheckedIOException("could not read model size: " + model, exception);
    }
  }
}
