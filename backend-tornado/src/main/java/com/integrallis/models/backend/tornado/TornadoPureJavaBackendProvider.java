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
import com.integrallis.models.backend.purejava.spi.PureJavaBackendProvider;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/** Service-loaded Tornado provider used by Models and ModelJars automatic backend loading. */
public final class TornadoPureJavaBackendProvider implements PureJavaBackendProvider {

  @Override
  public Optional<PureJavaBackend> tryLoad(
      Path modelPath, BackendConfiguration backendConfiguration) {
    Path model = modelPath.toAbsolutePath().normalize();
    if (!Files.isRegularFile(model)) {
      return Optional.empty();
    }
    List<AcceleratorEligibility.DeviceCapabilities> devices;
    try {
      devices = TornadoRuntimeDevices.discover();
    } catch (LinkageError | RuntimeException failure) {
      return Optional.empty();
    }
    long modelBytes;
    try {
      modelBytes = Files.size(model);
    } catch (IOException failure) {
      return Optional.empty();
    }
    TornadoBackendOptions options = TornadoBackendOptions.defaults();
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(devices, modelBytes, options.accelerateDecode());
    if (!decision.eligible()) {
      return Optional.empty();
    }
    TornadoBackendOptions required =
        new TornadoBackendOptions(
            options.accelerateDecode(),
            options.eagerReadiness(),
            true,
            options.executionBatchSize());
    TornadoBackendRuntime runtime = TornadoBackend.open(model, backendConfiguration, required);
    return Optional.of(runtime.detachBackend());
  }
}
