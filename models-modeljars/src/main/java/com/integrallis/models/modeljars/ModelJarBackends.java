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
package com.integrallis.models.modeljars;

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.semanticorder.WordTour;
import java.io.ByteArrayInputStream;
import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarException;
import org.modeljars.ModelJarResourceLoader;
import org.modeljars.ModelPerformanceProfileRegistry;

/** Optional bridge from ModelJars descriptors and measured profiles to Models backends. */
public final class ModelJarBackends {

  private ModelJarBackends() {}

  /** Selects safe recommendations for this exact artifact, backend, and running JVM. */
  public static BackendConfiguration configuration(ModelJarDescriptor descriptor, String backend) {
    return configuration(
        descriptor,
        backend,
        ModelPerformanceProfileRegistry.fromClasspath(),
        RuntimeFingerprint.capture().asEnvironment(),
        ManagementFactory.getRuntimeMXBean().getInputArguments());
  }

  static BackendConfiguration configuration(
      ModelJarDescriptor descriptor,
      String backend,
      ModelPerformanceProfileRegistry registry,
      Map<String, String> runtime,
      List<String> inputArguments) {
    requireBackend(descriptor, backend);
    return ModelJarProfileSelector.select(
        descriptor, registry, runtime, inputArguments, backend.trim().toLowerCase());
  }

  /** Loads a GGUF descriptor through the pure-Java backend and its exact measured profile. */
  public static PureJavaBackend loadPureJava(ModelJarDescriptor descriptor) {
    Path modelPath = requireGgufPath(descriptor, "pure-java");
    return PureJavaBackend.load(modelPath, configuration(descriptor, "pure-java"));
  }

  /** Loads a GGUF descriptor through the bundled Rust/FFM backend and its measured profile. */
  public static RustFfmBackend loadRustFfm(ModelJarDescriptor descriptor) {
    Path modelPath = requireGgufPath(descriptor, "rust-ffm");
    return RustFfmBackend.load(modelPath, configuration(descriptor, "rust-ffm"));
  }

  /** Loads a GGUF descriptor through an explicit Rust library and its measured profile. */
  public static RustFfmBackend loadRustFfm(ModelJarDescriptor descriptor, Path nativeLibrary) {
    Path modelPath = requireGgufPath(descriptor, "rust-ffm");
    return RustFfmBackend.load(
        modelPath,
        Objects.requireNonNull(nativeLibrary, "nativeLibrary"),
        configuration(descriptor, "rust-ffm"));
  }

  /** Loads and verifies a bundled WordTour payload from a ModelJars marker. */
  public static WordTour loadWordTour(ModelJarDescriptor descriptor) {
    Objects.requireNonNull(descriptor, "descriptor");
    requireBackend(descriptor, "semantic-order");
    if (!"wordtour-v1".equals(descriptor.format())) {
      throw new ModelJarException(
          "WordTour only supports wordtour-v1 ModelJars descriptors: " + descriptor.alias());
    }
    ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
    if (classLoader == null) {
      classLoader = ModelJarBackends.class.getClassLoader();
    }
    byte[] payload = new ModelJarResourceLoader(classLoader).readVerified(descriptor);
    return WordTour.load(new ByteArrayInputStream(payload));
  }

  private static Path requireGgufPath(ModelJarDescriptor descriptor, String backend) {
    requireBackend(descriptor, backend);
    if (!"gguf".equals(descriptor.format())) {
      throw new ModelJarException(
          "Models GGUF backends do not support format "
              + descriptor.format()
              + ": "
              + descriptor.alias());
    }
    return descriptor
        .localPath()
        .orElseThrow(
            () ->
                new ModelJarException(
                    "ModelJars descriptor has no local path: " + descriptor.alias()));
  }

  private static void requireBackend(ModelJarDescriptor descriptor, String backend) {
    Objects.requireNonNull(descriptor, "descriptor");
    if (backend == null || backend.isBlank()) {
      throw new IllegalArgumentException("backend must not be blank");
    }
    String normalized = backend.trim().toLowerCase();
    if (!descriptor.supportsBackend(normalized)) {
      throw new ModelJarException(
          "ModelJars descriptor does not support "
              + normalized
              + " backend: "
              + descriptor.alias());
    }
  }
}
