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
package com.integrallis.models.backend.apple;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

/** Factory for the optional Apple Foundation Models bridge. */
public final class AppleFoundationModels {

  /**
   * System property that controls the bridge mode. The supported non-default value is {@code stub}.
   */
  public static final String MODE_PROPERTY = "models.apple.foundation.mode";

  /**
   * Environment variable that controls the bridge mode. The supported non-default value is {@code
   * stub}.
   */
  public static final String MODE_ENV = "MODELS_APPLE_FOUNDATION_MODE";

  private static final String STUB_MODE = "stub";

  private AppleFoundationModels() {}

  /**
   * Creates an Apple Foundation Models client.
   *
   * <p>The native bridge path can be supplied with the {@code models.apple.foundation.library}
   * system property or {@code MODELS_APPLE_FOUNDATION_LIBRARY} environment variable.
   *
   * <p>For local development on machines without Apple Intelligence, set {@code
   * models.apple.foundation.mode=stub} or {@code MODELS_APPLE_FOUNDATION_MODE=stub}.
   */
  public static AppleFoundationModelsClient create() {
    if (stubModeEnabled()) {
      return createStub();
    }
    return create(ApplePlatform.current(), NativeLibraryLocator.system());
  }

  /**
   * Creates a deterministic stub client that simulates an available Apple Foundation Models
   * runtime.
   */
  public static AppleFoundationModelsClient createStub() {
    return AppleFoundationModelsClient.of(new StubAppleFoundationModelsBridge());
  }

  static AppleFoundationModelsClient create(
      ApplePlatform platform, NativeLibraryLocator libraryLocator) {
    Objects.requireNonNull(platform, "platform");
    Objects.requireNonNull(libraryLocator, "libraryLocator");

    AppleFoundationModelsAvailability platformSupport = platform.foundationModelsSupport();
    if (!platformSupport.supported()) {
      return AppleFoundationModelsClient.unavailable(platformSupport);
    }

    Optional<Path> nativeLibrary = libraryLocator.locate();
    if (nativeLibrary.isEmpty()) {
      return AppleFoundationModelsClient.unavailable(platformSupport);
    }

    Path libraryPath = nativeLibrary.get();
    if (!Files.isRegularFile(libraryPath)) {
      return AppleFoundationModelsClient.unavailable(
          AppleFoundationModelsAvailability.unavailable(
              "Apple Foundation Models native bridge not found: " + libraryPath));
    }

    try {
      return AppleFoundationModelsClient.of(FfmAppleFoundationModelsBridge.open(libraryPath));
    } catch (RuntimeException | LinkageError e) {
      return AppleFoundationModelsClient.unavailable(
          AppleFoundationModelsAvailability.unavailable(
              "Apple Foundation Models native bridge failed to load: " + e.getMessage()));
    }
  }

  private static boolean stubModeEnabled() {
    String mode = System.getProperty(MODE_PROPERTY);
    if (mode == null || mode.isBlank()) {
      mode = System.getenv(MODE_ENV);
    }
    return mode != null && STUB_MODE.equalsIgnoreCase(mode.trim());
  }
}
