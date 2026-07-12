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

import java.nio.file.Path;
import java.util.Optional;

@FunctionalInterface
interface NativeLibraryLocator {

  String LIBRARY_PATH_PROPERTY = "models.apple.foundation.library";
  String LIBRARY_PATH_ENV = "MODELS_APPLE_FOUNDATION_LIBRARY";

  Optional<Path> locate();

  static NativeLibraryLocator system() {
    return () -> {
      String configured = System.getProperty(LIBRARY_PATH_PROPERTY);
      if (configured == null || configured.isBlank()) {
        configured = System.getenv(LIBRARY_PATH_ENV);
      }
      if (configured == null || configured.isBlank()) {
        return Optional.empty();
      }
      return Optional.of(Path.of(configured));
    };
  }

  static NativeLibraryLocator fixed(Path path) {
    return () -> Optional.of(path);
  }

  static NativeLibraryLocator empty() {
    return Optional::empty;
  }
}
