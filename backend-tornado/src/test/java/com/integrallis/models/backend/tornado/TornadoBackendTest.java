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

import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.models.api.BackendConfiguration;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TornadoBackendTest {

  @Test
  void rejectsAMissingModelBeforeInspectingAcceleratorDrivers(@TempDir Path directory) {
    Path missing = directory.resolve("missing.gguf");

    assertThatIllegalArgumentException()
        .isThrownBy(() -> TornadoBackend.open(missing))
        .withMessageContaining("regular model file");
  }

  @Test
  void validatesConfigurationBeforeInspectingAcceleratorDrivers(@TempDir Path directory)
      throws Exception {
    Path placeholder = Files.createFile(directory.resolve("model.gguf"));

    assertThatNullPointerException()
        .isThrownBy(() -> TornadoBackend.open(placeholder, null, TornadoBackendOptions.defaults()))
        .withMessage("backendConfiguration");
    assertThatNullPointerException()
        .isThrownBy(() -> TornadoBackend.open(placeholder, BackendConfiguration.empty(), null))
        .withMessage("options");
  }
}
