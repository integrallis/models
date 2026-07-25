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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.VectorizationProvider;
import java.nio.file.Files;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;

final class ModelOracleTestSupport {

  private ModelOracleTestSupport() {}

  static ModelJarDescriptor installedDescriptor(
      ModelJarRequirement requirement, String fixtureTask) {
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath().resolve(requirement).orElseThrow();
    assertThat(Files.exists(descriptor.localPath().orElseThrow()))
        .as("%s must be present. Run %s.", descriptor.localPath().orElseThrow(), fixtureTask)
        .isTrue();
    return descriptor;
  }

  static void assertPanamaEnabled() {
    assertThat(VectorizationProvider.isPanamaEnabled())
        .as(
            "model token oracle requires Panama SIMD; provider=%s, panamaFailure=%s",
            VectorizationProvider.getProviderName(),
            VectorizationProvider.getPanamaFailure().map(Throwable::toString).orElse("none"))
        .isTrue();
  }

  static int[] greedyTokens(PureJavaBackend backend, int[] promptTokens, int count) {
    backend.reset();
    float[] logits = null;
    int position = 0;
    for (int token : promptTokens) {
      logits = backend.forward(token, position++);
    }

    int[] generated = new int[count];
    for (int index = 0; index < count; index++) {
      int token = argmax(logits);
      generated[index] = token;
      if (index + 1 < count) {
        logits = backend.forward(token, position++);
      }
    }
    return generated;
  }

  static void restoreSystemProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }
}
