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

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class JvmLaunchSupportProcessIntegrationTest {

  @Test
  void failsFastInARealC1OnlyJvmInsteadOfStartingModelInference() throws Exception {
    Path java = Path.of(System.getProperty("java.home"), "bin", "java");
    Process process =
        new ProcessBuilder(
                java.toString(),
                "--add-modules",
                "jdk.incubator.vector",
                "-XX:TieredStopAtLevel=1",
                "-cp",
                System.getProperty("java.class.path"),
                C1OnlyModelLoadProbe.class.getName())
            .redirectErrorStream(true)
            .start();

    assertThat(process.waitFor(Duration.ofSeconds(10).toMillis(), TimeUnit.MILLISECONDS)).isTrue();
    String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);

    assertThat(process.exitValue()).isNotZero();
    assertThat(output)
        .contains("-XX:TieredStopAtLevel=1")
        .contains("optimizedLaunch = false")
        .doesNotContain("NoSuchFileException");
  }

  static final class C1OnlyModelLoadProbe {
    public static void main(String[] args) {
      PureJavaBackend.load(Path.of("model-loading-must-not-start.gguf"));
    }
  }
}
