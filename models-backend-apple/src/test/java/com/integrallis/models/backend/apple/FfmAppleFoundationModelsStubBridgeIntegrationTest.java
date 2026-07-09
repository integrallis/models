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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assumptions.assumeThat;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class FfmAppleFoundationModelsStubBridgeIntegrationTest {

  @Test
  void loadsNativeStubThroughFfmBridge() throws IOException, InterruptedException {
    assumeThat(isPosix()).as("native stub build script requires a POSIX shell").isTrue();
    assumeThat(hasCompiler()).as("native stub build script requires cc").isTrue();

    Path projectDir = Path.of(System.getProperty("user.dir"));
    Path moduleDir =
        projectDir.getFileName().toString().equals("models-backend-apple")
            ? projectDir
            : projectDir.resolve("models-backend-apple");
    Path script = moduleDir.resolve("src/native/apple-foundation-models/build-stub-bridge.sh");
    assertThat(script).isRegularFile();

    Path outputDirectory = Files.createTempDirectory("jmodels-apple-stub-");
    Process process =
        new ProcessBuilder(script.toString(), outputDirectory.toString())
            .redirectErrorStream(true)
            .start();
    String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);

    assertThat(process.waitFor()).as(output).isZero();

    Path library = Path.of(output.trim().lines().reduce((first, second) -> second).orElseThrow());
    assertThat(library).isRegularFile();

    try (AppleFoundationModelsClient client =
        AppleFoundationModels.create(
            new ApplePlatform("Mac OS X", "aarch64"), NativeLibraryLocator.fixed(library))) {
      assertThat(client.availability().available()).isTrue();
      assertThat(client.availability().reason()).contains("stub");
      assertThat(client.generate("Reply with the single word hello.").text()).isEqualTo("hello");
    }
  }

  private static boolean isPosix() {
    return !System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("windows");
  }

  private static boolean hasCompiler() throws InterruptedException {
    try {
      Process process = new ProcessBuilder("cc", "--version").redirectErrorStream(true).start();
      process.getInputStream().readAllBytes();
      return process.waitFor() == 0;
    } catch (IOException e) {
      return false;
    }
  }
}
