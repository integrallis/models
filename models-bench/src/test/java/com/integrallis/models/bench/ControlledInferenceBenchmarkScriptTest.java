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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class ControlledInferenceBenchmarkScriptTest {

  @Test
  void passesArtifactsOnlyToExternalBackends() throws IOException {
    String script = Files.readString(benchmarkScript());

    String commonArguments = section(script, "COMMON_ARGS=(", "\n)\n\n");
    String pureJavaInvocation = section(script, "--backend pure-java", "if ! curl");
    String ollamaInvocation = section(script, "--backend ollama", "ollama stop");
    String llamaCppInvocation =
        section(script, "--backend llama.cpp", "\"$BENCHMARK_CLI\" compare");

    assertThat(commonArguments).doesNotContain("--artifact");
    assertThat(pureJavaInvocation).doesNotContain("--artifact");
    assertThat(ollamaInvocation).contains("--artifact \"$MODEL_PATH\"");
    assertThat(llamaCppInvocation).contains("--artifact \"$MODEL_PATH\"");
  }

  private static Path benchmarkScript() {
    Path fromModule = Path.of("..", "scripts", "run-controlled-inference-benchmarks.sh");
    return Files.isRegularFile(fromModule)
        ? fromModule
        : Path.of("scripts", "run-controlled-inference-benchmarks.sh");
  }

  private static String section(String script, String startMarker, String endMarker) {
    int start = script.indexOf(startMarker);
    int end = script.indexOf(endMarker, start);
    assertThat(start).as("start marker %s", startMarker).isGreaterThanOrEqualTo(0);
    assertThat(end).as("end marker %s", endMarker).isGreaterThan(start);
    return script.substring(start, end);
  }
}
