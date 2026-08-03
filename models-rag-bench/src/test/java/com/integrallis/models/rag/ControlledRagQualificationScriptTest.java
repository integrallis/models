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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class ControlledRagQualificationScriptTest {

  @Test
  void gatesTunedQualificationOnLibraryDefaultCorrectness()
      throws IOException {
    String script = Files.readString(qualificationScript());

    assertThat(script)
        .contains("DEFAULT_CORRECTNESS_DIR=\"$OUTPUT_DIR/default-correctness\"")
        .contains("--warmups 0")
        .contains("--iterations 1")
        .contains(".settings.generationControls.promptCache == \"longest-common-prefix\"")
        .contains(".summary.successfulAttempts == .summary.totalAttempts")
        .contains("and (.failures | length) == 0")
        .contains("tuningSystemProperties: []")
        .contains("RAG_MODELS_BACKEND must be pure-java or rust-ffm")
        .contains("for option_source in JAVA_OPTS JAVA_TOOL_OPTIONS JDK_JAVA_OPTIONS _JAVA_OPTIONS")
        .contains("must not contain Models tuning properties during qualification");

    int defaultRun = script.indexOf("--output \"$DEFAULT_REPORT\"");
    int tunedProperty = script.indexOf("-Dmodels.native.quantizedDecode=true");
    int tunedRun = script.indexOf("--output \"$OUTPUT_DIR/models-$MODELS_BACKEND.json\"");
    assertThat(defaultRun).isPositive();
    assertThat(tunedProperty).isGreaterThan(defaultRun);
    assertThat(tunedRun).isGreaterThan(tunedProperty);
  }

  @Test
  void permitsTunedNativeDecodeThreadCountWithoutChangingComparatorThreads() throws IOException {
    String script = Files.readString(qualificationScript());

    assertThat(script)
        .contains("NATIVE_THREADS=${RAG_NATIVE_THREADS:-$THREADS}")
        .contains("-Dmodels.native.kernels.threads=$NATIVE_THREADS")
        .contains("-XX:ActiveProcessorCount=$THREADS")
        .contains("--threads \"$THREADS\"");
  }

  private static Path qualificationScript() {
    Path repositoryRoot =
        Path.of(System.getProperty("models.repositoryRoot", ".")).toAbsolutePath().normalize();
    return repositoryRoot.resolve("scripts/run-controlled-rag-qualification.sh");
  }
}
