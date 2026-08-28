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
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class Needle2ToolQualificationCliTest {
  private static final String REVISION = "1234567890abcdef1234567890abcdef12345678";

  @TempDir Path temporary;

  @Test
  void parsesAReproducibleQualificationConfiguration() throws Exception {
    Path artifact = java.nio.file.Files.writeString(temporary.resolve("needle2.cact"), "fixture");
    Path report = temporary.resolve("report.json");

    Needle2ToolQualificationCli.Configuration configuration =
        Needle2ToolQualificationCli.parse(
            new String[] {
              "--model",
              artifact.toString(),
              "--report",
              report.toString(),
              "--max-tokens",
              "192",
              "--case",
              "robot",
              "--models-revision",
              REVISION
            });

    assertThat(configuration.model()).isEqualTo(artifact);
    assertThat(configuration.report()).isEqualTo(report);
    assertThat(configuration.maxTokens()).isEqualTo(192);
    assertThat(configuration.caseId()).isEqualTo("robot");
    assertThat(configuration.modelsRevision()).isEqualTo(REVISION);
  }

  @Test
  void defaultsToThePublishedNeedleGenerationLimit() throws Exception {
    Path artifact = java.nio.file.Files.writeString(temporary.resolve("needle2.cact"), "fixture");

    assertThat(
            Needle2ToolQualificationCli.parse(
                    new String[] {"--model", artifact.toString(), "--models-revision", REVISION})
                .maxTokens())
        .isEqualTo(256);
    assertThat(
            Needle2ToolQualificationCli.parse(
                    new String[] {"--model", artifact.toString(), "--models-revision", REVISION})
                .caseId())
        .isNull();
  }

  @Test
  void identifiesAnUnpackagedQualificationByItsExactSourceRevision() {
    assertThat(Needle2ToolQualificationCli.implementationVersion(REVISION))
        .isEqualTo("models@" + REVISION);
  }

  @Test
  void rejectsMissingArtifactsAndUnsafeTokenLimits() {
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                Needle2ToolQualificationCli.parse(
                    new String[] {
                      "--model",
                      temporary.resolve("missing.cact").toString(),
                      "--models-revision",
                      REVISION
                    }));

    assertThatIllegalArgumentException()
        .isThrownBy(
            () -> {
              Path artifact =
                  java.nio.file.Files.writeString(temporary.resolve("needle2.cact"), "fixture");
              Needle2ToolQualificationCli.parse(
                  new String[] {
                    "--model",
                    artifact.toString(),
                    "--max-tokens",
                    "16",
                    "--models-revision",
                    REVISION
                  });
            });

    assertThatIllegalArgumentException()
        .isThrownBy(
            () -> {
              Path artifact =
                  java.nio.file.Files.writeString(temporary.resolve("revision.cact"), "fixture");
              Needle2ToolQualificationCli.parse(new String[] {"--model", artifact.toString()});
            });
  }

  @Test
  void serializesAndAtomicallyCheckpointsPartialProgressWithoutOptionalJacksonModules() {
    var summary = new Needle2ToolQualification.Summary(1, 0, 0, 0, 0, 0, 0, 0, 1, false, "FAIL");
    var report =
        new Needle2ToolQualificationCli.Report(
            1,
            "2026-08-27T12:00:00Z",
            "policy",
            REVISION,
            "model_id",
            "Model",
            "pure-java",
            "development",
            "a".repeat(64),
            1,
            new Needle2ToolQualificationCli.SuiteIdentity(
                "suite", "b".repeat(64), "https://example.test/repo", REVISION, "suite.json"),
            new Needle2ToolQualificationCli.GenerationControls(0, 256, "needle2"),
            null,
            null,
            13,
            false,
            summary,
            List.of());

    Path output = temporary.resolve("partial-report.json");
    ObjectMapper mapper = new ObjectMapper();

    assertThatCode(() -> Needle2ToolQualificationCli.write(output, mapper, report))
        .doesNotThrowAnyException();
    assertThatCode(
            () -> {
              var persisted = mapper.readTree(output.toFile());
              assertThat(persisted.path("createdAt").asText()).isEqualTo("2026-08-27T12:00:00Z");
              assertThat(persisted.path("plannedAttempts").asInt()).isEqualTo(13);
              assertThat(persisted.path("complete").asBoolean()).isFalse();
            })
        .doesNotThrowAnyException();
  }
}
