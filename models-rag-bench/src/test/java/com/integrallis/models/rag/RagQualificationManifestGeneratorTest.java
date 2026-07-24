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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RagQualificationManifestGeneratorTest {
  private static final String CERTIFIED_REPORT =
      "benchmark-results/certified-20260724/rag/launch-qualification/"
          + "qwen3-0.6b-llama-grounded.json";

  private final ObjectMapper mapper = new ObjectMapper();

  @Test
  void generatesAuditableManifestAndEnforcesDistinctModelTarget(@TempDir Path reports)
      throws Exception {
    ObjectNode qualified = (ObjectNode) mapper.readTree(certifiedReport().toFile());
    ObjectNode diagnostics = (ObjectNode) qualified.path("backendDiagnostics");
    diagnostics.put("backend", "llama.cpp");
    diagnostics.put("planVersion", "local-http-v1");
    ((ObjectNode) qualified.path("settings"))
        .put("groundingPolicy", GroundedAnswerPolicy.POLICY_ID);
    mapper
        .writerWithDefaultPrettyPrinter()
        .writeValue(reports.resolve("qualified.json").toFile(), qualified);

    ObjectNode rejected = qualified.deepCopy();
    rejected.put("modelId", "unshipped-model");
    ((ObjectNode) rejected.path("backendDiagnostics")).put("planVersion", "unavailable");
    mapper
        .writerWithDefaultPrettyPrinter()
        .writeValue(reports.resolve("rejected.json").toFile(), rejected);

    RagQualificationManifest manifest =
        RagQualificationManifestGenerator.generate(reports, "models-revision", 1);

    assertThat(manifest.qualifiedModels()).isEqualTo(1);
    assertThat(manifest.rejectedModels()).isEqualTo(1);
    assertThat(manifest.entries())
        .filteredOn(RagQualificationManifestEntry::qualified)
        .singleElement()
        .satisfies(
            entry -> {
              assertThat(entry.modelId()).isEqualTo("qwen3-0.6b-q4_0");
              assertThat(entry.reportSha256()).hasSize(64);
              assertThat(entry.rawCorrectAnswerRate()).isBetween(0.0, 1.0);
              assertThat(entry.extractiveFallbackRate()).isBetween(0.0, 1.0);
            });
    manifest.requireTarget();

    RagQualificationManifest impossible =
        RagQualificationManifestGenerator.generate(reports, "models-revision", 2);
    assertThatThrownBy(impossible::requireTarget)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("1 distinct models qualified; target is 2");
  }

  @Test
  void commandWritesManifestBeforeFailingAnUnmetTarget(@TempDir Path directory) throws Exception {
    Path reports = Files.createDirectory(directory.resolve("reports"));
    ObjectNode rejected = (ObjectNode) mapper.readTree(certifiedReport().toFile());
    rejected.put("modelId", "unshipped-model");
    Path input = reports.resolve("rejected.json");
    mapper.writerWithDefaultPrettyPrinter().writeValue(input.toFile(), rejected);
    Path output = directory.resolve("manifest.json");

    assertThatThrownBy(
            () ->
                RagQualificationManifestCli.run(
                    new String[] {
                      "--reports",
                      reports.toString(),
                      "--output",
                      output.toString(),
                      "--models-revision",
                      "revision",
                      "--target",
                      "1"
                    }))
        .isInstanceOf(IllegalStateException.class);
    assertThat(output).isRegularFile();
    assertThat(mapper.readTree(output.toFile()).path("qualifiedModels").asInt()).isZero();
  }

  private static Path certifiedReport() {
    Path current = Path.of("").toAbsolutePath();
    while (current != null) {
      Path candidate = current.resolve(CERTIFIED_REPORT);
      if (Files.isRegularFile(candidate)) {
        return candidate;
      }
      current = current.getParent();
    }
    throw new IllegalStateException("certified report not found: " + CERTIFIED_REPORT);
  }
}
