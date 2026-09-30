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

import static org.assertj.core.api.Assertions.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RouterSelectionEvaluationCliTest {
  @TempDir Path directory;
  private final ObjectMapper json = new ObjectMapper();

  @Test
  void scoresReferenceAnswersAndReportsBaselinesAblationsAndFailedAttemptCost() throws Exception {
    Path manifest = fixture();
    var report = json.valueToTree(RouterSelectionEvaluationCli.evaluate(manifest));
    assertThat(report.path("mode").asText()).isEqualTo("paired-observation-replay");
    assertThat(report.path("rows").size()).isEqualTo(16);
    assertThat(report.path("summaries").has("static-best-calibration")).isTrue();
    assertThat(report.path("ablationChangedSelections").size()).isEqualTo(3);
    var staticCheap = report.path("summaries").path("static-cheapest").path("dataset-one/math");
    assertThat(staticCheap.path("quality").asDouble()).isZero();
    assertThat(staticCheap.path("cost").asDouble()).isEqualTo(0.2);
    assertThat(staticCheap.path("failedOrRejected").asInt()).isEqualTo(1);
    Path output = directory.resolve("report.json");
    RouterSelectionEvaluationCli.run(new String[] {manifest.toString(), output.toString()});
    assertThat(Files.readString(output)).contains("manifestSha256", "calibrationCandidates");
    assertThatThrownBy(
            () ->
                RouterSelectionEvaluationCli.run(
                    new String[] {manifest.toString(), output.toString()}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("overwrite");
  }

  @Test
  void refusesDigestMismatchMissingPairsAndCalibrationLeakage() throws Exception {
    Path manifest = fixture();
    Path cases = directory.resolve("cases.json");
    Files.writeString(cases, "[]");
    Path mismatchedManifest = manifest;
    assertThatThrownBy(() -> RouterSelectionEvaluationCli.evaluate(mismatchedManifest))
        .hasMessageContaining("digest mismatch");
    manifest = fixture();
    ArrayNode observations =
        (ArrayNode) json.readTree(directory.resolve("observations.json").toFile());
    observations.remove(0);
    json.writeValue(directory.resolve("observations.json").toFile(), observations);
    repin(manifest, "observations", "observations.json");
    Path pairedManifest = manifest;
    assertThatThrownBy(() -> RouterSelectionEvaluationCli.evaluate(pairedManifest))
        .hasMessageContaining("missing paired");
    manifest = fixture();
    ArrayNode rows = (ArrayNode) json.readTree(cases.toFile());
    ((ObjectNode) rows.get(1)).put("prompt", rows.get(0).path("prompt").asText());
    json.writeValue(cases.toFile(), rows);
    repin(manifest, "cases", "cases.json");
    Path leakyManifest = manifest;
    assertThatThrownBy(() -> RouterSelectionEvaluationCli.evaluate(leakyManifest))
        .hasMessageContaining("leakage");
  }

  private Path fixture() throws Exception {
    ArrayNode cases = json.createArrayNode(), observations = json.createArrayNode();
    for (String dataset : List.of("dataset-one", "dataset-two")) {
      for (String split : List.of("calibration", "eval")) {
        String id = dataset + split;
        var item = cases.addObject();
        item.put("id", id)
            .put("dataset", dataset)
            .put("split", split)
            .put("prompt", "fixture only: " + id)
            .put("task", "math")
            .put("predictedTask", "math")
            .put("classifierMillis", 1)
            .put("inputBound", 100)
            .put("outputBound", 100)
            .put("budget", 1)
            .put("deadlineMillis", 1000);
        item.putArray("answers").add("correct");
        for (String model : List.of("cheap", "best")) {
          boolean success = !(model.equals("cheap") && split.equals("eval"));
          observations
              .addObject()
              .put("caseId", id)
              .put("modelId", model)
              .put("success", success)
              .put("latencyMillis", 10)
              .put("ttftMillis", 2)
              .put("response", model.equals("best") ? "correct" : "incorrect")
              .put("receipt", "synthetic-unit-fixture")
              .put("inputTokens", 10)
              .put("outputTokens", 10);
        }
      }
    }
    json.writeValue(directory.resolve("cases.json").toFile(), cases);
    json.writeValue(directory.resolve("observations.json").toFile(), observations);
    ObjectNode manifest = json.createObjectNode().put("schemaVersion", 1);
    for (String field :
        List.of(
            "protocolRevision",
            "modelsRevision",
            "classifierRevision",
            "hardware",
            "runtime",
            "generationSettings")) manifest.put(field, "synthetic-fixture-only");
    for (String dataset : List.of("dataset-one", "dataset-two"))
      manifest
          .withArray("datasets")
          .addObject()
          .put("id", dataset)
          .put("source", "https://example.invalid/fixture")
          .put("revision", "test-only")
          .put("license", "test-only");
    for (String model : List.of("cheap", "best")) {
      var descriptor =
          manifest
              .withArray("models")
              .addObject()
              .put("id", model)
              .put("revision", "test-only")
              .put("clientRevision", "test-only")
              .put("local", false)
              .put("contextWindow", 2048)
              .put("inputPrice", model.equals("cheap") ? 1000 : 2000)
              .put("outputPrice", model.equals("cheap") ? 1000 : 2000);
      descriptor.putArray("capabilities");
    }
    Path path = directory.resolve("manifest.json");
    json.writeValue(path.toFile(), manifest);
    repin(path, "cases", "cases.json");
    repin(path, "observations", "observations.json");
    return path;
  }

  private void repin(Path manifest, String field, String file) throws Exception {
    ObjectNode root = (ObjectNode) json.readTree(manifest.toFile());
    root.putObject(field)
        .put("path", file)
        .put(
            "sha256",
            HexFormat.of()
                .formatHex(
                    MessageDigest.getInstance("SHA-256")
                        .digest(Files.readAllBytes(directory.resolve(file)))));
    json.writeValue(manifest.toFile(), root);
  }
}
