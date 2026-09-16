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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.ToolApplicabilityHead;
import com.integrallis.models.runtime.ToolApplicabilityScore;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Runs the frozen V16 hidden-state applicability gate in the production Java Q4 runtime. */
final class ActivatedHiddenDecisionProfileCli {
  static final String EXPERIMENT = "qwen3-17b-hidden-applicability-v16";
  static final String SOURCE_SHA256 = ActivatedDecisionProfileCli.SOURCE_SHA256;
  static final String HEAD_FILE_SHA256 =
      "67b1f5fd231f924554d02daf450ce049fbccd24c084f7730d4ce2724ca234d1e";
  static final String HEAD_ARTIFACT_SHA256 =
      "acece5d8e0188b874ef8a919597ba7a359d876bc20fd828af5fbfab00984445a";
  static final String MODEL_SHA256 =
      "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a";
  static final String ADAPTER_SHA256 =
      "f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21";
  private static final Set<String> OPTIONS =
      Set.of("model", "adapter", "head", "records", "output", "models-revision");

  private ActivatedHiddenDecisionProfileCli() {}

  record Observation(
      String id,
      String kind,
      boolean callExpected,
      int promptTokens,
      int sharedPrefixTokens,
      long sharedPrefixBytes,
      boolean physicallyShared,
      double score,
      boolean callPredicted,
      long elapsedMillis) {}

  record Score(
      int calls,
      int noCalls,
      int correctCalls,
      int correctNoCalls,
      double callAccuracy,
      double noCallAccuracy,
      double balancedAccuracy) {}

  record Report(
      int schemaVersion,
      String experiment,
      String createdAt,
      String modelsRevision,
      String sourceSha256,
      String modelSha256,
      String headFileSha256,
      String headArtifactSha256,
      ActivatedAdapterMetadata adapter,
      BenchmarkEnvironment environment,
      Score result,
      boolean passed,
      String verdict,
      List<Observation> observations) {}

  static int run(String[] args) throws IOException {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    Path modelPath = requiredFile(values, "model");
    Path adapterPath = requiredDirectory(values, "adapter");
    Path headPath = requiredFile(values, "head");
    Path recordsPath = requiredFile(values, "records");
    Path output = Path.of(required(values, "output"));
    String revision = required(values, "models-revision");
    if (!revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact Git SHA");
    }
    requireHash(modelPath, MODEL_SHA256, "model");
    requireHash(adapterPath.resolve("adapter_model.safetensors"), ADAPTER_SHA256, "adapter");
    requireHash(headPath, HEAD_FILE_SHA256, "head");
    requireHash(recordsPath, SOURCE_SHA256, "records");

    ObjectMapper mapper = ActivatedDecisionProfileCli.mapper();
    ToolApplicabilityHead head = loadHead(mapper, headPath);
    List<ActivatedDecisionProfileCli.SourceCase> cases =
        ActivatedDecisionProfileCli.loadCases(mapper, recordsPath).cases();
    List<Observation> observations = new ArrayList<>();
    ActivatedAdapterMetadata adapter;
    PureJavaBackend backend = PureJavaBackend.loadActivatedAdapter(modelPath, adapterPath);
    ActivatedToolCallingModel activatedModel;
    try {
      activatedModel = new ActivatedToolCallingModel(backend, 1);
    } catch (RuntimeException | Error failure) {
      try {
        backend.close();
      } catch (RuntimeException | Error closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
    try (ActivatedToolCallingModel activated = activatedModel) {
      adapter = activated.adapter();
      int ordinal = 0;
      for (ActivatedDecisionProfileCli.SourceCase item : cases) {
        ordinal++;
        long started = System.nanoTime();
        try (ActivatedToolTurn turn =
            activated.openToolTurn(
                item.prompt(), ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
          ToolApplicabilityScore applicability = turn.scoreToolApplicability(head);
          long elapsedMillis = (System.nanoTime() - started) / 1_000_000L;
          Observation observation =
              new Observation(
                  item.id(),
                  item.kind(),
                  item.callExpected(),
                  backend.tokenizer().encode(item.prompt()).length,
                  turn.sharedPrefixTokens(),
                  turn.sharedPrefixBytes(),
                  turn.physicallySharesPrefix(),
                  applicability.score(),
                  applicability.shouldCall(),
                  elapsedMillis);
          observations.add(observation);
          System.out.printf(
              "%2d/%d %-18s expected=%-5s predicted=%-5s score=%11.5f shared=%s %d ms%n",
              ordinal,
              cases.size(),
              item.id(),
              item.callExpected(),
              observation.callPredicted(),
              observation.score(),
              observation.physicallyShared(),
              elapsedMillis);
        }
      }
    }

    Score score = score(observations);
    boolean passed =
        observations.size() == 75
            && observations.stream().allMatch(Observation::physicallyShared)
            && score.correctCalls() >= 47
            && score.correctNoCalls() >= 24;
    Report report =
        new Report(
            1,
            EXPERIMENT,
            Instant.now().toString(),
            revision,
            SOURCE_SHA256,
            MODEL_SHA256,
            HEAD_FILE_SHA256,
            HEAD_ARTIFACT_SHA256,
            adapter,
            BenchmarkEnvironment.capture(),
            score,
            passed,
            passed ? "PASS" : "FAIL",
            List.copyOf(observations));
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    mapper.writeValue(output.toFile(), report);
    System.out.printf(
        "%s balanced=%.6f calls=%d/%d no-calls=%d/%d physical=%d/%d report=%s%n",
        report.verdict(),
        score.balancedAccuracy(),
        score.correctCalls(),
        score.calls(),
        score.correctNoCalls(),
        score.noCalls(),
        observations.stream().filter(Observation::physicallyShared).count(),
        observations.size(),
        output.toAbsolutePath());
    return passed ? 0 : 1;
  }

  static ToolApplicabilityHead loadHead(ObjectMapper mapper, Path path) throws IOException {
    JsonNode artifact = mapper.readTree(path.toFile());
    if (artifact.path("schemaVersion").asInt() != 1
        || !EXPERIMENT.equals(artifact.path("experiment").asText())
        || !HEAD_ARTIFACT_SHA256.equals(artifact.path("artifactSha256").asText())
        || !MODEL_SHA256.equals(artifact.path("base").path("productionGgufSha256").asText())
        || !ADAPTER_SHA256.equals(artifact.path("adapterSha256").asText())
        || artifact.path("representation").path("hiddenDimension").asInt() != 2048
        || !artifact.path("validationPassed").asBoolean()) {
      throw new IllegalArgumentException("V16 applicability-head identity or validation changed");
    }
    return new ToolApplicabilityHead(
        floats(artifact.path("normalization").path("mean"), "normalization.mean"),
        floats(artifact.path("normalization").path("scale"), "normalization.scale"),
        floats(artifact.path("classifier").path("weight"), "classifier.weight"),
        (float) artifact.path("classifier").path("bias").asDouble());
  }

  static Score score(List<Observation> observations) {
    int calls = Math.toIntExact(observations.stream().filter(Observation::callExpected).count());
    int noCalls = observations.size() - calls;
    int correctCalls =
        Math.toIntExact(
            observations.stream()
                .filter(Observation::callExpected)
                .filter(Observation::callPredicted)
                .count());
    int correctNoCalls =
        Math.toIntExact(
            observations.stream()
                .filter(item -> !item.callExpected())
                .filter(item -> !item.callPredicted())
                .count());
    if (calls == 0 || noCalls == 0) {
      throw new IllegalArgumentException("both applicability classes are required");
    }
    double callAccuracy = (double) correctCalls / calls;
    double noCallAccuracy = (double) correctNoCalls / noCalls;
    return new Score(
        calls,
        noCalls,
        correctCalls,
        correctNoCalls,
        callAccuracy,
        noCallAccuracy,
        (callAccuracy + noCallAccuracy) / 2.0);
  }

  private static float[] floats(JsonNode values, String name) {
    if (!values.isArray() || values.size() != 2048) {
      throw new IllegalArgumentException(name + " must contain 2,048 values");
    }
    float[] result = new float[values.size()];
    for (int index = 0; index < values.size(); index++) {
      result[index] = (float) values.get(index).asDouble();
    }
    return result;
  }

  private static void requireHash(Path path, String expected, String name) throws IOException {
    String actual = Hashing.sha256(path);
    if (!expected.equals(actual)) {
      throw new IllegalArgumentException("V16 " + name + " SHA-256 changed: " + actual);
    }
  }

  private static Path requiredFile(Map<String, String> values, String name) {
    Path path = Path.of(required(values, name));
    if (!Files.isRegularFile(path)) {
      throw new IllegalArgumentException("--" + name + " is not a regular file: " + path);
    }
    return path;
  }

  private static Path requiredDirectory(Map<String, String> values, String name) {
    Path path = Path.of(required(values, name));
    if (!Files.isDirectory(path)) {
      throw new IllegalArgumentException("--" + name + " is not a directory: " + path);
    }
    return path;
  }

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }
}
