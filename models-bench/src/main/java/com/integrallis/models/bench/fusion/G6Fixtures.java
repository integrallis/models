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
package com.integrallis.models.bench.fusion;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/** Gate G6: hand-constructed reasoning traces with known a_r, a_u and consistency. */
final class G6Fixtures {

  private static final ObjectMapper JSON = new ObjectMapper();

  private G6Fixtures() {}

  record Fixture(
      String id,
      String category,
      List<String> labels,
      String text,
      String expectedReasoningAnswer,
      String expectedStatedAnswer,
      boolean expectedConsistent) {}

  record Failure(
      String id,
      String expectedReasoning,
      String actualReasoning,
      String expectedStated,
      String actualStated,
      boolean expectedConsistent,
      boolean actualConsistent) {}

  record Result(
      String dataset,
      int total,
      int passed,
      Map<String, Integer> perCategory,
      List<Failure> failures,
      boolean pass) {
    Result {
      perCategory = Map.copyOf(perCategory);
      failures = List.copyOf(failures);
    }
  }

  static String fileName(DatasetKind dataset) {
    return switch (dataset) {
      case GSM8K -> "gsm8k.json";
      case ARC -> "arc.json";
      case MATH500 -> "math500.json";
      case GENERIC -> throw new IllegalArgumentException("no G6 fixtures for generic datasets");
    };
  }

  static List<Fixture> loadBundled(DatasetKind dataset) {
    String resource = "/com/integrallis/models/bench/fusion/g6/" + fileName(dataset);
    try (InputStream input = G6Fixtures.class.getResourceAsStream(resource)) {
      Objects.requireNonNull(input, "missing bundled fixture " + resource);
      return parse(JSON.readTree(input));
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  static List<Fixture> load(Path directory, DatasetKind dataset) throws IOException {
    return parse(JSON.readTree(Files.readString(directory.resolve(fileName(dataset)))));
  }

  private static List<Fixture> parse(JsonNode root) {
    List<Fixture> fixtures = new ArrayList<>();
    for (JsonNode node : root.path("fixtures")) {
      List<String> labels = new ArrayList<>();
      node.path("labels").forEach(label -> labels.add(label.asText()));
      fixtures.add(
          new Fixture(
              node.path("id").asText(),
              node.path("category").asText(),
              labels.isEmpty() ? null : List.copyOf(labels),
              node.path("text").asText(),
              textOrNull(node.path("expectedReasoningAnswer")),
              textOrNull(node.path("expectedStatedAnswer")),
              node.path("expectedConsistent").asBoolean()));
    }
    return fixtures;
  }

  private static String textOrNull(JsonNode node) {
    return node.isMissingNode() || node.isNull() ? null : node.asText();
  }

  static Result evaluate(DatasetKind dataset, List<Fixture> fixtures) {
    AnswerExtractor extractor = AnswerExtractor.forDataset(dataset);
    Map<String, Integer> perCategory = new TreeMap<>();
    List<Failure> failures = new ArrayList<>();
    int passed = 0;
    for (Fixture fixture : fixtures) {
      perCategory.merge(fixture.category(), 1, Integer::sum);
      TraceAnalysis analysis = TraceAnalysis.analyze(extractor, fixture.text(), fixture.labels());
      boolean ok =
          Objects.equals(fixture.expectedReasoningAnswer(), analysis.reasoningAnswer())
              && Objects.equals(fixture.expectedStatedAnswer(), analysis.statedAnswer())
              && fixture.expectedConsistent() == analysis.consistent();
      if (ok) {
        passed++;
      } else {
        failures.add(
            new Failure(
                fixture.id(),
                fixture.expectedReasoningAnswer(),
                analysis.reasoningAnswer(),
                fixture.expectedStatedAnswer(),
                analysis.statedAnswer(),
                fixture.expectedConsistent(),
                analysis.consistent()));
      }
    }
    return new Result(
        dataset.id(),
        fixtures.size(),
        passed,
        perCategory,
        failures,
        !fixtures.isEmpty() && passed == fixtures.size());
  }
}
