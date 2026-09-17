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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * One scored question.
 *
 * @param id stable item id
 * @param question question text
 * @param answer gold answer
 * @param choices multiple-choice options, empty for generative items
 */
public record DatasetItem(String id, String question, String answer, List<Choice> choices) {

  /** One multiple-choice option. */
  public record Choice(String label, String text) {}

  private static final ObjectMapper JSON = new ObjectMapper();

  public DatasetItem {
    Objects.requireNonNull(id, "id");
    Objects.requireNonNull(question, "question");
    Objects.requireNonNull(answer, "answer");
    choices = List.copyOf(choices);
  }

  /** Choice labels, or {@code null} for generative items. */
  public List<String> labels() {
    return choices.isEmpty() ? null : choices.stream().map(Choice::label).toList();
  }

  /** Loads a JSONL file; blank lines are rejected, ids must be unique. */
  public static List<DatasetItem> loadJsonl(Path path, DatasetKind dataset, FieldMapping fields)
      throws IOException {
    List<DatasetItem> items = new ArrayList<>();
    Set<String> seen = new HashSet<>();
    int lineNumber = 0;
    for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
      lineNumber++;
      if (line.isBlank()) {
        throw new IllegalArgumentException(path + ":" + lineNumber + " is blank");
      }
      JsonNode node = JSON.readTree(line);
      String id = required(node, fields.id(), path, lineNumber);
      String answer = required(node, fields.answer(), path, lineNumber);
      String question = node.path(fields.question()).asText("");
      List<Choice> choices = new ArrayList<>();
      JsonNode choiceNodes = node.path(fields.choices());
      if (choiceNodes.isArray()) {
        for (JsonNode choice : choiceNodes) {
          choices.add(new Choice(choice.path("label").asText(), choice.path("text").asText()));
        }
      } else if (node.path("labels").isArray()) {
        for (JsonNode label : node.path("labels")) {
          choices.add(new Choice(label.asText(), ""));
        }
      }
      if (dataset == DatasetKind.ARC && choices.isEmpty()) {
        throw new IllegalArgumentException(path + ":" + lineNumber + " ARC item has no choices");
      }
      if (!seen.add(id)) {
        throw new IllegalArgumentException(path + ":" + lineNumber + " duplicate id " + id);
      }
      items.add(new DatasetItem(id, question, answer, choices));
    }
    return List.copyOf(items);
  }

  private static String required(JsonNode node, String field, Path path, int line) {
    JsonNode value = node.path(field);
    if (value.isMissingNode() || value.isNull() || value.asText().isEmpty()) {
      throw new IllegalArgumentException(path + ":" + line + " missing field " + field);
    }
    return value.asText();
  }

  /** sha256 over the ids joined by newlines, matching prepare_fusion_data.py. */
  public static String idsSha256(List<DatasetItem> items) {
    return Digests.sha256(String.join("\n", items.stream().map(DatasetItem::id).toList()));
  }
}
