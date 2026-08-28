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
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.models.api.ToolSpec;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/** Evaluates Needle 2 output against the versioned upstream playground conformance suite. */
final class Needle2ToolQualification {
  static final String SUITE_RESOURCE = "tool-qualification/needle2-playground-v1.json";
  static final String POLICY_VERSION = "needle2-tool-conformance-v1";
  static final double MINIMUM_STRUCTURED_OUTPUT_RATE = 1.0;
  static final double MINIMUM_TOOL_SELECTION_EXACT_RATE = 1.0;
  static final double MINIMUM_SCHEMA_VALIDITY_RATE = 1.0;
  static final double MINIMUM_DECLARED_ARGUMENTS_ONLY_RATE = 1.0;
  static final double MINIMUM_EXPECTED_ARGUMENT_ACCURACY = 0.90;
  static final double MINIMUM_REFUSAL_ACCURACY = 1.0;

  private static final String SECTION_START = "<tool_call>";
  private static final String SECTION_END = "</tool_call>";

  private Needle2ToolQualification() {}

  record Suite(
      int schemaVersion,
      String suiteId,
      String sourceRepository,
      String sourceRevision,
      String sourcePath,
      List<Case> cases) {
    Suite {
      cases = List.copyOf(Objects.requireNonNull(cases, "cases"));
    }
  }

  record Case(
      String id,
      String label,
      List<ToolDeclaration> tools,
      String query,
      List<ExpectedCall> expectedCalls) {
    Case {
      tools = List.copyOf(Objects.requireNonNull(tools, "tools"));
      expectedCalls = List.copyOf(Objects.requireNonNull(expectedCalls, "expectedCalls"));
    }

    List<ToolSpec> toolSpecs(ObjectMapper mapper) {
      return tools.stream()
          .map(
              tool ->
                  new ToolSpec(
                      tool.name(), tool.description(), writeJson(mapper, tool.parameters())))
          .toList();
    }
  }

  record ToolDeclaration(String name, String description, ObjectNode parameters) {}

  record ExpectedCall(String name, ObjectNode arguments) {}

  record ActualCall(String name, ObjectNode arguments) {}

  record CaseResult(
      String id,
      long endToEndMillis,
      String output,
      boolean structured,
      boolean selectionExact,
      boolean schemaValid,
      boolean declaredArgumentsOnly,
      int expectedArgumentMatches,
      int expectedArguments,
      boolean refusalExpected,
      boolean refusalCorrect,
      boolean passed,
      List<String> diagnostics) {
    CaseResult {
      diagnostics = List.copyOf(Objects.requireNonNull(diagnostics, "diagnostics"));
    }
  }

  record Summary(
      int attempts,
      int passed,
      double structuredOutputRate,
      double toolSelectionExactRate,
      double schemaValidityRate,
      double declaredArgumentsOnlyRate,
      double expectedArgumentAccuracy,
      double refusalAccuracy,
      double p95EndToEndMillis,
      boolean qualified,
      String verdict) {}

  static Suite loadSuite(ObjectMapper mapper) throws IOException {
    Objects.requireNonNull(mapper, "mapper");
    ClassLoader loader = Needle2ToolQualification.class.getClassLoader();
    try (InputStream input = loader.getResourceAsStream(SUITE_RESOURCE)) {
      if (input == null) {
        throw new IOException("Missing qualification suite: " + SUITE_RESOURCE);
      }
      Suite suite = mapper.readValue(input, Suite.class);
      validateSuite(suite);
      return suite;
    }
  }

  static CaseResult evaluate(ObjectMapper mapper, Case item, String output, long elapsedMillis) {
    Objects.requireNonNull(mapper, "mapper");
    Objects.requireNonNull(item, "item");
    String generated = output == null ? "" : output;
    List<String> diagnostics = new ArrayList<>();
    ParsedOutput parsed = parseOutput(mapper, generated, diagnostics);
    Map<String, ToolDeclaration> declarations = new HashMap<>();
    for (ToolDeclaration tool : item.tools()) {
      declarations.put(tool.name(), tool);
    }

    List<String> expectedNames = item.expectedCalls().stream().map(ExpectedCall::name).toList();
    List<String> actualNames = parsed.calls().stream().map(ActualCall::name).toList();
    boolean selectionExact = parsed.structured() && actualNames.equals(expectedNames);
    if (!selectionExact) {
      diagnostics.add("expected tool sequence " + expectedNames + " but received " + actualNames);
    }

    boolean schemaValid = parsed.structured();
    boolean declaredOnly = parsed.structured();
    for (ActualCall call : parsed.calls()) {
      ToolDeclaration declaration = declarations.get(call.name());
      if (declaration == null) {
        schemaValid = false;
        declaredOnly = false;
        diagnostics.add("unknown tool " + call.name());
        continue;
      }
      if (!validAgainstSchema(
          call.arguments(), declaration.parameters(), diagnostics, call.name())) {
        schemaValid = false;
      }
      if (!declaredArgumentsOnly(call.arguments(), declaration.parameters())) {
        declaredOnly = false;
        diagnostics.add("tool " + call.name() + " emitted undeclared arguments");
      }
    }

    int expectedArguments = 0;
    int matches = 0;
    for (int callIndex = 0; callIndex < item.expectedCalls().size(); callIndex++) {
      ExpectedCall expected = item.expectedCalls().get(callIndex);
      Iterator<Map.Entry<String, JsonNode>> fields = expected.arguments().properties().iterator();
      while (fields.hasNext()) {
        Map.Entry<String, JsonNode> field = fields.next();
        expectedArguments++;
        if (callIndex < parsed.calls().size()) {
          ActualCall actual = parsed.calls().get(callIndex);
          if (expected.name().equals(actual.name())
              && equivalent(field.getValue(), actual.arguments().get(field.getKey()))) {
            matches++;
          } else {
            diagnostics.add(
                "expected " + expected.name() + "." + field.getKey() + "=" + field.getValue());
          }
        }
      }
    }

    boolean refusalExpected = item.expectedCalls().isEmpty();
    boolean refusalCorrect = !refusalExpected || (parsed.structured() && parsed.calls().isEmpty());
    boolean passed =
        parsed.structured()
            && selectionExact
            && schemaValid
            && declaredOnly
            && matches == expectedArguments
            && refusalCorrect;
    return new CaseResult(
        item.id(),
        elapsedMillis,
        generated,
        parsed.structured(),
        selectionExact,
        schemaValid,
        declaredOnly,
        matches,
        expectedArguments,
        refusalExpected,
        refusalCorrect,
        passed,
        diagnostics);
  }

  static Summary summarize(List<CaseResult> results) {
    List<CaseResult> attempts = List.copyOf(Objects.requireNonNull(results, "results"));
    if (attempts.isEmpty()) {
      throw new IllegalArgumentException("qualification results must not be empty");
    }
    int size = attempts.size();
    double structured = rate(attempts.stream().filter(CaseResult::structured).count(), size);
    double selection = rate(attempts.stream().filter(CaseResult::selectionExact).count(), size);
    double schema = rate(attempts.stream().filter(CaseResult::schemaValid).count(), size);
    double declaredOnly =
        rate(attempts.stream().filter(CaseResult::declaredArgumentsOnly).count(), size);
    int expected = attempts.stream().mapToInt(CaseResult::expectedArguments).sum();
    int matched = attempts.stream().mapToInt(CaseResult::expectedArgumentMatches).sum();
    double argumentAccuracy = expected == 0 ? 1.0 : rate(matched, expected);
    List<CaseResult> refusals = attempts.stream().filter(CaseResult::refusalExpected).toList();
    double refusalAccuracy =
        refusals.isEmpty()
            ? 1.0
            : rate(refusals.stream().filter(CaseResult::refusalCorrect).count(), refusals.size());
    double p95 =
        percentile95(attempts.stream().mapToLong(CaseResult::endToEndMillis).sorted().toArray());
    boolean qualified =
        structured >= MINIMUM_STRUCTURED_OUTPUT_RATE
            && selection >= MINIMUM_TOOL_SELECTION_EXACT_RATE
            && schema >= MINIMUM_SCHEMA_VALIDITY_RATE
            && declaredOnly >= MINIMUM_DECLARED_ARGUMENTS_ONLY_RATE
            && argumentAccuracy >= MINIMUM_EXPECTED_ARGUMENT_ACCURACY
            && refusalAccuracy >= MINIMUM_REFUSAL_ACCURACY;
    return new Summary(
        size,
        Math.toIntExact(attempts.stream().filter(CaseResult::passed).count()),
        structured,
        selection,
        schema,
        declaredOnly,
        argumentAccuracy,
        refusalAccuracy,
        p95,
        qualified,
        qualified ? "PASS" : "FAIL");
  }

  private static ParsedOutput parseOutput(
      ObjectMapper mapper, String output, List<String> diagnostics) {
    int sectionStart = output.indexOf(SECTION_START);
    if (sectionStart < 0) {
      diagnostics.add("missing " + SECTION_START);
      return new ParsedOutput(false, List.of());
    }
    int contentStart = sectionStart + SECTION_START.length();
    int sectionEnd = output.indexOf(SECTION_END, contentStart);
    if (sectionEnd < 0) {
      diagnostics.add("missing " + SECTION_END);
      return new ParsedOutput(false, List.of());
    }
    try {
      JsonNode root = mapper.readTree(output.substring(contentStart, sectionEnd));
      if (!root.isArray()) {
        diagnostics.add("tool section must contain a JSON array");
        return new ParsedOutput(false, List.of());
      }
      List<ActualCall> calls = new ArrayList<>();
      for (JsonNode value : root) {
        JsonNode name = value.get("name");
        JsonNode arguments = value.get("arguments");
        if (!value.isObject()
            || name == null
            || !name.isTextual()
            || arguments == null
            || !arguments.isObject()) {
          diagnostics.add("each tool call must contain a string name and object arguments");
          return new ParsedOutput(false, List.of());
        }
        calls.add(new ActualCall(name.textValue(), (ObjectNode) arguments));
      }
      return new ParsedOutput(true, List.copyOf(calls));
    } catch (IOException malformed) {
      diagnostics.add("malformed tool JSON: " + malformed.getMessage());
      return new ParsedOutput(false, List.of());
    }
  }

  private static boolean validAgainstSchema(
      ObjectNode arguments, ObjectNode schema, List<String> diagnostics, String toolName) {
    JsonNode properties = schema.path("properties");
    JsonNode required = schema.path("required");
    if (required.isArray()) {
      for (JsonNode field : required) {
        if (!arguments.has(field.asText())) {
          diagnostics.add("tool " + toolName + " is missing required argument " + field.asText());
          return false;
        }
      }
    }
    Iterator<Map.Entry<String, JsonNode>> fields = arguments.properties().iterator();
    while (fields.hasNext()) {
      Map.Entry<String, JsonNode> field = fields.next();
      JsonNode fieldSchema = properties.path(field.getKey());
      if (!fieldSchema.isObject()) {
        continue;
      }
      JsonNode value = field.getValue();
      if (!matchesType(value, fieldSchema.path("type").asText(""))) {
        diagnostics.add("tool " + toolName + " argument " + field.getKey() + " has wrong type");
        return false;
      }
      JsonNode allowed = fieldSchema.path("enum");
      if (allowed.isArray() && !containsEquivalent(allowed, value)) {
        diagnostics.add("tool " + toolName + " argument " + field.getKey() + " is outside enum");
        return false;
      }
      if (value.isArray() && fieldSchema.path("items").isObject()) {
        String itemType = fieldSchema.path("items").path("type").asText("");
        for (JsonNode item : value) {
          if (!matchesType(item, itemType)) {
            diagnostics.add(
                "tool " + toolName + " argument " + field.getKey() + " has wrong item type");
            return false;
          }
        }
      }
    }
    return true;
  }

  private static boolean declaredArgumentsOnly(ObjectNode arguments, ObjectNode schema) {
    JsonNode properties = schema.path("properties");
    Iterator<String> names = arguments.fieldNames();
    while (names.hasNext()) {
      if (!properties.has(names.next())) {
        return false;
      }
    }
    return true;
  }

  private static boolean matchesType(JsonNode value, String type) {
    return switch (type) {
      case "string" -> value.isTextual();
      case "integer" -> value.isIntegralNumber();
      case "number" -> value.isNumber();
      case "boolean" -> value.isBoolean();
      case "array" -> value.isArray();
      case "object" -> value.isObject();
      case "", "null" -> true;
      default -> false;
    };
  }

  private static boolean containsEquivalent(JsonNode values, JsonNode expected) {
    for (JsonNode value : values) {
      if (equivalent(value, expected)) {
        return true;
      }
    }
    return false;
  }

  private static boolean equivalent(JsonNode expected, JsonNode actual) {
    if (expected == null || actual == null) {
      return expected == actual;
    }
    if (expected.isNumber() && actual.isNumber()) {
      return Double.compare(expected.doubleValue(), actual.doubleValue()) == 0;
    }
    if (expected.isTextual() && actual.isTextual()) {
      return normalize(expected.textValue()).equals(normalize(actual.textValue()));
    }
    if (expected.isArray() && actual.isArray() && expected.size() == actual.size()) {
      for (int index = 0; index < expected.size(); index++) {
        if (!equivalent(expected.get(index), actual.get(index))) {
          return false;
        }
      }
      return true;
    }
    return expected.equals(actual);
  }

  private static String normalize(String value) {
    return value.strip().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
  }

  private static double rate(long numerator, long denominator) {
    return (double) numerator / denominator;
  }

  private static double percentile95(long[] sorted) {
    int index = Math.max(0, (int) Math.ceil(sorted.length * 0.95) - 1);
    return sorted[index];
  }

  private static void validateSuite(Suite suite) {
    if (suite.schemaVersion() != 1) {
      throw new IllegalArgumentException(
          "Unsupported Needle 2 qualification suite schema: " + suite.schemaVersion());
    }
    if (suite.cases().size() != 13) {
      throw new IllegalArgumentException(
          "Needle 2 upstream playground suite must contain 13 cases: " + suite.cases().size());
    }
    if (!suite.sourceRevision().matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("sourceRevision must be an exact Git commit");
    }
    if (suite.cases().stream().map(Case::id).distinct().count() != suite.cases().size()) {
      throw new IllegalArgumentException("qualification case IDs must be unique");
    }
  }

  private static String writeJson(ObjectMapper mapper, JsonNode value) {
    try {
      return mapper.writer().without(SerializationFeature.INDENT_OUTPUT).writeValueAsString(value);
    } catch (IOException impossible) {
      throw new IllegalArgumentException("Unable to serialize tool schema", impossible);
    }
  }

  private record ParsedOutput(boolean structured, List<ActualCall> calls) {}
}
