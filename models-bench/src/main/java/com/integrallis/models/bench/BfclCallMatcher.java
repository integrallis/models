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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/** Java port of the pinned BFCL AST checker's accepted-value comparison. */
final class BfclCallMatcher {
  private static final Pattern STRING_PUNCTUATION = Pattern.compile("[ ,./\\-_*^]");
  private static final Pattern TOOL_BLOCK =
      Pattern.compile("<tool_call>\\s*(.*?)\\s*</tool_call>", Pattern.DOTALL);

  private BfclCallMatcher() {}

  static Optional<List<ToolCall>> parseStrictCompletion(ObjectMapper mapper, String completion) {
    Objects.requireNonNull(mapper, "mapper");
    if (completion == null) {
      return Optional.empty();
    }
    var matcher = TOOL_BLOCK.matcher(completion);
    List<ToolCall> calls = new ArrayList<>();
    int cursor = 0;
    boolean sawBlock = false;
    boolean sawEmpty = false;
    while (matcher.find()) {
      if (!completion.substring(cursor, matcher.start()).isBlank()) {
        return Optional.empty();
      }
      cursor = matcher.end();
      sawBlock = true;
      JsonNode value;
      try {
        value = mapper.readTree(matcher.group(1));
      } catch (JsonProcessingException failure) {
        return Optional.empty();
      }
      if (value.isArray() && value.isEmpty()) {
        sawEmpty = true;
        continue;
      }
      if (!value.isObject()
          || value.size() != 2
          || !value.has("name")
          || !value.path("name").isTextual()
          || value.path("name").asText().isBlank()
          || !value.has("arguments")
          || !value.path("arguments").isObject()) {
        return Optional.empty();
      }
      calls.add(
          ToolCall.of(
              calls.size(), value.path("name").asText(), value.path("arguments").toString()));
    }
    if (!sawBlock || !completion.substring(cursor).isBlank() || (sawEmpty && !calls.isEmpty())) {
      return Optional.empty();
    }
    return Optional.of(List.copyOf(calls));
  }

  static boolean exactCallsMatch(
      ObjectMapper mapper, List<ToolCall> actual, JsonNode expected, List<ToolSpec> tools) {
    Objects.requireNonNull(mapper, "mapper");
    Objects.requireNonNull(actual, "actual");
    Objects.requireNonNull(expected, "expected");
    Objects.requireNonNull(tools, "tools");
    if (!expected.isArray() || actual.size() != expected.size()) {
      return false;
    }

    Map<String, JsonNode> schemas = schemas(mapper, tools);
    List<JsonNode> unmatched = new ArrayList<>();
    expected.forEach(unmatched::add);
    for (ToolCall call : actual) {
      JsonNode arguments;
      try {
        arguments = mapper.readTree(call.argumentsJson());
      } catch (JsonProcessingException failure) {
        return false;
      }
      if (!arguments.isObject()) {
        return false;
      }
      int match = -1;
      for (int index = 0; index < unmatched.size(); index++) {
        if (expectedCallMatches(
            call.name(), arguments, unmatched.get(index), schemas.get(call.name()))) {
          match = index;
          break;
        }
      }
      if (match < 0) {
        return false;
      }
      unmatched.remove(match);
    }
    return unmatched.isEmpty();
  }

  static boolean callsValidate(ObjectMapper mapper, List<ToolCall> calls, List<ToolSpec> tools) {
    Objects.requireNonNull(mapper, "mapper");
    Objects.requireNonNull(calls, "calls");
    Objects.requireNonNull(tools, "tools");
    Map<String, JsonNode> schemas = schemas(mapper, tools);
    for (ToolCall call : calls) {
      JsonNode schema = schemas.get(call.name());
      if (schema == null) {
        return false;
      }
      JsonNode arguments;
      try {
        arguments = mapper.readTree(call.argumentsJson());
      } catch (JsonProcessingException failure) {
        return false;
      }
      if (!arguments.isObject() || !objectValid(arguments, schema)) {
        return false;
      }
    }
    return true;
  }

  private static Map<String, JsonNode> schemas(ObjectMapper mapper, List<ToolSpec> tools) {
    Map<String, JsonNode> schemas = new HashMap<>();
    for (ToolSpec tool : tools) {
      try {
        JsonNode schema = mapper.readTree(tool.inputSchema());
        if (!schema.isObject() || schemas.put(tool.name(), schema) != null) {
          throw new IllegalArgumentException("invalid or duplicate tool schema: " + tool.name());
        }
      } catch (JsonProcessingException failure) {
        throw new IllegalArgumentException("invalid tool schema: " + tool.name(), failure);
      }
    }
    return Map.copyOf(schemas);
  }

  private static boolean expectedCallMatches(
      String actualName, JsonNode actualArguments, JsonNode expected, JsonNode schema) {
    if (!expected.isObject() || expected.size() != 1) {
      throw new IllegalArgumentException("ground-truth call must name exactly one function");
    }
    Map.Entry<String, JsonNode> entry = expected.properties().iterator().next();
    if (!entry.getKey().equals(actualName) || !entry.getValue().isObject()) {
      return false;
    }
    JsonNode expectedArguments = entry.getValue();
    Iterator<String> actualNames = actualArguments.fieldNames();
    while (actualNames.hasNext()) {
      if (!expectedArguments.has(actualNames.next())) {
        return false;
      }
    }
    JsonNode properties = schema == null ? null : schema.path("properties");
    for (Map.Entry<String, JsonNode> argument : expectedArguments.properties()) {
      if (!argument.getValue().isArray() || argument.getValue().isEmpty()) {
        throw new IllegalArgumentException(
            "ground-truth alternatives must be a nonempty array: " + argument.getKey());
      }
      if (!actualArguments.has(argument.getKey())) {
        if (!containsEmptyString(argument.getValue())) {
          return false;
        }
      } else {
        JsonNode argumentSchema =
            properties != null && properties.isObject() ? properties.path(argument.getKey()) : null;
        if (!valueMatches(
            actualArguments.get(argument.getKey()), argument.getValue(), argumentSchema)) {
          return false;
        }
      }
    }
    return true;
  }

  private static boolean valueMatches(JsonNode actual, JsonNode alternatives, JsonNode schema) {
    String type = schema == null ? "" : schema.path("type").asText("");
    return switch (type) {
      case "object" -> actual.isObject() && dictionaryMatches(actual, alternatives, schema);
      case "array" -> arrayMatches(actual, alternatives, schema.path("items"));
      case "string" ->
          actual.isTextual()
              && values(alternatives)
                  .filter(JsonNode::isTextual)
                  .map(JsonNode::textValue)
                  .map(BfclCallMatcher::normalizeString)
                  .anyMatch(normalizeString(actual.textValue())::equals);
      case "integer" ->
          actual.isIntegralNumber()
              && values(alternatives)
                  .filter(JsonNode::isIntegralNumber)
                  .anyMatch(item -> item.bigIntegerValue().equals(actual.bigIntegerValue()));
      case "number" ->
          actual.isNumber()
              && values(alternatives)
                  .filter(JsonNode::isNumber)
                  .anyMatch(item -> Double.compare(item.doubleValue(), actual.doubleValue()) == 0);
      case "boolean" ->
          actual.isBoolean()
              && values(alternatives)
                  .filter(JsonNode::isBoolean)
                  .anyMatch(item -> item.booleanValue() == actual.booleanValue());
      default -> values(alternatives).anyMatch(actual::equals);
    };
  }

  private static boolean objectValid(JsonNode object, JsonNode schema) {
    JsonNode properties = schema.path("properties");
    if (!properties.isObject()) {
      return false;
    }
    JsonNode required = schema.path("required");
    if (required.isArray()) {
      for (JsonNode name : required) {
        if (!name.isTextual() || !object.has(name.textValue())) {
          return false;
        }
      }
    }
    for (Map.Entry<String, JsonNode> field : object.properties()) {
      JsonNode fieldSchema = properties.get(field.getKey());
      if (fieldSchema == null || !schemaValueValid(field.getValue(), fieldSchema)) {
        return false;
      }
    }
    return true;
  }

  private static boolean schemaValueValid(JsonNode value, JsonNode schema) {
    String type = schema.path("type").asText("");
    boolean typeValid =
        switch (type) {
          case "string" -> value.isTextual();
          case "integer" -> value.isIntegralNumber();
          case "number" -> value.isNumber();
          case "boolean" -> value.isBoolean();
          case "array" -> value.isArray();
          case "object" -> value.isObject();
          case "", "null" -> true;
          default -> false;
        };
    if (!typeValid) {
      return false;
    }
    JsonNode allowed = schema.path("enum");
    if (allowed.isArray() && values(allowed).noneMatch(value::equals)) {
      return false;
    }
    if (value.isObject() && "object".equals(type) && !objectValid(value, schema)) {
      return false;
    }
    if (value.isArray() && "array".equals(type) && schema.path("items").isObject()) {
      JsonNode itemSchema = schema.path("items");
      for (JsonNode item : value) {
        if (!schemaValueValid(item, itemSchema)) {
          return false;
        }
      }
    }
    return true;
  }

  private static boolean dictionaryMatches(
      JsonNode actual, JsonNode alternatives, JsonNode schema) {
    JsonNode properties = schema.path("properties");
    for (JsonNode candidate : alternatives) {
      if (!candidate.isObject()) {
        continue;
      }
      boolean matches = true;
      for (Map.Entry<String, JsonNode> field : actual.properties()) {
        if (!candidate.has(field.getKey())
            || !valueMatches(
                field.getValue(), candidate.get(field.getKey()), properties.path(field.getKey()))) {
          matches = false;
          break;
        }
      }
      if (!matches) {
        continue;
      }
      for (Map.Entry<String, JsonNode> field : candidate.properties()) {
        if (!actual.has(field.getKey()) && !containsEmptyString(field.getValue())) {
          matches = false;
          break;
        }
      }
      if (matches) {
        return true;
      }
    }
    return false;
  }

  private static boolean arrayMatches(JsonNode actual, JsonNode alternatives, JsonNode itemSchema) {
    if (!actual.isArray()) {
      return false;
    }
    for (JsonNode candidate : alternatives) {
      if (!candidate.isArray() || candidate.size() != actual.size()) {
        continue;
      }
      boolean matches = true;
      for (int index = 0; index < actual.size(); index++) {
        JsonNode value = actual.get(index);
        JsonNode expected = candidate.get(index);
        if ("object".equals(itemSchema.path("type").asText())) {
          if (!value.isObject()
              || !dictionaryMatches(
                  value,
                  com.fasterxml.jackson.databind.node.JsonNodeFactory.instance
                      .arrayNode()
                      .add(expected),
                  itemSchema)) {
            matches = false;
            break;
          }
        } else if (!normalizedScalar(value).equals(normalizedScalar(expected))) {
          matches = false;
          break;
        }
      }
      if (matches) {
        return true;
      }
    }
    return false;
  }

  private static JsonNode normalizedScalar(JsonNode value) {
    if (!value.isTextual()) {
      return value;
    }
    return new com.fasterxml.jackson.databind.node.TextNode(normalizeString(value.textValue()));
  }

  private static boolean containsEmptyString(JsonNode alternatives) {
    return values(alternatives).anyMatch(item -> item.isTextual() && item.textValue().isEmpty());
  }

  private static Stream<JsonNode> values(JsonNode array) {
    return StreamSupport.stream(array.spliterator(), false);
  }

  private static String normalizeString(String value) {
    return STRING_PUNCTUATION
        .matcher(value)
        .replaceAll("")
        .toLowerCase(Locale.ROOT)
        .replace('\'', '"');
  }
}
