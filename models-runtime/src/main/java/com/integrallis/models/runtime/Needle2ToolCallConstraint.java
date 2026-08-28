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
package com.integrallis.models.runtime;

import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.api.ToolSpec;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Grammar-constrained decoder for Needle 2's array-wrapped tool-call protocol. */
final class Needle2ToolCallConstraint implements TokenConstraint {

  private static final String TOOL_START = "<tool_call>";
  private static final String TOOL_END = "</tool_call>";
  private static final int MAX_OBJECT_ALTERNATIVES = 4096;
  private static final Pattern JSON_PATTERN =
      Pattern.compile(
          "\\\"(?:[^\\\"\\\\\\x00-\\x1f]|\\\\[\\\"\\\\/bfnrt]|\\\\u[0-9a-fA-F]{4})*\\\"");
  private static final String JSON_STRING = JSON_PATTERN.pattern();
  private static final String JSON_INTEGER = "-?(?:0|[1-9][0-9]*)";
  private static final String JSON_NUMBER = JSON_INTEGER + "(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?";

  private final Tokenizer tokenizer;
  private final Pattern payloadPattern;
  private final StringBuilder generated = new StringBuilder();
  private int payloadOffset = -1;

  private Needle2ToolCallConstraint(Tokenizer tokenizer, Pattern payloadPattern) {
    this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    this.payloadPattern = Objects.requireNonNull(payloadPattern, "payloadPattern");
  }

  static Needle2ToolCallConstraint compile(Tokenizer tokenizer, List<ToolSpec> tools) {
    Objects.requireNonNull(tools, "tools");
    if (tools.isEmpty()) {
      throw new IllegalArgumentException("Needle 2 tool grammar requires at least one tool");
    }
    List<String> calls = new ArrayList<>(tools.size());
    Set<String> names = new LinkedHashSet<>();
    for (ToolSpec tool : tools) {
      if (!names.add(tool.name())) {
        throw new IllegalArgumentException("duplicate Needle 2 tool name " + tool.name());
      }
      Object schema = SchemaJson.parse(tool.inputSchema());
      String arguments = schemaRegex(schema, "$tool." + tool.name());
      calls.add(
          literal("{\"name\":")
              + literal(quoteJson(tool.name()))
              + literal(",\"arguments\":")
              + arguments
              + literal("}"));
    }
    String call = choice(calls);
    String array =
        literal("[") + "(?:" + call + "(?:" + literal(",") + call + ")*)?" + literal("]");
    return new Needle2ToolCallConstraint(tokenizer, Pattern.compile(array + literal(TOOL_END)));
  }

  @Override
  public boolean allows(int token) {
    if (isComplete() || tokenizer.isEndOfGeneration(token)) {
      return false;
    }
    String fragment = tokenizer.decode(token);
    if (fragment.isEmpty()) {
      return false;
    }
    return validPrefix(generated + fragment);
  }

  @Override
  public void accept(int token) {
    if (!allows(token)) {
      throw new IllegalArgumentException("token " + token + " violates the Needle 2 tool grammar");
    }
    generated.append(tokenizer.decode(token));
    if (payloadOffset < 0) {
      int marker = generated.indexOf(TOOL_START);
      if (marker >= 0) {
        payloadOffset = marker + TOOL_START.length();
      }
    }
  }

  @Override
  public boolean isComplete() {
    if (payloadOffset < 0) {
      return false;
    }
    return payloadPattern.matcher(generated.substring(payloadOffset)).matches();
  }

  private boolean validPrefix(CharSequence candidate) {
    int marker = payloadOffset;
    if (marker < 0) {
      marker = candidate.toString().indexOf(TOOL_START);
      if (marker < 0) {
        return true;
      }
      marker += TOOL_START.length();
    }
    Matcher matcher = payloadPattern.matcher(candidate.subSequence(marker, candidate.length()));
    return matcher.matches() || matcher.hitEnd();
  }

  private static String schemaRegex(Object value, String location) {
    Map<String, Object> schema = object(value, location);
    Object constant = schema.get("const");
    if (constant != null) {
      return literal(jsonLiteral(constant));
    }
    List<Object> enumeration = list(schema.get("enum"), location + ".enum", false);
    if (enumeration != null && !enumeration.isEmpty()) {
      return choice(enumeration.stream().map(item -> literal(jsonLiteral(item))).toList());
    }
    List<Object> variants = list(schema.get("oneOf"), location + ".oneOf", false);
    if (variants == null) {
      variants = list(schema.get("anyOf"), location + ".anyOf", false);
    }
    if (variants != null) {
      return choice(variants.stream().map(item -> schemaRegex(item, location)).toList());
    }

    Object typeValue = schema.get("type");
    if (typeValue instanceof List<?>) {
      return choice(
          ((List<?>) typeValue)
              .stream()
                  .map(
                      type ->
                          schemaRegex(withType(schema, string(type, location + ".type")), location))
                  .toList());
    }
    String type =
        typeValue == null
            ? (schema.containsKey("properties") ? "object" : "object")
            : string(typeValue, location + ".type");
    return switch (type) {
      case "object" -> objectRegex(schema, location);
      case "array" -> arrayRegex(schema, location);
      case "string" -> JSON_STRING;
      case "integer" -> JSON_INTEGER;
      case "number" -> JSON_NUMBER;
      case "boolean" -> "(?:true|false)";
      case "null" -> "null";
      default ->
          throw new IllegalArgumentException(
              "unsupported JSON Schema type " + type + " at " + location);
    };
  }

  private static String objectRegex(Map<String, Object> schema, String location) {
    Map<String, Object> properties =
        schema.get("properties") == null
            ? Map.of()
            : object(schema.get("properties"), location + ".properties");
    List<Object> requiredValues = list(schema.get("required"), location + ".required", false);
    Set<String> required = new LinkedHashSet<>();
    if (requiredValues != null) {
      for (Object value : requiredValues) {
        required.add(string(value, location + ".required"));
      }
    }
    if (!properties.keySet().containsAll(required)) {
      Set<String> missing = new LinkedHashSet<>(required);
      missing.removeAll(properties.keySet());
      throw new IllegalArgumentException(
          "required properties are undeclared at " + location + ": " + missing);
    }
    List<Map.Entry<String, Object>> entries = List.copyOf(properties.entrySet());
    List<String> alternatives = new ArrayList<>();
    buildObjectAlternatives(entries, required, 0, new ArrayList<>(), alternatives, location);
    return choice(alternatives);
  }

  private static void buildObjectAlternatives(
      List<Map.Entry<String, Object>> properties,
      Set<String> required,
      int index,
      List<Map.Entry<String, Object>> selected,
      List<String> alternatives,
      String location) {
    if (alternatives.size() >= MAX_OBJECT_ALTERNATIVES) {
      throw new IllegalArgumentException(
          "JSON Schema object expands beyond " + MAX_OBJECT_ALTERNATIVES + " forms at " + location);
    }
    if (index == properties.size()) {
      StringBuilder regex = new StringBuilder(literal("{"));
      for (int property = 0; property < selected.size(); property++) {
        if (property > 0) {
          regex.append(literal(","));
        }
        Map.Entry<String, Object> entry = selected.get(property);
        regex
            .append(literal(quoteJson(entry.getKey())))
            .append(literal(":"))
            .append(schemaRegex(entry.getValue(), location + ".properties." + entry.getKey()));
      }
      alternatives.add(regex.append(literal("}")).toString());
      return;
    }
    Map.Entry<String, Object> property = properties.get(index);
    if (!required.contains(property.getKey())) {
      buildObjectAlternatives(properties, required, index + 1, selected, alternatives, location);
    }
    selected.add(property);
    buildObjectAlternatives(properties, required, index + 1, selected, alternatives, location);
    selected.removeLast();
  }

  private static String arrayRegex(Map<String, Object> schema, String location) {
    Object itemsValue = schema.get("items");
    if (itemsValue == null) {
      throw new IllegalArgumentException("array schema requires items at " + location);
    }
    String item = schemaRegex(itemsValue, location + ".items");
    int minimum = integer(schema.get("minItems"), 0, location + ".minItems");
    Integer maximum = optionalInteger(schema.get("maxItems"), location + ".maxItems");
    if (maximum != null && maximum < minimum) {
      throw new IllegalArgumentException("maxItems is less than minItems at " + location);
    }
    String body;
    if (maximum == null) {
      body =
          minimum == 0
              ? "(?:" + item + "(?:" + literal(",") + item + ")*)?"
              : repeatedItems(item, minimum) + "(?:" + literal(",") + item + ")*";
    } else {
      List<String> counts = new ArrayList<>();
      for (int count = minimum; count <= maximum; count++) {
        counts.add(repeatedItems(item, count));
      }
      body = choice(counts);
    }
    return literal("[") + body + literal("]");
  }

  private static String repeatedItems(String item, int count) {
    if (count == 0) {
      return "";
    }
    StringBuilder regex = new StringBuilder(item);
    for (int index = 1; index < count; index++) {
      regex.append(literal(",")).append(item);
    }
    return regex.toString();
  }

  private static Map<String, Object> withType(Map<String, Object> schema, String type) {
    Map<String, Object> copy = new LinkedHashMap<>(schema);
    copy.put("type", type);
    return copy;
  }

  private static String choice(List<String> alternatives) {
    if (alternatives.isEmpty()) {
      throw new IllegalArgumentException("grammar choice must not be empty");
    }
    return alternatives.size() == 1
        ? alternatives.getFirst()
        : "(?:" + String.join("|", alternatives) + ")";
  }

  private static String literal(String value) {
    return Pattern.quote(value);
  }

  private static Map<String, Object> object(Object value, String location) {
    if (!(value instanceof Map<?, ?> raw)) {
      throw new IllegalArgumentException("expected JSON object at " + location);
    }
    Map<String, Object> result = new LinkedHashMap<>();
    for (Map.Entry<?, ?> entry : raw.entrySet()) {
      result.put(string(entry.getKey(), location), entry.getValue());
    }
    return result;
  }

  @SuppressWarnings("unchecked")
  private static List<Object> list(Object value, String location, boolean required) {
    if (value == null && !required) {
      return null;
    }
    if (!(value instanceof List<?> list)) {
      throw new IllegalArgumentException("expected JSON array at " + location);
    }
    return (List<Object>) list;
  }

  private static String string(Object value, String location) {
    if (!(value instanceof String text)) {
      throw new IllegalArgumentException("expected JSON string at " + location);
    }
    return text;
  }

  private static int integer(Object value, int defaultValue, String location) {
    Integer parsed = optionalInteger(value, location);
    return parsed == null ? defaultValue : parsed;
  }

  private static Integer optionalInteger(Object value, String location) {
    if (value == null) {
      return null;
    }
    if (value instanceof JsonNumber number) {
      try {
        return Integer.valueOf(number.text());
      } catch (NumberFormatException invalid) {
        throw new IllegalArgumentException("expected integer at " + location, invalid);
      }
    }
    throw new IllegalArgumentException("expected integer at " + location);
  }

  private static String jsonLiteral(Object value) {
    if (value == null) {
      return "null";
    }
    if (value instanceof String text) {
      return quoteJson(text);
    }
    if (value instanceof JsonNumber number) {
      return number.text();
    }
    if (value instanceof Boolean bool) {
      return bool.toString();
    }
    throw new IllegalArgumentException("enum and const values must be scalar JSON values");
  }

  private static String quoteJson(String value) {
    StringBuilder json = new StringBuilder(value.length() + 2).append('"');
    for (int index = 0; index < value.length(); index++) {
      char ch = value.charAt(index);
      switch (ch) {
        case '"' -> json.append("\\\"");
        case '\\' -> json.append("\\\\");
        case '\b' -> json.append("\\b");
        case '\f' -> json.append("\\f");
        case '\n' -> json.append("\\n");
        case '\r' -> json.append("\\r");
        case '\t' -> json.append("\\t");
        default -> {
          if (ch < 0x20) {
            json.append("\\u%04x".formatted((int) ch));
          } else {
            json.append(ch);
          }
        }
      }
    }
    return json.append('"').toString();
  }

  private record JsonNumber(String text) {}

  private static final class SchemaJson {
    private static final int MAX_DEPTH = 128;

    private SchemaJson() {}

    private static Object parse(String json) {
      Parser parser = new Parser(Objects.requireNonNull(json, "json"));
      Object value = parser.readValue(0);
      parser.skipWhitespace();
      if (!parser.atEnd()) {
        throw parser.error("trailing JSON content");
      }
      return value;
    }

    private static final class Parser {
      private final String json;
      private int offset;

      private Parser(String json) {
        this.json = json;
      }

      private Object readValue(int depth) {
        if (depth > MAX_DEPTH) {
          throw error("JSON nesting exceeds " + MAX_DEPTH);
        }
        skipWhitespace();
        if (atEnd()) {
          throw error("expected a JSON value");
        }
        return switch (json.charAt(offset)) {
          case '{' -> readObject(depth);
          case '[' -> readArray(depth);
          case '"' -> readString();
          case 't' -> readLiteral("true", Boolean.TRUE);
          case 'f' -> readLiteral("false", Boolean.FALSE);
          case 'n' -> readLiteral("null", null);
          default -> readNumber();
        };
      }

      private Map<String, Object> readObject(int depth) {
        expect('{');
        Map<String, Object> values = new LinkedHashMap<>();
        skipWhitespace();
        if (consume('}')) {
          return values;
        }
        while (true) {
          skipWhitespace();
          if (atEnd() || json.charAt(offset) != '"') {
            throw error("expected a JSON object field");
          }
          String name = readString();
          skipWhitespace();
          expect(':');
          Object value = readValue(depth + 1);
          if (values.containsKey(name)) {
            throw error("duplicate JSON Schema field " + name);
          }
          values.put(name, value);
          skipWhitespace();
          if (consume('}')) {
            return values;
          }
          expect(',');
        }
      }

      private List<Object> readArray(int depth) {
        expect('[');
        List<Object> values = new ArrayList<>();
        skipWhitespace();
        if (consume(']')) {
          return values;
        }
        while (true) {
          values.add(readValue(depth + 1));
          skipWhitespace();
          if (consume(']')) {
            return values;
          }
          expect(',');
        }
      }

      private String readString() {
        expect('"');
        StringBuilder value = new StringBuilder();
        while (!atEnd()) {
          char ch = json.charAt(offset++);
          if (ch == '"') {
            return value.toString();
          }
          if (ch < 0x20) {
            throw error("unescaped control character in JSON string");
          }
          if (ch != '\\') {
            value.append(ch);
            continue;
          }
          if (atEnd()) {
            throw error("unfinished JSON escape");
          }
          char escape = json.charAt(offset++);
          switch (escape) {
            case '"', '\\', '/' -> value.append(escape);
            case 'b' -> value.append('\b');
            case 'f' -> value.append('\f');
            case 'n' -> value.append('\n');
            case 'r' -> value.append('\r');
            case 't' -> value.append('\t');
            case 'u' -> value.append(readUnicodeEscape());
            default -> throw error("invalid JSON escape \\" + escape);
          }
        }
        throw error("unterminated JSON string");
      }

      private char readUnicodeEscape() {
        if (offset + 4 > json.length()) {
          throw error("unfinished Unicode escape");
        }
        int value = 0;
        for (int index = 0; index < 4; index++) {
          int digit = Character.digit(json.charAt(offset++), 16);
          if (digit < 0) {
            throw error("invalid Unicode escape");
          }
          value = (value << 4) | digit;
        }
        return (char) value;
      }

      private JsonNumber readNumber() {
        int start = offset;
        if (consume('-') && atEnd()) {
          throw error("unfinished JSON number");
        }
        if (consume('0')) {
          if (!atEnd() && isDigit(json.charAt(offset))) {
            throw error("leading zero in JSON number");
          }
        } else {
          if (atEnd() || json.charAt(offset) < '1' || json.charAt(offset) > '9') {
            throw error("expected a JSON value");
          }
          consumeDigits();
        }
        if (consume('.')) {
          requireDigit("fraction requires a digit");
          consumeDigits();
        }
        if (!atEnd() && (json.charAt(offset) == 'e' || json.charAt(offset) == 'E')) {
          offset++;
          if (!atEnd() && (json.charAt(offset) == '+' || json.charAt(offset) == '-')) {
            offset++;
          }
          requireDigit("exponent requires a digit");
          consumeDigits();
        }
        return new JsonNumber(json.substring(start, offset));
      }

      private Object readLiteral(String literal, Object value) {
        if (!json.startsWith(literal, offset)) {
          throw error("expected a JSON value");
        }
        offset += literal.length();
        return value;
      }

      private void consumeDigits() {
        while (!atEnd() && isDigit(json.charAt(offset))) {
          offset++;
        }
      }

      private void requireDigit(String message) {
        if (atEnd() || !isDigit(json.charAt(offset))) {
          throw error(message);
        }
      }

      private static boolean isDigit(char ch) {
        return ch >= '0' && ch <= '9';
      }

      private void skipWhitespace() {
        while (!atEnd()) {
          char ch = json.charAt(offset);
          if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
            return;
          }
          offset++;
        }
      }

      private boolean consume(char expected) {
        if (!atEnd() && json.charAt(offset) == expected) {
          offset++;
          return true;
        }
        return false;
      }

      private void expect(char expected) {
        if (!consume(expected)) {
          throw error("expected '" + expected + "'");
        }
      }

      private boolean atEnd() {
        return offset >= json.length();
      }

      private IllegalArgumentException error(String message) {
        return new IllegalArgumentException(
            "malformed tool JSON Schema at character " + offset + ": " + message);
      }
    }
  }
}
