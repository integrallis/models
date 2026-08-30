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
package com.integrallis.models.runtime.chat;

import com.integrallis.models.api.ToolSpec;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Renders JSON Schema tool declarations in GPT-OSS Harmony's TypeScript notation. */
final class HarmonyToolRenderer {

  private HarmonyToolRenderer() {}

  static String render(List<ToolSpec> tools) {
    StringBuilder text = new StringBuilder("# Tools\n\n## functions\n\nnamespace functions {\n");
    for (ToolSpec tool : tools) {
      text.append('\n');
      appendComment(text, tool.description(), "");
      text.append("type ").append(tool.name()).append(" = (_: ");
      Map<String, Object> schema = object(Json.parse(tool.inputSchema()), "$tool." + tool.name());
      if (schema.isEmpty()) {
        text.append("any");
      } else {
        appendType(text, schema, "", "$tool." + tool.name());
      }
      text.append(") => any;\n");
    }
    return text.append("\n} // namespace functions").toString();
  }

  private static void appendType(
      StringBuilder text, Map<String, Object> schema, String indent, String location) {
    List<Object> enumeration = list(schema.get("enum"), location + ".enum", false);
    if (enumeration != null && !enumeration.isEmpty()) {
      for (int index = 0; index < enumeration.size(); index++) {
        if (index > 0) {
          text.append(" | ");
        }
        text.append(jsonLiteral(enumeration.get(index), location + ".enum"));
      }
      return;
    }
    Object constant = schema.get("const");
    if (constant != null) {
      text.append(jsonLiteral(constant, location + ".const"));
      return;
    }
    List<Object> alternatives = list(schema.get("oneOf"), location + ".oneOf", false);
    if (alternatives == null) {
      alternatives = list(schema.get("anyOf"), location + ".anyOf", false);
    }
    if (alternatives != null && !alternatives.isEmpty()) {
      for (int index = 0; index < alternatives.size(); index++) {
        if (index > 0) {
          text.append(" | ");
        }
        appendType(text, object(alternatives.get(index), location), indent, location);
      }
      return;
    }

    Object typeValue = schema.get("type");
    if (typeValue instanceof List<?> types) {
      for (int index = 0; index < types.size(); index++) {
        if (index > 0) {
          text.append(" | ");
        }
        appendScalarType(
            text, string(types.get(index), location + ".type"), schema, indent, location);
      }
      return;
    }
    String type =
        typeValue == null
            ? (schema.containsKey("properties") ? "object" : "any")
            : string(typeValue, location + ".type");
    appendScalarType(text, type, schema, indent, location);
  }

  private static void appendScalarType(
      StringBuilder text, String type, Map<String, Object> schema, String indent, String location) {
    switch (type) {
      case "object" -> appendObject(text, schema, indent, location);
      case "array" -> {
        Object items = schema.get("items");
        if (items == null) {
          text.append("any[]");
        } else {
          appendType(text, object(items, location + ".items"), indent, location + ".items");
          text.append("[]");
        }
      }
      case "string" -> text.append("string");
      case "integer", "number" -> text.append("number");
      case "boolean" -> text.append("boolean");
      case "null" -> text.append("null");
      case "any" -> text.append("any");
      default ->
          throw new IllegalArgumentException(
              "unsupported JSON Schema type " + type + " at " + location);
    }
  }

  private static void appendObject(
      StringBuilder text, Map<String, Object> schema, String indent, String location) {
    Map<String, Object> properties =
        schema.get("properties") == null
            ? Map.of()
            : object(schema.get("properties"), location + ".properties");
    if (properties.isEmpty()) {
      text.append("any");
      return;
    }
    Set<String> required = new LinkedHashSet<>();
    List<Object> requiredValues = list(schema.get("required"), location + ".required", false);
    if (requiredValues != null) {
      for (Object value : requiredValues) {
        required.add(string(value, location + ".required"));
      }
    }
    text.append("{\n");
    String propertyIndent = indent;
    for (Map.Entry<String, Object> property : properties.entrySet()) {
      Map<String, Object> propertySchema =
          object(property.getValue(), location + ".properties." + property.getKey());
      Object description = propertySchema.get("description");
      if (description instanceof String value && !value.isBlank()) {
        appendComment(text, value, propertyIndent);
      }
      text.append(propertyIndent).append(property.getKey());
      if (!required.contains(property.getKey())) {
        text.append('?');
      }
      text.append(": ");
      appendType(
          text,
          propertySchema,
          propertyIndent + "    ",
          location + ".properties." + property.getKey());
      text.append(",\n");
    }
    text.append(indent).append('}');
  }

  private static void appendComment(StringBuilder text, String description, String indent) {
    if (description == null || description.isBlank()) {
      return;
    }
    for (String line : description.lines().toList()) {
      text.append(indent).append("// ").append(line).append('\n');
    }
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
    if (!(value instanceof List<?> values)) {
      throw new IllegalArgumentException("expected JSON array at " + location);
    }
    return (List<Object>) values;
  }

  private static String string(Object value, String location) {
    if (!(value instanceof String text)) {
      throw new IllegalArgumentException("expected JSON string at " + location);
    }
    return text;
  }

  private static String jsonLiteral(Object value, String location) {
    if (value == null) {
      return "null";
    }
    if (value instanceof String text) {
      return '"' + escape(text) + '"';
    }
    if (value instanceof JsonNumber number) {
      return number.text();
    }
    if (value instanceof Boolean bool) {
      return bool.toString();
    }
    throw new IllegalArgumentException("expected scalar JSON value at " + location);
  }

  private static String escape(String value) {
    return value.replace("\\", "\\\\").replace("\"", "\\\"");
  }

  private record JsonNumber(String text) {}

  private static final class Json {
    private static final int MAX_DEPTH = 128;

    private Json() {}

    static Object parse(String json) {
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
          String name = readString();
          skipWhitespace();
          expect(':');
          if (values.putIfAbsent(name, readValue(depth + 1)) != null) {
            throw error("duplicate field " + name);
          }
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
          if (ch != '\\') {
            if (ch < 0x20) {
              throw error("unescaped control character");
            }
            value.append(ch);
            continue;
          }
          if (atEnd()) {
            throw error("unfinished escape");
          }
          char escaped = json.charAt(offset++);
          switch (escaped) {
            case '"', '\\', '/' -> value.append(escaped);
            case 'b' -> value.append('\b');
            case 'f' -> value.append('\f');
            case 'n' -> value.append('\n');
            case 'r' -> value.append('\r');
            case 't' -> value.append('\t');
            case 'u' -> value.append(readUnicode());
            default -> throw error("invalid escape");
          }
        }
        throw error("unterminated string");
      }

      private char readUnicode() {
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

      private Object readLiteral(String expected, Object value) {
        if (!json.startsWith(expected, offset)) {
          throw error("expected JSON value");
        }
        offset += expected.length();
        return value;
      }

      private JsonNumber readNumber() {
        int start = offset;
        if (!atEnd() && json.charAt(offset) == '-') {
          offset++;
        }
        while (!atEnd()) {
          char ch = json.charAt(offset);
          if ((ch >= '0' && ch <= '9')
              || ch == '.'
              || ch == 'e'
              || ch == 'E'
              || ch == '+'
              || ch == '-') {
            offset++;
          } else {
            break;
          }
        }
        if (start == offset) {
          throw error("expected JSON value");
        }
        return new JsonNumber(json.substring(start, offset));
      }

      private void skipWhitespace() {
        while (!atEnd() && Character.isWhitespace(json.charAt(offset))) {
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
