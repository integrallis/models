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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Flint's published tagged protocol; strings stay raw and nonstrings use strict JSON. */
final class FlintToolProtocol {
  private FlintToolProtocol() {}

  static ModelPrompt render(List<ChatMessage> messages, List<ToolSpec> tools) {
    var prompt = ModelPrompt.builder();
    boolean pending = true;
    for (var message : messages) {
      if (message.hasToolCalls() && message.role() != ChatRole.ASSISTANT)
        throw new IllegalArgumentException("Only assistants can call Flint tools");
      if (pending && message.role() != ChatRole.SYSTEM) {
        appendTools(prompt, tools);
        pending = false;
      }
      if (message.role() == ChatRole.TOOL) {
        prompt
            .control("<|observation|><tool_response>")
            .text(escapeMessage(message.text()))
            .control("</tool_response>");
      } else {
        prompt.control("<|" + message.role().templateName() + "|>");
        if (message.role() == ChatRole.ASSISTANT) {
          appendAssistantText(prompt, message.text());
          for (var call : message.toolCalls()) {
            prompt.control("<tool_call>").text(call.name());
            for (var argument : object(Json.parse(call.argumentsJson())).entrySet()) {
              prompt.control("<arg_key>").text(argument.getKey()).control("</arg_key><arg_value>");
              Object value = argument.getValue();
              prompt
                  .text(value instanceof String s ? s : json(value, true))
                  .control("</arg_value>");
            }
            prompt.control("</tool_call>");
          }
        } else {
          prompt.text(escapeMessage(message.text()));
        }
      }
    }
    if (pending) appendTools(prompt, tools);
    return prompt.control("<|assistant|>").build();
  }

  private static void appendTools(ModelPrompt.Builder prompt, List<ToolSpec> tools) {
    if (tools.isEmpty()) return;
    prompt.control(
        "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
    for (var tool : tools) {
      Map<String, Object> function = new LinkedHashMap<>();
      function.put("name", tool.name());
      function.put("description", tool.description());
      function.put("parameters", object(Json.parse(tool.inputSchema())));
      prompt.text(json(function, true));
    }
    prompt.control(
        "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n");
  }

  private static String escapeMessage(String text) {
    text = text.replace("\u0000", "").replace("<think>", "").replace("</think>", "");
    for (String role :
        List.of(
            "begin_of_text",
            "end_of_text",
            "system",
            "developer",
            "user",
            "assistant",
            "observation")) {
      text = text.replace("<|" + role + "|>", "&lt;|" + role + "|&gt;");
    }
    return text;
  }

  private static void appendAssistantText(ModelPrompt.Builder prompt, String text) {
    int cursor = 0;
    while (cursor < text.length()) {
      int open = text.indexOf("<think>", cursor);
      int close = text.indexOf("</think>", cursor);
      int next = open < 0 ? close : close < 0 ? open : Math.min(open, close);
      if (next < 0) {
        prompt.text(text.substring(cursor));
        break;
      }
      prompt.text(text.substring(cursor, next));
      String marker = next == open ? "<think>" : "</think>";
      prompt.control(marker);
      cursor = next + marker.length();
    }
  }

  static ToolCallScanner.Result scan(String generated, List<ToolSpec> tools) {
    if (generated == null || generated.isEmpty()) return ToolCallScanner.Result.plainText("");
    try {
      Map<String, ToolSpec> declarations = new LinkedHashMap<>();
      for (var tool : tools) {
        if (declarations.putIfAbsent(tool.name(), tool) != null) throw malformed();
      }
      var calls = new ArrayList<ToolCall>();
      var prose = new StringBuilder();
      int cursor = 0;
      while (cursor < generated.length()) {
        int start = generated.indexOf("<tool_call>", cursor);
        int thought = generated.indexOf("<think>", cursor);
        if (thought >= 0 && (start < 0 || thought < start)) {
          int end = generated.indexOf("</think>", thought + 7);
          if (end < 0) throw malformed();
          int nested = generated.indexOf("<think>", thought + 7);
          if (nested >= 0 && nested < end) throw malformed();
          prose.append(generated, cursor, thought);
          cursor = end + 8;
          // Preserve reasoning in content so framework history can replay it.
          prose.append(generated, thought, cursor);
          continue;
        }
        if (start < 0) {
          prose.append(generated, cursor, generated.length());
          break;
        }
        prose.append(generated, cursor, start);
        int end = generated.indexOf("</tool_call>", start + 11);
        if (end < 0) throw malformed();
        String body = generated.substring(start + 11, end);
        int delimiter = body.indexOf('<');
        String name = (delimiter < 0 ? body : body.substring(0, delimiter)).strip();
        ToolSpec declaration = declarations.get(name);
        if (declaration == null) throw malformed();
        Map<String, Object> schema = object(Json.parse(declaration.inputSchema()));
        Map<String, Object> properties =
            schema.containsKey("properties") ? object(schema.get("properties")) : Map.of();
        Map<String, Object> arguments = new LinkedHashMap<>();
        int offset = delimiter < 0 ? body.length() : delimiter;
        while (offset < body.length()) {
          while (offset < body.length() && Character.isWhitespace(body.charAt(offset))) offset++;
          if (offset == body.length()) break;
          if (!body.startsWith("<arg_key>", offset)) throw malformed();
          int keyEnd = body.indexOf("</arg_key>", offset + 9);
          if (keyEnd < 0) throw malformed();
          String key = body.substring(offset + 9, keyEnd);
          offset = keyEnd + 10;
          while (offset < body.length() && Character.isWhitespace(body.charAt(offset))) offset++;
          if (!body.startsWith("<arg_value>", offset)) throw malformed();
          int valueEnd = body.indexOf("</arg_value>", offset + 11);
          if (valueEnd < 0 || arguments.containsKey(key) || !properties.containsKey(key))
            throw malformed();
          String raw = body.substring(offset + 11, valueEnd);
          if (raw.contains("<arg_key>")
              || raw.contains("<arg_value>")
              || raw.contains("<tool_call>")) throw malformed();
          var property = object(properties.get(key));
          Object value = "string".equals(property.get("type")) ? raw : Json.parse(raw);
          if (!matches(value, property)) throw malformed();
          arguments.put(key, value);
          offset = valueEnd + 12;
        }
        if (!matches(arguments, schema)) throw malformed();
        calls.add(ToolCall.of(calls.size(), name, json(arguments, false)));
        cursor = end + 12;
      }
      return calls.isEmpty()
          ? ToolCallScanner.Result.plainText(generated)
          : new ToolCallScanner.Result(prose.toString().strip(), calls);
    } catch (IllegalArgumentException failure) {
      return ToolCallScanner.Result.plainText(generated);
    }
  }

  // Type recovery and the structural/enum subset needed by the tool protocol, not a full JSON
  // Schema validator.
  private static boolean matches(Object value, Map<String, Object> schema) {
    if (schema.containsKey("$ref") || schema.containsKey("oneOf") || schema.containsKey("anyOf"))
      return false;
    Object declared = schema.get("type");
    List<?> types =
        declared instanceof List<?> list ? list : declared == null ? List.of() : List.of(declared);
    boolean typed = types.isEmpty();
    for (Object type : types) {
      if (!(type instanceof String)) return false;
      typed |=
          switch (type.toString()) {
            case "string" -> value instanceof String;
            case "integer" ->
                value instanceof JsonNumber n
                    && new java.math.BigDecimal(n.text()).stripTrailingZeros().scale() <= 0;
            case "number" -> value instanceof JsonNumber;
            case "boolean" -> value instanceof Boolean;
            case "null" -> value == null;
            case "object" -> value instanceof Map<?, ?>;
            case "array" -> value instanceof List<?>;
            default -> false;
          };
    }
    if (!typed) return false;
    if (schema.get("enum") instanceof List<?> enumeration && !enumeration.contains(value))
      return false;
    if (schema.containsKey("const") && !Objects.equals(value, schema.get("const"))) return false;
    if (value instanceof Map<?, ?> raw) {
      Map<String, Object> values = object(raw);
      Map<String, Object> properties =
          schema.containsKey("properties") ? object(schema.get("properties")) : Map.of();
      if (schema.get("required") instanceof List<?> required
          && !values.keySet().containsAll(required)) return false;
      for (var entry : values.entrySet()) {
        if (properties.containsKey(entry.getKey())) {
          if (!matches(entry.getValue(), object(properties.get(entry.getKey())))) return false;
        } else if (Boolean.FALSE.equals(schema.get("additionalProperties"))) return false;
      }
    }
    if (value instanceof List<?> values && schema.containsKey("items")) {
      for (Object item : values) if (!matches(item, object(schema.get("items")))) return false;
    }
    return true;
  }

  private static Map<String, Object> object(Object value) {
    if (!(value instanceof Map<?, ?> map)) throw malformed();
    Map<String, Object> result = new LinkedHashMap<>();
    for (var entry : map.entrySet()) {
      if (!(entry.getKey() instanceof String name)) throw malformed();
      result.put(name, entry.getValue());
    }
    return result;
  }

  private static IllegalArgumentException malformed() {
    return new IllegalArgumentException("Malformed or unsupported Flint tool call");
  }

  private static String json(Object value, boolean spaced) {
    if (value == null) return "null";
    if (value instanceof JsonNumber n) return n.text();
    if (value instanceof Boolean b) return b.toString();
    if (value instanceof String s) {
      var out = new StringBuilder("\"");
      for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i);
        switch (c) {
          case '"' -> out.append("\\\"");
          case '\\' -> out.append("\\\\");
          case '\n' -> out.append("\\n");
          case '\r' -> out.append("\\r");
          case '\t' -> out.append("\\t");
          case '\b' -> out.append("\\b");
          case '\f' -> out.append("\\f");
          default -> {
            if (c < 32) out.append(String.format("\\u%04x", (int) c));
            else out.append(c);
          }
        }
      }
      return out.append('"').toString();
    }
    var parts = new ArrayList<String>();
    if (value instanceof Map<?, ?> map) {
      for (var entry : map.entrySet())
        parts.add(
            json(entry.getKey(), spaced) + (spaced ? ": " : ":") + json(entry.getValue(), spaced));
      return "{" + String.join(spaced ? ", " : ",", parts) + "}";
    }
    if (value instanceof List<?> list) {
      for (var item : list) parts.add(json(item, spaced));
      return "[" + String.join(spaced ? ", " : ",", parts) + "]";
    }
    throw malformed();
  }

  private record JsonNumber(String text) {}

  private static final class Json {
    private static final int MAX_DEPTH = 128;

    private Json() {}

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
