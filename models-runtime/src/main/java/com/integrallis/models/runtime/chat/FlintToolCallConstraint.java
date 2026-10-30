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

import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.TokenConstraint;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Opt-in experimental grammar for Flint's schema-dependent tagged tool protocol.
 *
 * <p>Only declared schemas constrain values: this API never takes expected answers. Free prose,
 * reasoning and refusal remain possible before the first call. Multiple calls remain possible until
 * the model emits its end token. Strings are raw at the argument boundary and JSON-quoted inside
 * objects/arrays. Object properties use declaration order; optional properties may be omitted.
 * Strings containing a literal {@code <} are outside this prototype's output subset.
 *
 * <p>The {@code format} keyword is treated as an annotation, not a validation assertion.
 * Unsupported validation keywords fail compilation rather than silently weakening the schema. Token
 * fragments must preserve decoded text, as with the runtime's other text constraints. This grammar
 * guarantees neither semantic argument accuracy nor completion within a token budget.
 */
public final class FlintToolCallConstraint implements TokenConstraint {
  private static final String START = "<tool_call>";
  private static final String END = "</tool_call>";
  private static final String THINK = "<think>";
  private static final String THINK_END = "</think>";
  private static final String INTEGER = "-?(?:0|[1-9][0-9]*)";
  private static final String NUMBER = INTEGER + "(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?";
  private static final String JSON_STRING =
      "\"(?:[^\"\\\\<\\x00-\\x1f]|\\\\[\"\\\\/bfnrt]|\\\\u[0-9a-fA-F]{4})*\"";
  private static final Set<String> KEYS =
      Set.of(
          "type",
          "properties",
          "required",
          "additionalProperties",
          "items",
          "enum",
          "const",
          "description",
          "title",
          "format",
          "$schema");
  private static final List<String> CONTROLS =
      List.of(
          START,
          END,
          THINK,
          THINK_END,
          "<arg_key>",
          "</arg_key>",
          "<arg_value>",
          "</arg_value>",
          "<|",
          "<tools>",
          "</tools>");
  private static final int MAX_FORMS = 4096;

  private final Tokenizer tokenizer;
  private final Pattern call;
  private final StringBuilder generated = new StringBuilder();
  private boolean ended;

  private FlintToolCallConstraint(Tokenizer tokenizer, Pattern call) {
    this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    this.call = call;
  }

  /** Compiles a request-scoped constraint from tool declarations alone. */
  public static FlintToolCallConstraint compile(Tokenizer tokenizer, List<ToolSpec> tools) {
    Objects.requireNonNull(tools, "tools");
    if (tools.isEmpty()) throw new IllegalArgumentException("Flint grammar requires tools");
    Set<String> names = new LinkedHashSet<>();
    List<String> calls = new ArrayList<>();
    for (ToolSpec tool : tools) {
      if (!names.add(tool.name()) || tool.name().contains("<"))
        throw new IllegalArgumentException("Invalid or duplicate Flint tool name: " + tool.name());
      Map<String, Object> schema = object(FlintToolProtocol.parseJson(tool.inputSchema()));
      validateSchema(schema, 0);
      if (!"object".equals(schema.get("type")))
        throw new IllegalArgumentException("Flint tool schema must declare type object");
      // Root enum/const would constrain the whole argument map, not individual tagged values.
      if (schema.containsKey("enum") || schema.containsKey("const"))
        throw new IllegalArgumentException("Root enum/const is not supported by Flint grammar");
      calls.add(literal(START + tool.name()) + objectPattern(schema, true) + literal(END));
    }
    return new FlintToolCallConstraint(tokenizer, Pattern.compile(choice(calls)));
  }

  @Override
  public boolean allows(int token) {
    if (ended) return false;
    if (tokenizer.isEndOfGeneration(token)) return valid(generated.toString(), true);
    String fragment = tokenizer.decode(token);
    return !fragment.isEmpty() && valid(generated + fragment, false);
  }

  @Override
  public void accept(int token) {
    if (!allows(token))
      throw new IllegalArgumentException("Token violates Flint tool grammar: " + token);
    if (tokenizer.isEndOfGeneration(token)) ended = true;
    else generated.append(tokenizer.decode(token));
  }

  @Override
  public boolean isComplete() {
    return ended;
  }

  private boolean valid(String text, boolean terminal) {
    int offset = 0;
    boolean called = false;
    while (offset < text.length()) {
      if (!called && text.startsWith(THINK, offset)) {
        int end = text.indexOf(THINK_END, offset + THINK.length());
        int nested = text.indexOf(THINK, offset + THINK.length());
        if (nested >= 0 && (end < 0 || nested < end)) return false;
        if (end < 0) return !terminal;
        offset = end + THINK_END.length();
      } else if (text.startsWith(START, offset)) {
        int end = text.indexOf(END, offset + START.length());
        if (end < 0) {
          if (terminal) return false;
          Matcher matcher = call.matcher(text.substring(offset));
          return matcher.matches() || matcher.hitEnd();
        }
        end += END.length();
        if (!call.matcher(text.substring(offset, end)).matches()) return false;
        offset = end;
        called = true;
      } else {
        String remaining = text.substring(offset);
        // A marker may span tokens. EOS must not turn its unfinished prefix into prose.
        if (CONTROLS.stream()
            .anyMatch(
                marker -> marker.length() > remaining.length() && marker.startsWith(remaining)))
          return !terminal && (!called || START.startsWith(remaining));
        if (CONTROLS.stream().anyMatch(remaining::startsWith)) return false;
        if (called && !Character.isWhitespace(text.charAt(offset))) return false;
        offset++;
      }
    }
    return true;
  }

  private static void validateSchema(Map<String, Object> schema, int depth) {
    if (depth > 32) throw new IllegalArgumentException("Flint schema nesting exceeds 32");
    for (String key : schema.keySet()) {
      if (!KEYS.contains(key))
        throw new IllegalArgumentException("Unsupported Flint schema keyword: " + key);
    }
    Object type = schema.get("type");
    if (!(type instanceof String)
        || !Set.of("object", "array", "string", "integer", "number", "boolean", "null")
            .contains(type))
      throw new IllegalArgumentException("Flint schema requires a supported single type");
    if (schema.containsKey("additionalProperties")
        && !(schema.get("additionalProperties") instanceof Boolean))
      throw new IllegalArgumentException("Schema-valued additionalProperties is unsupported");
    if (schema.containsKey("properties")) {
      for (var entry : object(schema.get("properties")).entrySet()) {
        if (entry.getKey().contains("<"))
          throw new IllegalArgumentException("Property names cannot contain <");
        validateSchema(object(entry.getValue()), depth + 1);
      }
    }
    if (schema.containsKey("items")) validateSchema(object(schema.get("items")), depth + 1);
    if (schema.containsKey("enum")) {
      Object enumeration = schema.get("enum");
      if (!(enumeration instanceof List<?> values) || values.isEmpty() || values.size() > MAX_FORMS)
        throw new IllegalArgumentException("Flint enum must contain 1..4096 values");
      for (Object value : values) validateConstant(value, schema);
    }
    if (schema.containsKey("const")) validateConstant(schema.get("const"), schema);
  }

  private static void validateConstant(Object value, Map<String, Object> schema) {
    if (!FlintToolProtocol.matches(value, schema)
        || FlintToolProtocol.json(value, false).contains("<"))
      throw new IllegalArgumentException("Enum/const is outside the Flint grammar schema subset");
  }

  private static String valuePattern(Map<String, Object> schema, boolean rawString) {
    if (schema.containsKey("const")) return constantPattern(schema.get("const"), rawString);
    if (schema.get("enum") instanceof List<?> values)
      return choice(values.stream().map(value -> constantPattern(value, rawString)).toList());
    return switch ((String) schema.get("type")) {
      case "string" -> rawString ? "[^<\\x00]*" : JSON_STRING;
      case "integer" -> INTEGER;
      case "number" -> NUMBER;
      case "boolean" -> "(?:true|false)";
      case "null" -> "null";
      case "object" -> objectPattern(schema, false);
      case "array" -> {
        String item = valuePattern(object(schema.get("items")), false);
        yield "\\[(?:" + item + "(?:," + item + ")*)?\\]";
      }
      default -> throw new IllegalArgumentException("Unsupported Flint schema type");
    };
  }

  private static String constantPattern(Object value, boolean rawString) {
    return literal(
        rawString && value instanceof String text ? text : FlintToolProtocol.json(value, false));
  }

  private static String objectPattern(Map<String, Object> schema, boolean tagged) {
    Map<String, Object> properties =
        schema.containsKey("properties") ? object(schema.get("properties")) : Map.of();
    Set<String> required = new LinkedHashSet<>();
    if (schema.containsKey("required")) {
      if (!(schema.get("required") instanceof List<?> values))
        throw new IllegalArgumentException("required must be an array");
      for (Object value : values) {
        if (!(value instanceof String name) || !properties.containsKey(name) || !required.add(name))
          throw new IllegalArgumentException("Invalid required property: " + value);
      }
    }
    List<String> forms = new ArrayList<>();
    forms.add("");
    for (var entry : properties.entrySet()) {
      List<String> next = new ArrayList<>();
      String property =
          tagged
              ? literal("<arg_key>" + entry.getKey() + "</arg_key><arg_value>")
                  + valuePattern(object(entry.getValue()), true)
                  + literal("</arg_value>")
              : literal(FlintToolProtocol.json(entry.getKey(), false) + ":")
                  + valuePattern(object(entry.getValue()), false);
      for (String form : forms) {
        if (!required.contains(entry.getKey())) next.add(form);
        next.add(form + (!tagged && !form.isEmpty() ? "," : "") + property);
        if (next.size() > MAX_FORMS)
          throw new IllegalArgumentException("Flint object exceeds 4096 forms");
      }
      forms = next;
    }
    String body = choice(forms);
    return tagged ? body : "\\{" + body + "\\}";
  }

  private static Map<String, Object> object(Object value) {
    if (!(value instanceof Map<?, ?> map))
      throw new IllegalArgumentException("Expected schema object");
    Map<String, Object> result = new java.util.LinkedHashMap<>();
    for (var entry : map.entrySet()) {
      if (!(entry.getKey() instanceof String key))
        throw new IllegalArgumentException("Expected string property");
      result.put(key, entry.getValue());
    }
    return result;
  }

  private static String choice(List<String> alternatives) {
    return "(?:" + String.join("|", alternatives) + ")";
  }

  private static String literal(String text) {
    return Pattern.quote(text);
  }
}
