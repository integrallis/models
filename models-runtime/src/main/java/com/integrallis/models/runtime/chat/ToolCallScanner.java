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

import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Recovers tool calls from generated text, driven by a {@link ToolSyntax} descriptor.
 *
 * <p>Deliberately tolerant. Small models produce well-formed calls far less often than their
 * benchmark scores suggest, and the dominant failure is presentational rather than structural —
 * wrapping output in a markdown fence, or drifting between the {@code arguments} and {@code
 * parameters} spellings. Stripping fences alone recovers more calls than constrained decoding does,
 * so it happens first.
 *
 * <p>Malformed output is never an error: the scan degrades to plain text and the caller sees a
 * normal response. Throwing here would turn a recoverable formatting slip into a failed turn.
 *
 * <p>Argument text is extracted but never parsed. Locating its extent needs only brace matching,
 * and both target frameworks want the raw JSON string, so this module stays dependency-free.
 */
public final class ToolCallScanner {

  private ToolCallScanner() {}

  /** Outcome of a scan: the prose the model produced, plus any calls recovered from it. */
  public record Result(String content, List<ToolCall> toolCalls) {

    public Result {
      content = content == null ? "" : content;
      toolCalls = List.copyOf(Objects.requireNonNull(toolCalls, "toolCalls"));
    }

    public boolean hasCalls() {
      return !toolCalls.isEmpty();
    }

    /** A result carrying no calls, for callers that skip scanning entirely. */
    public static Result plainText(String text) {
      return new Result(text == null ? "" : text, List.of());
    }
  }

  /** Scans generated text for tool calls expressed in {@code syntax}. */
  public static Result scan(String generated, ToolSyntax syntax) {
    Objects.requireNonNull(syntax, "syntax");
    if (generated == null || generated.isEmpty()) {
      return Result.plainText("");
    }
    if (!syntax.parsable()) {
      // Either the family has no tool format, or it has a tagged one that cannot be decoded
      // without the declared schemas. See ToolSyntax#parsable().
      return Result.plainText(generated);
    }
    if (syntax.mode() == ToolSyntax.Mode.HARMONY) {
      return scanHarmony(generated, syntax);
    }

    String text = stripCodeFences(generated);
    List<ToolCall> calls = new ArrayList<>();
    StringBuilder prose = new StringBuilder();

    int cursor = 0;
    while (cursor < text.length()) {
      int callStart = nextCallStart(text, cursor, syntax);
      if (callStart < 0) {
        prose.append(text, cursor, text.length());
        break;
      }
      prose.append(text, cursor, callStart);

      if (syntax.arrayWrapped()) {
        int arrayStart = text.indexOf('[', callStart + syntax.sectionStart().length());
        int arrayEnd = arrayStart < 0 ? -1 : matchingArray(text, arrayStart);
        if (arrayStart < 0 || arrayEnd < 0) {
          prose.append(text, callStart, text.length());
          break;
        }
        int arrayCursor = arrayStart + 1;
        while (arrayCursor < arrayEnd) {
          int objectStart = text.indexOf('{', arrayCursor);
          if (objectStart < 0 || objectStart >= arrayEnd) {
            break;
          }
          int objectEnd = matchingBrace(text, objectStart);
          if (objectEnd < 0 || objectEnd > arrayEnd) {
            prose.append(text, callStart, text.length());
            return Result.plainText(generated);
          }
          ToolCall call = toCall(text.substring(objectStart, objectEnd + 1), calls.size(), syntax);
          if (call != null) {
            calls.add(call);
            if (!syntax.parallelCalls()) {
              break;
            }
          }
          arrayCursor = objectEnd + 1;
        }
        cursor = advancePastSectionEnd(text, arrayEnd + 1, syntax);
        continue;
      }

      int objectStart = text.indexOf('{', callStart);
      if (objectStart < 0) {
        prose.append(text, callStart, text.length());
        break;
      }
      int objectEnd = matchingBrace(text, objectStart);
      if (objectEnd < 0) {
        // Truncated mid-object; nothing recoverable past this point.
        prose.append(text, callStart, text.length());
        break;
      }

      String object = text.substring(objectStart, objectEnd + 1);
      ToolCall call = toCall(object, calls.size(), syntax);
      if (call != null) {
        calls.add(call);
        if (!syntax.parallelCalls()) {
          // Families capped at one call per turn cannot render a second one back into history,
          // so anything trailing the first is discarded rather than surfaced.
          break;
        }
      } else {
        prose.append(object);
      }
      cursor = advancePastSectionEnd(text, objectEnd + 1, syntax);
    }

    if (calls.isEmpty()) {
      return Result.plainText(generated);
    }
    return new Result(prose.toString().strip(), calls);
  }

  /**
   * Scans output with the declared tool schemas available for tagged, non-JSON protocols.
   *
   * <p>JSON-shaped families delegate to {@link #scan(String, ToolSyntax)}. MiniCPM5 needs this
   * overload because its XML parameter values carry no type information; the declared JSON Schema
   * determines whether a value becomes a JSON string, number, boolean, array, or object.
   */
  public static Result scan(String generated, ToolSyntax syntax, List<ToolSpec> tools) {
    Objects.requireNonNull(tools, "tools");
    if (syntax != ToolSyntax.MINICPM5) {
      return scan(generated, syntax);
    }
    return scanMiniCpm5(generated, tools);
  }

  private static Result scanMiniCpm5(String generated, List<ToolSpec> tools) {
    if (generated == null || generated.isEmpty()) {
      return Result.plainText("");
    }
    Map<String, ToolSpec> declarations = new HashMap<>();
    for (ToolSpec tool : tools) {
      declarations.put(tool.name(), tool);
    }
    String text = stripCodeFences(generated);
    List<ToolCall> calls = new ArrayList<>();
    StringBuilder prose = new StringBuilder();
    int cursor = 0;
    while (cursor < text.length()) {
      int callStart = text.indexOf(ToolSyntax.MINICPM5.sectionStart(), cursor);
      if (callStart < 0) {
        prose.append(text, cursor, text.length());
        break;
      }
      prose.append(text, cursor, callStart);
      int nameStart = callStart + ToolSyntax.MINICPM5.sectionStart().length();
      int nameEnd = text.indexOf('"', nameStart);
      int bodyStart = nameEnd < 0 ? -1 : text.indexOf('>', nameEnd + 1);
      int callEnd = bodyStart < 0 ? -1 : text.indexOf(ToolSyntax.MINICPM5.sectionEnd(), bodyStart);
      if (nameEnd < 0 || bodyStart < 0 || callEnd < 0) {
        return Result.plainText(generated);
      }
      String name = text.substring(nameStart, nameEnd);
      ToolSpec declaration = declarations.get(name);
      if (declaration == null) {
        return Result.plainText(generated);
      }
      String arguments = miniCpmArguments(text.substring(bodyStart + 1, callEnd), declaration);
      if (arguments == null) {
        return Result.plainText(generated);
      }
      calls.add(ToolCall.of(calls.size(), name, arguments));
      cursor = callEnd + ToolSyntax.MINICPM5.sectionEnd().length();
    }
    return calls.isEmpty()
        ? Result.plainText(generated)
        : new Result(prose.toString().strip(), calls);
  }

  private static String miniCpmArguments(String body, ToolSpec declaration) {
    String properties = rawField(declaration.inputSchema(), "properties");
    if (properties == null) {
      return null;
    }
    List<String> names = new ArrayList<>();
    List<String> values = new ArrayList<>();
    int cursor = 0;
    String startMarker = "<param name=\"";
    while (cursor < body.length()) {
      int parameterStart = body.indexOf(startMarker, cursor);
      if (parameterStart < 0) {
        break;
      }
      int nameStart = parameterStart + startMarker.length();
      int nameEnd = body.indexOf('"', nameStart);
      int valueStart = nameEnd < 0 ? -1 : body.indexOf('>', nameEnd + 1);
      int valueEnd = valueStart < 0 ? -1 : body.indexOf("</param>", valueStart + 1);
      if (nameEnd < 0 || valueStart < 0 || valueEnd < 0) {
        return null;
      }
      String name = body.substring(nameStart, nameEnd);
      if (names.contains(name)) {
        return null;
      }
      String property = rawField(properties, name);
      String type = property == null ? null : stringField(property, "type");
      String value = jsonValue(body.substring(valueStart + 1, valueEnd), type);
      if (value == null) {
        return null;
      }
      names.add(name);
      values.add(value);
      cursor = valueEnd + "</param>".length();
    }
    String required = rawField(declaration.inputSchema(), "required");
    if (required != null) {
      for (String name : jsonStringValues(required)) {
        if (!names.contains(name)) {
          return null;
        }
      }
    }
    StringBuilder json = new StringBuilder("{");
    for (int index = 0; index < names.size(); index++) {
      if (index > 0) {
        json.append(',');
      }
      json.append(quoteJson(names.get(index))).append(':').append(values.get(index));
    }
    return json.append('}').toString();
  }

  private static String jsonValue(String taggedValue, String type) {
    if (type == null || type.isBlank()) {
      return null;
    }
    String value = unwrapCdata(taggedValue).strip();
    return switch (type) {
      case "string" -> quoteJson(value);
      case "integer" -> value.matches("-?(0|[1-9][0-9]*)") ? value : null;
      case "number" -> finiteNumber(value) ? value : null;
      case "boolean" -> value.equals("true") || value.equals("false") ? value : null;
      case "array" ->
          value.startsWith("[") && matchingArray(value, 0) == value.length() - 1 ? value : null;
      case "object" ->
          value.startsWith("{") && matchingBrace(value, 0) == value.length() - 1 ? value : null;
      default -> null;
    };
  }

  private static boolean finiteNumber(String value) {
    try {
      return Double.isFinite(Double.parseDouble(value));
    } catch (NumberFormatException ignored) {
      return false;
    }
  }

  private static String unwrapCdata(String value) {
    String stripped = value.strip();
    if (stripped.startsWith("<![CDATA[") && stripped.endsWith("]]>")) {
      return stripped.substring("<![CDATA[".length(), stripped.length() - "]]>".length());
    }
    return value;
  }

  private static List<String> jsonStringValues(String array) {
    List<String> values = new ArrayList<>();
    boolean escaped = false;
    StringBuilder value = null;
    for (int index = 0; index < array.length(); index++) {
      char current = array.charAt(index);
      if (value == null) {
        if (current == '"') {
          value = new StringBuilder();
        }
      } else if (escaped) {
        value.append(current);
        escaped = false;
      } else if (current == '\\') {
        escaped = true;
      } else if (current == '"') {
        values.add(value.toString());
        value = null;
      } else {
        value.append(current);
      }
    }
    return values;
  }

  private static String quoteJson(String value) {
    StringBuilder result = new StringBuilder(value.length() + 2).append('"');
    for (int index = 0; index < value.length(); index++) {
      char current = value.charAt(index);
      switch (current) {
        case '"' -> result.append("\\\"");
        case '\\' -> result.append("\\\\");
        case '\b' -> result.append("\\b");
        case '\f' -> result.append("\\f");
        case '\n' -> result.append("\\n");
        case '\r' -> result.append("\\r");
        case '\t' -> result.append("\\t");
        default -> {
          if (current < 0x20) {
            result.append(String.format("\\u%04x", (int) current));
          } else {
            result.append(current);
          }
        }
      }
    }
    return result.append('"').toString();
  }

  private static Result scanHarmony(String generated, ToolSyntax syntax) {
    List<ToolCall> calls = new ArrayList<>();
    int cursor = 0;
    while (true) {
      int recipient = generated.indexOf("to=functions.", cursor);
      if (recipient < 0) {
        break;
      }
      int nameStart = recipient + "to=functions.".length();
      int nameEnd = nameStart;
      while (nameEnd < generated.length()) {
        char character = generated.charAt(nameEnd);
        if (Character.isWhitespace(character) || character == '<') {
          break;
        }
        nameEnd++;
      }
      String name = generated.substring(nameStart, nameEnd);
      int message = generated.indexOf("<|message|>", nameEnd);
      int boundary = nextHarmonyBoundary(generated, nameEnd);
      if (message < 0 || boundary < message) {
        cursor = Math.max(nameEnd, recipient + 1);
        continue;
      }
      int objectStart = generated.indexOf('{', message + "<|message|>".length());
      int objectEnd = objectStart < 0 ? -1 : matchingBrace(generated, objectStart);
      if (!name.isEmpty() && objectEnd >= 0) {
        calls.add(ToolCall.of(calls.size(), name, generated.substring(objectStart, objectEnd + 1)));
        cursor = objectEnd + 1;
      } else {
        cursor = message + "<|message|>".length();
      }
    }

    StringBuilder content = new StringBuilder();
    boolean foundHarmonyMessage = false;
    cursor = 0;
    while (true) {
      int marker = generated.indexOf("<|message|>", cursor);
      if (marker < 0) {
        break;
      }
      foundHarmonyMessage = true;
      int header = generated.lastIndexOf("<|start|>", marker);
      int headerStart = header < 0 ? 0 : header;
      String messageHeader = generated.substring(headerStart, marker);
      boolean toolCall = messageHeader.contains("to=functions.");
      String channel = harmonyChannel(messageHeader);
      int valueStart = marker + "<|message|>".length();
      int valueEnd = nextHarmonyBoundary(generated, valueStart);
      boolean privateAnalysis = channel.equals("analysis") && calls.isEmpty();
      if (!toolCall && !privateAnalysis && valueEnd > valueStart) {
        if (!content.isEmpty()) {
          content.append('\n');
        }
        content.append(generated, valueStart, valueEnd);
      }
      cursor = Math.max(valueEnd, valueStart + 1);
    }
    if (!foundHarmonyMessage && calls.isEmpty()) {
      return Result.plainText(generated);
    }
    return new Result(content.toString().strip(), calls);
  }

  private static String harmonyChannel(String header) {
    int marker = header.indexOf("<|channel|>");
    if (marker < 0) {
      return "";
    }
    int start = marker + "<|channel|>".length();
    int end = start;
    while (end < header.length()) {
      char character = header.charAt(end);
      if (Character.isWhitespace(character) || character == '<') {
        break;
      }
      end++;
    }
    return header.substring(start, end);
  }

  private static int nextHarmonyBoundary(String text, int from) {
    int end = text.length();
    for (String marker : new String[] {"<|end|>", "<|call|>", "<|return|>", "<|start|>"}) {
      int candidate = text.indexOf(marker, from);
      if (candidate >= 0 && candidate < end) {
        end = candidate;
      }
    }
    return end;
  }

  /**
   * Locates where the next candidate call begins, honouring the family's delimiter if it has one.
   */
  private static int nextCallStart(String text, int from, ToolSyntax syntax) {
    if (!syntax.sectionStart().isEmpty()) {
      return text.indexOf(syntax.sectionStart(), from);
    }
    return text.indexOf('{', from);
  }

  private static int advancePastSectionEnd(String text, int from, ToolSyntax syntax) {
    if (syntax.sectionEnd().isEmpty()) {
      return from;
    }
    int end = text.indexOf(syntax.sectionEnd(), from);
    return end < 0 ? from : end + syntax.sectionEnd().length();
  }

  /** Builds a call from one JSON object, or returns null when required fields are missing. */
  private static ToolCall toCall(String object, int index, ToolSyntax syntax) {
    String name = stringField(object, syntax.nameField());
    if (name == null || name.isBlank()) {
      return null;
    }
    // Accept whichever spelling the model used: models drift between the two, and the declared
    // one is only the family's preference.
    String arguments = rawField(object, syntax.argsField());
    if (arguments == null) {
      arguments = rawField(object, "arguments");
    }
    if (arguments == null) {
      arguments = rawField(object, "parameters");
    }
    return ToolCall.of(index, name, arguments);
  }

  /**
   * Removes markdown code fences, keeping their contents.
   *
   * <p>This is the highest-yield tolerance in the scanner: fenced output is the most common reason
   * a syntactically correct call fails to parse.
   */
  private static String stripCodeFences(String text) {
    if (text.indexOf("```") < 0) {
      return text;
    }
    StringBuilder stripped = new StringBuilder(text.length());
    int cursor = 0;
    while (cursor < text.length()) {
      int fence = text.indexOf("```", cursor);
      if (fence < 0) {
        stripped.append(text, cursor, text.length());
        break;
      }
      stripped.append(text, cursor, fence);
      int afterFence = fence + 3;
      // Skip an optional language tag on the opening fence.
      int newline = text.indexOf('\n', afterFence);
      int closing = text.indexOf("```", afterFence);
      if (closing < 0) {
        int bodyStart = newline >= 0 && newline < text.length() ? newline + 1 : afterFence;
        stripped.append(text, bodyStart, text.length());
        break;
      }
      int bodyStart = newline >= 0 && newline < closing ? newline + 1 : afterFence;
      stripped.append(text, bodyStart, closing);
      cursor = closing + 3;
    }
    return stripped.toString();
  }

  /** Returns the index of the brace closing the one at {@code open}, or -1 if unbalanced. */
  private static int matchingBrace(String text, int open) {
    int depth = 0;
    boolean inString = false;
    boolean escaped = false;
    for (int index = open; index < text.length(); index++) {
      char current = text.charAt(index);
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (current == '\\') {
          escaped = true;
        } else if (current == '"') {
          inString = false;
        }
        continue;
      }
      switch (current) {
        case '"' -> inString = true;
        case '{' -> depth++;
        case '}' -> {
          depth--;
          if (depth == 0) {
            return index;
          }
        }
        default -> {
          // Structural characters outside strings need no handling.
        }
      }
    }
    return -1;
  }

  /** Returns the closing bracket of a JSON array, ignoring brackets inside strings. */
  private static int matchingArray(String text, int open) {
    int depth = 0;
    boolean inString = false;
    boolean escaped = false;
    for (int index = open; index < text.length(); index++) {
      char current = text.charAt(index);
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (current == '\\') {
          escaped = true;
        } else if (current == '"') {
          inString = false;
        }
        continue;
      }
      if (current == '"') {
        inString = true;
      } else if (current == '[') {
        depth++;
      } else if (current == ']') {
        depth--;
        if (depth == 0) {
          return index;
        }
      }
    }
    return -1;
  }

  /** Reads a top-level string field, or null when absent. */
  private static String stringField(String object, String key) {
    int value = valueStart(object, key);
    if (value < 0 || object.charAt(value) != '"') {
      return null;
    }
    StringBuilder text = new StringBuilder();
    boolean escaped = false;
    for (int index = value + 1; index < object.length(); index++) {
      char current = object.charAt(index);
      if (escaped) {
        text.append(current);
        escaped = false;
      } else if (current == '\\') {
        escaped = true;
      } else if (current == '"') {
        return text.toString();
      } else {
        text.append(current);
      }
    }
    return null;
  }

  /** Reads a top-level field verbatim, preserving whatever JSON text it holds. */
  private static String rawField(String object, String key) {
    int value = valueStart(object, key);
    if (value < 0) {
      return null;
    }
    if (object.charAt(value) == '{') {
      int end = matchingBrace(object, value);
      return end < 0 ? null : object.substring(value, end + 1);
    }
    // A non-object value: read to the next top-level separator.
    int depth = 0;
    boolean inString = false;
    boolean escaped = false;
    for (int index = value; index < object.length(); index++) {
      char current = object.charAt(index);
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (current == '\\') {
          escaped = true;
        } else if (current == '"') {
          inString = false;
        }
        continue;
      }
      if (current == '"') {
        inString = true;
      } else if (current == '[' || current == '{') {
        depth++;
      } else if (current == ']' || current == '}') {
        if (depth == 0) {
          return object.substring(value, index).strip();
        }
        depth--;
      } else if (current == ',' && depth == 0) {
        return object.substring(value, index).strip();
      }
    }
    return null;
  }

  /** Finds where the value for a top-level {@code key} begins, or -1 when the key is absent. */
  private static int valueStart(String object, String key) {
    String quoted = '"' + key + '"';
    int depth = 0;
    boolean inString = false;
    boolean escaped = false;
    for (int index = 0; index < object.length(); index++) {
      char current = object.charAt(index);
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (current == '\\') {
          escaped = true;
        } else if (current == '"') {
          inString = false;
        }
        continue;
      }
      switch (current) {
        case '"' -> {
          if (depth == 1 && object.startsWith(quoted, index)) {
            int colon = object.indexOf(':', index + quoted.length());
            if (colon < 0) {
              return -1;
            }
            int value = colon + 1;
            while (value < object.length() && Character.isWhitespace(object.charAt(value))) {
              value++;
            }
            return value < object.length() ? value : -1;
          }
          inString = true;
        }
        case '{', '[' -> depth++;
        case '}', ']' -> depth--;
        default -> {
          // Only structural characters affect key lookup.
        }
      }
    }
    return -1;
  }
}
