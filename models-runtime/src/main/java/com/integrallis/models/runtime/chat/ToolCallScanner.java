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
import java.util.ArrayList;
import java.util.List;
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
