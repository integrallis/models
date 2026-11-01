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
package com.integrallis.models.audio.phonemizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A small recursive-descent JSON reader, enough for lexicon files and model configs: objects become
 * {@code LinkedHashMap}, arrays {@code ArrayList}, strings {@code String}, numbers {@code Double}
 * or {@code Long}, and {@code true}/{@code false}/{@code null} their Java values.
 */
final class Json {

  private final String text;
  private int position;

  private Json(String text) {
    this.text = text;
  }

  static Object parse(String text) {
    Json json = new Json(text);
    json.skipWhitespace();
    Object value = json.value();
    json.skipWhitespace();
    if (json.position != text.length()) {
      throw new IllegalArgumentException("trailing characters at " + json.position);
    }
    return value;
  }

  @SuppressWarnings("unchecked")
  static Map<String, Object> parseObject(String text) {
    Object value = parse(text);
    if (!(value instanceof Map)) {
      throw new IllegalArgumentException("expected a JSON object");
    }
    return (Map<String, Object>) value;
  }

  private Object value() {
    if (position >= text.length()) {
      throw new IllegalArgumentException("unexpected end of JSON");
    }
    char c = text.charAt(position);
    switch (c) {
      case '{':
        return object();
      case '[':
        return array();
      case '"':
        return string();
      case 't':
        return literal("true", Boolean.TRUE);
      case 'f':
        return literal("false", Boolean.FALSE);
      case 'n':
        return literal("null", null);
      default:
        return number();
    }
  }

  private Map<String, Object> object() {
    Map<String, Object> map = new LinkedHashMap<>();
    position++;
    skipWhitespace();
    if (peek() == '}') {
      position++;
      return map;
    }
    while (true) {
      skipWhitespace();
      String key = string();
      skipWhitespace();
      expect(':');
      skipWhitespace();
      map.put(key, value());
      skipWhitespace();
      char c = text.charAt(position++);
      if (c == '}') {
        return map;
      }
      if (c != ',') {
        throw new IllegalArgumentException("expected ',' or '}' at " + (position - 1));
      }
    }
  }

  private List<Object> array() {
    List<Object> list = new ArrayList<>();
    position++;
    skipWhitespace();
    if (peek() == ']') {
      position++;
      return list;
    }
    while (true) {
      skipWhitespace();
      list.add(value());
      skipWhitespace();
      char c = text.charAt(position++);
      if (c == ']') {
        return list;
      }
      if (c != ',') {
        throw new IllegalArgumentException("expected ',' or ']' at " + (position - 1));
      }
    }
  }

  private String string() {
    expect('"');
    StringBuilder out = new StringBuilder();
    while (true) {
      if (position >= text.length()) {
        throw new IllegalArgumentException("unterminated string");
      }
      char c = text.charAt(position++);
      if (c == '"') {
        return out.toString();
      }
      if (c != '\\') {
        out.append(c);
        continue;
      }
      char escaped = text.charAt(position++);
      switch (escaped) {
        case '"', '\\', '/' -> out.append(escaped);
        case 'b' -> out.append('\b');
        case 'f' -> out.append('\f');
        case 'n' -> out.append('\n');
        case 'r' -> out.append('\r');
        case 't' -> out.append('\t');
        case 'u' -> {
          out.append((char) Integer.parseInt(text.substring(position, position + 4), 16));
          position += 4;
        }
        default -> throw new IllegalArgumentException("bad escape \\" + escaped);
      }
    }
  }

  private Object number() {
    int start = position;
    while (position < text.length() && "+-0123456789.eE".indexOf(text.charAt(position)) >= 0) {
      position++;
    }
    String literal = text.substring(start, position);
    if (literal.isEmpty()) {
      throw new IllegalArgumentException("unexpected character at " + start);
    }
    if (literal.indexOf('.') < 0 && literal.indexOf('e') < 0 && literal.indexOf('E') < 0) {
      return Long.parseLong(literal);
    }
    return Double.parseDouble(literal);
  }

  private Object literal(String word, Object value) {
    if (!text.startsWith(word, position)) {
      throw new IllegalArgumentException("unexpected token at " + position);
    }
    position += word.length();
    return value;
  }

  private void expect(char c) {
    if (position >= text.length() || text.charAt(position) != c) {
      throw new IllegalArgumentException("expected '" + c + "' at " + position);
    }
    position++;
  }

  private char peek() {
    return position < text.length() ? text.charAt(position) : '\0';
  }

  private void skipWhitespace() {
    while (position < text.length() && Character.isWhitespace(text.charAt(position))) {
      position++;
    }
  }
}
