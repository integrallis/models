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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Experimental constrained decoder for finite JSON Schema shapes.
 *
 * <p>This is deliberately limited to finite schema subsets so the runtime can validate the sampler
 * and generation hooks before adding a full JSON Schema parser/automaton. The constraint operates
 * on decoded token fragments, so it handles tokenizations that do not align with JSON punctuation.
 */
public final class JsonSchemaConstraint implements TokenConstraint {

  private static final int MAX_ALTERNATIVES = 256;

  private final Tokenizer tokenizer;
  private final List<String> alternatives;
  private final StringBuilder generated = new StringBuilder();

  private JsonSchemaConstraint(Tokenizer tokenizer, List<String> alternatives) {
    this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    if (alternatives.isEmpty()) {
      throw new IllegalArgumentException("alternatives must not be empty");
    }
    this.alternatives = List.copyOf(alternatives);
  }

  /** Constrains output to a JSON string whose value is one of {@code values}. */
  public static JsonSchemaConstraint stringEnum(Tokenizer tokenizer, List<String> values) {
    Objects.requireNonNull(values, "values");
    if (values.isEmpty()) {
      throw new IllegalArgumentException("values must not be empty");
    }
    List<String> alternatives = new ArrayList<>(values.size());
    for (String value : values) {
      alternatives.add(quoteJson(Objects.requireNonNull(value, "value")));
    }
    return new JsonSchemaConstraint(tokenizer, alternatives);
  }

  /**
   * Constrains output to a canonical JSON object whose required string properties each have a
   * finite enum.
   *
   * <p>Properties are emitted in the map's iteration order. Use an insertion-ordered map when the
   * canonical field order matters.
   */
  public static JsonSchemaConstraint requiredStringEnums(
      Tokenizer tokenizer, Map<String, ? extends List<String>> properties) {
    Objects.requireNonNull(properties, "properties");
    if (properties.isEmpty()) {
      throw new IllegalArgumentException("properties must not be empty");
    }
    List<Map.Entry<String, ? extends List<String>>> entries = List.copyOf(properties.entrySet());
    List<String> alternatives = new ArrayList<>();
    buildObjectAlternatives(entries, 0, new ArrayList<>(), alternatives);
    return new JsonSchemaConstraint(tokenizer, alternatives);
  }

  @Override
  public boolean allows(int token) {
    if (isComplete()) {
      return false;
    }
    String fragment = tokenizer.decode(token);
    if (fragment.isEmpty()) {
      return false;
    }
    String candidate = generated + fragment;
    for (String alternative : alternatives) {
      if (alternative.startsWith(candidate)) {
        return true;
      }
    }
    return false;
  }

  @Override
  public void accept(int token) {
    if (!allows(token)) {
      throw new IllegalArgumentException("token " + token + " violates the JSON constraint");
    }
    generated.append(tokenizer.decode(token));
  }

  @Override
  public boolean isComplete() {
    for (String alternative : alternatives) {
      if (alternative.contentEquals(generated)) {
        return true;
      }
    }
    return false;
  }

  private static void buildObjectAlternatives(
      List<Map.Entry<String, ? extends List<String>>> entries,
      int index,
      List<String> values,
      List<String> alternatives) {
    if (alternatives.size() > MAX_ALTERNATIVES) {
      throw new IllegalArgumentException(
          "schema expands beyond " + MAX_ALTERNATIVES + " alternatives");
    }
    if (index == entries.size()) {
      alternatives.add(renderObject(entries, values));
      return;
    }

    Map.Entry<String, ? extends List<String>> entry = entries.get(index);
    String key = entry.getKey();
    if (key == null || key.isBlank()) {
      throw new IllegalArgumentException("property names must not be blank");
    }
    List<String> enumValues = Objects.requireNonNull(entry.getValue(), "enumValues");
    if (enumValues.isEmpty()) {
      throw new IllegalArgumentException("enum values for " + key + " must not be empty");
    }
    for (String value : enumValues) {
      values.add(Objects.requireNonNull(value, "value"));
      buildObjectAlternatives(entries, index + 1, values, alternatives);
      values.removeLast();
    }
  }

  private static String renderObject(
      List<Map.Entry<String, ? extends List<String>>> entries, List<String> values) {
    StringBuilder json = new StringBuilder();
    json.append('{');
    for (int index = 0; index < entries.size(); index++) {
      if (index > 0) {
        json.append(',');
      }
      json.append(quoteJson(entries.get(index).getKey()));
      json.append(':');
      json.append(quoteJson(values.get(index)));
    }
    json.append('}');
    return json.toString();
  }

  private static String quoteJson(String value) {
    StringBuilder json = new StringBuilder(value.length() + 2);
    json.append('"');
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
            appendUnicodeEscape(json, ch);
          } else {
            json.append(ch);
          }
        }
      }
    }
    json.append('"');
    return json.toString();
  }

  private static void appendUnicodeEscape(StringBuilder json, char value) {
    json.append("\\u");
    for (int shift = 12; shift >= 0; shift -= 4) {
      int digit = (value >>> shift) & 0xf;
      json.append((char) (digit < 10 ? '0' + digit : 'a' + digit - 10));
    }
  }
}
