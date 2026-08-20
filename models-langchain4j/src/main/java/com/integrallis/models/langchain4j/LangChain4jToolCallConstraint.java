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
package com.integrallis.models.langchain4j;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.TokenConstraint;
import com.integrallis.models.runtime.ToolCallTokenConstraints;
import com.integrallis.models.runtime.chat.ToolSyntax;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

final class LangChain4jToolCallConstraint {

  private static final ObjectMapper JSON = new ObjectMapper();

  private LangChain4jToolCallConstraint() {}

  static Optional<TokenConstraint> compile(
      Tokenizer tokenizer, ToolSyntax syntax, List<ToolSpec> tools) {
    return ToolCallTokenConstraints.compile(
        tokenizer, syntax, tools, tool -> argumentAlternatives(tool.inputSchema()));
  }

  private static List<String> argumentAlternatives(String schema) {
    JsonNode root;
    try {
      root = JSON.readTree(schema);
    } catch (IOException | RuntimeException e) {
      return List.of();
    }
    JsonNode propertiesNode = root.path("properties");
    if (!propertiesNode.isObject()) {
      return List.of("{}");
    }
    List<String> required = requiredProperties(root.path("required"));
    List<Property> properties = new ArrayList<>();
    propertiesNode
        .properties()
        .forEach(
            entry -> {
              JsonNode enumNode = entry.getValue().path("enum");
              if (enumNode.isArray()) {
                List<String> values = enumValues(enumNode);
                if (!values.isEmpty()) {
                  properties.add(new Property(entry.getKey(), values));
                }
              }
            });
    for (String propertyName : required) {
      if (properties.stream().noneMatch(property -> property.name().equals(propertyName))) {
        return List.of();
      }
    }
    if (properties.isEmpty()) {
      return List.of("{}");
    }
    List<String> alternatives = new ArrayList<>();
    buildArguments(properties, 0, new ArrayList<>(), alternatives);
    return alternatives;
  }

  private static List<String> requiredProperties(JsonNode requiredNode) {
    if (!requiredNode.isArray()) {
      return List.of();
    }
    List<String> required = new ArrayList<>();
    for (JsonNode value : requiredNode) {
      if (!value.isTextual()) {
        return List.of();
      }
      required.add(value.asText());
    }
    return required;
  }

  private static List<String> enumValues(JsonNode enumNode) {
    List<String> values = new ArrayList<>();
    for (JsonNode value : enumNode) {
      if (!value.isTextual()) {
        return List.of();
      }
      values.add(value.asText());
    }
    return values;
  }

  private static void buildArguments(
      List<Property> properties, int index, List<String> values, List<String> alternatives) {
    if (index == properties.size()) {
      alternatives.add(renderArguments(properties, values));
      return;
    }
    for (String value : properties.get(index).values()) {
      values.add(value);
      buildArguments(properties, index + 1, values, alternatives);
      values.removeLast();
    }
  }

  private static String renderArguments(List<Property> properties, List<String> values) {
    StringBuilder json = new StringBuilder();
    json.append('{');
    for (int index = 0; index < properties.size(); index++) {
      if (index > 0) {
        json.append(',');
      }
      json.append(quoteJson(properties.get(index).name()));
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

  private record Property(String name, List<String> values) {}
}
