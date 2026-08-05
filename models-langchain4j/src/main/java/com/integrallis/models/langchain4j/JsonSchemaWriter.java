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

import dev.langchain4j.model.chat.request.json.JsonAnyOfSchema;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonBooleanSchema;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonNullSchema;
import dev.langchain4j.model.chat.request.json.JsonNumberSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonReferenceSchema;
import dev.langchain4j.model.chat.request.json.JsonSchemaElement;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import java.util.List;
import java.util.Map;

/**
 * Serialises a LangChain4j {@link JsonSchemaElement} tree to JSON Schema text.
 *
 * <p>Spring AI hands tool schemas over as a string; LangChain4j models them as a typed tree, so an
 * adapter has to flatten it before a {@code ToolSpec} can carry it.
 *
 * <p>LangChain4j does ship converters, but only under {@code dev.langchain4j.internal}. This
 * project compiles against LangChain4j as {@code compileOnly} and its CI spans 1.0.0 through
 * 1.17.2, so binding to an internal package would make the adapter fragile across exactly the range
 * it claims to support. Walking the public type hierarchy costs little and is stable.
 */
final class JsonSchemaWriter {

  private JsonSchemaWriter() {}

  /**
   * Writes {@code schema} as JSON Schema text, or {@code {}} when a tool declares no parameters.
   */
  static String write(JsonSchemaElement schema) {
    if (schema == null) {
      return "{}";
    }
    StringBuilder json = new StringBuilder();
    append(json, schema);
    return json.toString();
  }

  private static void append(StringBuilder json, JsonSchemaElement schema) {
    // Already JSON Schema text, so re-encoding it would only risk corrupting it.
    String raw = rawSchemaText(schema);
    if (raw != null) {
      json.append(raw);
      return;
    }
    json.append('{');
    boolean first = true;

    if (schema instanceof JsonReferenceSchema reference) {
      first = appendField(json, first, "$ref", quoted(reference.reference()));
      first = appendDescription(json, first, reference.description());
      json.append('}');
      return;
    }

    if (schema instanceof JsonObjectSchema object) {
      first = appendField(json, first, "type", "\"object\"");
      first = appendDescription(json, first, object.description());
      first = appendProperties(json, first, "properties", object.properties());
      if (object.required() != null && !object.required().isEmpty()) {
        first = appendField(json, first, "required", stringArray(object.required()));
      }
      if (object.additionalProperties() != null) {
        first =
            appendField(
                json, first, "additionalProperties", object.additionalProperties().toString());
      }
      appendProperties(json, first, "$defs", object.definitions());
      json.append('}');
      return;
    }

    if (schema instanceof JsonArraySchema array) {
      first = appendField(json, first, "type", "\"array\"");
      first = appendDescription(json, first, array.description());
      if (array.items() != null) {
        appendRaw(json, first, "items", array.items());
        first = false;
      }
      json.append('}');
      return;
    }

    if (schema instanceof JsonEnumSchema enumeration) {
      // JSON Schema has no `enum` type; the values carry the constraint.
      first = appendField(json, first, "type", "\"string\"");
      first = appendDescription(json, first, enumeration.description());
      if (enumeration.enumValues() != null && !enumeration.enumValues().isEmpty()) {
        first = appendField(json, first, "enum", stringArray(enumeration.enumValues()));
      }
      json.append('}');
      return;
    }

    if (schema instanceof JsonAnyOfSchema anyOf) {
      first = appendDescription(json, first, anyOf.description());
      if (anyOf.anyOf() != null && !anyOf.anyOf().isEmpty()) {
        if (!first) {
          json.append(',');
        }
        json.append("\"anyOf\":[");
        for (int index = 0; index < anyOf.anyOf().size(); index++) {
          if (index > 0) {
            json.append(',');
          }
          append(json, anyOf.anyOf().get(index));
        }
        json.append(']');
      }
      json.append('}');
      return;
    }

    first = appendField(json, first, "type", quoted(scalarType(schema)));
    appendDescription(json, first, description(schema));
    json.append('}');
  }

  /**
   * Returns the verbatim text of a {@code JsonRawSchema}, or null for any other element.
   *
   * <p>Resolved reflectively rather than by type: {@code JsonRawSchema} arrived after LangChain4j
   * 1.0.0, which this module still compiles against, so naming it directly breaks the oldest
   * supported version.
   */
  private static String rawSchemaText(JsonSchemaElement schema) {
    if (!"JsonRawSchema".equals(schema.getClass().getSimpleName())) {
      return null;
    }
    try {
      Object value = schema.getClass().getMethod("schema").invoke(schema);
      return value instanceof String text ? text : null;
    } catch (ReflectiveOperationException unavailable) {
      return null;
    }
  }

  private static String scalarType(JsonSchemaElement schema) {
    if (schema instanceof JsonStringSchema) {
      return "string";
    }
    if (schema instanceof JsonIntegerSchema) {
      return "integer";
    }
    if (schema instanceof JsonNumberSchema) {
      return "number";
    }
    if (schema instanceof JsonBooleanSchema) {
      return "boolean";
    }
    if (schema instanceof JsonNullSchema) {
      return "null";
    }
    // An element type added by a newer LangChain4j; `object` keeps the schema usable.
    return "object";
  }

  private static String description(JsonSchemaElement schema) {
    if (schema instanceof JsonStringSchema value) {
      return value.description();
    }
    if (schema instanceof JsonIntegerSchema value) {
      return value.description();
    }
    if (schema instanceof JsonNumberSchema value) {
      return value.description();
    }
    if (schema instanceof JsonBooleanSchema value) {
      return value.description();
    }
    if (schema instanceof JsonNullSchema value) {
      return value.description();
    }
    return null;
  }

  private static boolean appendProperties(
      StringBuilder json, boolean first, String field, Map<String, JsonSchemaElement> properties) {
    if (properties == null || properties.isEmpty()) {
      return first;
    }
    if (!first) {
      json.append(',');
    }
    json.append(quoted(field)).append(":{");
    boolean firstProperty = true;
    for (Map.Entry<String, JsonSchemaElement> property : properties.entrySet()) {
      if (!firstProperty) {
        json.append(',');
      }
      firstProperty = false;
      json.append(quoted(property.getKey())).append(':');
      append(json, property.getValue());
    }
    json.append('}');
    return false;
  }

  private static void appendRaw(
      StringBuilder json, boolean first, String field, JsonSchemaElement value) {
    if (!first) {
      json.append(',');
    }
    json.append(quoted(field)).append(':');
    append(json, value);
  }

  private static boolean appendDescription(StringBuilder json, boolean first, String description) {
    if (description == null || description.isEmpty()) {
      return first;
    }
    return appendField(json, first, "description", quoted(description));
  }

  private static boolean appendField(
      StringBuilder json, boolean first, String field, String rawValue) {
    if (!first) {
      json.append(',');
    }
    json.append(quoted(field)).append(':').append(rawValue);
    return false;
  }

  private static String stringArray(List<String> values) {
    StringBuilder array = new StringBuilder("[");
    for (int index = 0; index < values.size(); index++) {
      if (index > 0) {
        array.append(',');
      }
      array.append(quoted(values.get(index)));
    }
    return array.append(']').toString();
  }

  private static String quoted(String value) {
    StringBuilder quoted = new StringBuilder(value.length() + 2).append('"');
    for (int index = 0; index < value.length(); index++) {
      char current = value.charAt(index);
      switch (current) {
        case '"' -> quoted.append("\\\"");
        case '\\' -> quoted.append("\\\\");
        case '\n' -> quoted.append("\\n");
        case '\r' -> quoted.append("\\r");
        case '\t' -> quoted.append("\\t");
        default -> {
          if (current < 0x20) {
            quoted.append(String.format("\\u%04x", (int) current));
          } else {
            quoted.append(current);
          }
        }
      }
    }
    return quoted.append('"').toString();
  }
}
