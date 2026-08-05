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

import static org.assertj.core.api.Assertions.assertThat;

import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonBooleanSchema;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonNumberSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonRawSchema;
import dev.langchain4j.model.chat.request.json.JsonReferenceSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import java.util.List;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class JsonSchemaWriterTest {

  @Nested
  static class Scalars {

    @Test
    void writesEachPrimitiveType() {
      assertThat(JsonSchemaWriter.write(new JsonStringSchema())).isEqualTo("{\"type\":\"string\"}");
      assertThat(JsonSchemaWriter.write(new JsonIntegerSchema()))
          .isEqualTo("{\"type\":\"integer\"}");
      assertThat(JsonSchemaWriter.write(new JsonNumberSchema())).isEqualTo("{\"type\":\"number\"}");
      assertThat(JsonSchemaWriter.write(new JsonBooleanSchema()))
          .isEqualTo("{\"type\":\"boolean\"}");
    }

    @Test
    void includesDescriptionsWhenPresent() {
      assertThat(JsonSchemaWriter.write(JsonStringSchema.builder().description("a city").build()))
          .isEqualTo("{\"type\":\"string\",\"description\":\"a city\"}");
    }

    @Test
    void escapesDescriptionText() {
      assertThat(
              JsonSchemaWriter.write(
                  JsonStringSchema.builder().description("say \"hi\"\nthen stop").build()))
          .isEqualTo("{\"type\":\"string\",\"description\":\"say \\\"hi\\\"\\nthen stop\"}");
    }
  }

  @Nested
  static class Composites {

    @Test
    void writesAnObjectWithPropertiesAndRequired() {
      JsonObjectSchema schema =
          JsonObjectSchema.builder()
              .addStringProperty("city", "the city")
              .addIntegerProperty("days")
              .required("city")
              .build();

      assertThat(JsonSchemaWriter.write(schema))
          .isEqualTo(
              "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\","
                  + "\"description\":\"the city\"},\"days\":{\"type\":\"integer\"}},"
                  + "\"required\":[\"city\"]}");
    }

    @Test
    void writesArraysWithItemSchemas() {
      JsonArraySchema schema = JsonArraySchema.builder().items(new JsonStringSchema()).build();

      assertThat(JsonSchemaWriter.write(schema))
          .isEqualTo("{\"type\":\"array\",\"items\":{\"type\":\"string\"}}");
    }

    @Test
    void writesEnumerations() {
      JsonEnumSchema schema = JsonEnumSchema.builder().enumValues(List.of("c", "f")).build();

      assertThat(JsonSchemaWriter.write(schema))
          .isEqualTo("{\"type\":\"string\",\"enum\":[\"c\",\"f\"]}");
    }

    @Test
    void writesReferences() {
      assertThat(
              JsonSchemaWriter.write(JsonReferenceSchema.builder().reference("#/$defs/X").build()))
          .isEqualTo("{\"$ref\":\"#/$defs/X\"}");
    }

    @Test
    void passesRawSchemaThroughVerbatim() {
      // A caller who already has JSON Schema text should get it back unaltered.
      String raw = "{\"type\":\"object\",\"additionalProperties\":false}";

      assertThat(JsonSchemaWriter.write(JsonRawSchema.from(raw))).isEqualTo(raw);
    }

    @Test
    void nestsObjectsWithoutLosingStructure() {
      JsonObjectSchema schema =
          JsonObjectSchema.builder()
              .addProperty("where", JsonObjectSchema.builder().addStringProperty("city").build())
              .build();

      assertThat(JsonSchemaWriter.write(schema))
          .isEqualTo(
              "{\"type\":\"object\",\"properties\":{\"where\":{\"type\":\"object\","
                  + "\"properties\":{\"city\":{\"type\":\"string\"}}}}}");
    }
  }

  @Nested
  static class Degradation {

    @Test
    void writesAnEmptyObjectForANullSchema() {
      // A tool that declares no parameters is legitimate and must not fail rendering.
      assertThat(JsonSchemaWriter.write(null)).isEqualTo("{}");
    }

    @Test
    void writesAnObjectWithNoPropertiesWhenNoneAreDeclared() {
      assertThat(JsonSchemaWriter.write(JsonObjectSchema.builder().build()))
          .isEqualTo("{\"type\":\"object\"}");
    }
  }
}
