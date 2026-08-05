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
package com.integrallis.models.api;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ToolCallTest {

  @Nested
  static class Validation {

    @Test
    void keepsArgumentsAsAnOpaqueString() {
      // The runtime never parses tool arguments; both target frameworks want raw JSON text.
      ToolCall call = new ToolCall("000000000", "get_weather", "{\"city\":\"Austin\"}");

      assertThat(call.name()).isEqualTo("get_weather");
      assertThat(call.argumentsJson()).isEqualTo("{\"city\":\"Austin\"}");
    }

    @Test
    void rejectsABlankName() {
      assertThatThrownBy(() -> new ToolCall("000000000", "", "{}"))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("name");
    }

    @Test
    void defaultsMissingArgumentsToAnEmptyObject() {
      assertThat(new ToolCall("000000000", "ping", null).argumentsJson()).isEqualTo("{}");
    }
  }

  @Nested
  static class Identifiers {

    @Test
    void synthesizesNineDigitIdentifiers() {
      // Mistral templates raise unless the id is exactly nine characters; nine digits is
      // harmless for every other family, so it is the portable choice.
      assertThat(ToolCall.syntheticId(0)).isEqualTo("000000000");
      assertThat(ToolCall.syntheticId(7)).isEqualTo("000000007");
      assertThat(ToolCall.syntheticId(123456789)).hasSize(9);
    }

    @Test
    void fillsInASyntheticIdentifierWhenTheModelOmitsOne() {
      assertThat(ToolCall.of(3, "ping", "{}").id()).isEqualTo("000000003");
    }

    @Test
    void rejectsANegativeIndex() {
      assertThatThrownBy(() -> ToolCall.syntheticId(-1))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }
}
