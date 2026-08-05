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
class ToolSpecTest {

  @Nested
  static class Validation {

    @Test
    void acceptsANameDescriptionAndSchema() {
      ToolSpec spec = new ToolSpec("get_weather", "Look up the forecast", "{\"type\":\"object\"}");

      assertThat(spec.name()).isEqualTo("get_weather");
      assertThat(spec.description()).isEqualTo("Look up the forecast");
      assertThat(spec.inputSchema()).isEqualTo("{\"type\":\"object\"}");
    }

    @Test
    void rejectsABlankName() {
      assertThatThrownBy(() -> new ToolSpec(" ", "d", "{}"))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("name");
    }

    @Test
    void rejectsANullSchema() {
      assertThatThrownBy(() -> new ToolSpec("t", "d", null))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("inputSchema");
    }

    @Test
    void defaultsAMissingDescriptionToEmptyRatherThanFailing() {
      // Spring AI and LangChain4j both permit a tool with no description.
      assertThat(new ToolSpec("t", null, "{}").description()).isEmpty();
    }
  }
}
