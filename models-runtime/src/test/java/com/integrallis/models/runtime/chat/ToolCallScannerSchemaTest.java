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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.ToolSpec;
import java.util.List;
import org.junit.jupiter.api.Test;

class ToolCallScannerSchemaTest {

  @Test
  void recoversMiniCpmTaggedArgumentsUsingTheDeclaredSchema() {
    ToolSpec weather =
        new ToolSpec(
            "get-weather-for-zipcode",
            "Gets weather for a zipcode",
            """
            {"type":"object","properties":{"zipcode":{"type":"string"}},"required":["zipcode"]}
            """);

    ToolCallScanner.Result result =
        ToolCallScanner.scan(
            "<function name=\"get-weather-for-zipcode\"><param name=\"zipcode\">88252</param></function>",
            ToolSyntax.MINICPM5,
            List.of(weather));

    assertThat(result.content()).isEmpty();
    assertThat(result.toolCalls())
        .singleElement()
        .satisfies(
            call -> {
              assertThat(call.name()).isEqualTo("get-weather-for-zipcode");
              assertThat(call.argumentsJson()).isEqualTo("{\"zipcode\":\"88252\"}");
            });
  }

  @Test
  void preservesMiniCpmOutputWhenTheTaggedValueCannotSatisfyTheSchema() {
    ToolSpec tool =
        new ToolSpec(
            "set_count",
            "Sets a count",
            """
            {"type":"object","properties":{"count":{"type":"integer"}},"required":["count"]}
            """);
    String malformed = "<function name=\"set_count\"><param name=\"count\">many</param></function>";

    ToolCallScanner.Result result =
        ToolCallScanner.scan(malformed, ToolSyntax.MINICPM5, List.of(tool));

    assertThat(result.hasCalls()).isFalse();
    assertThat(result.content()).isEqualTo(malformed);
  }
}
