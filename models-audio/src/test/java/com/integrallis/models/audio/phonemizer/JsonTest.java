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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class JsonTest {

  @Test
  void parsesEveryValueKindAndEscapes() {
    Map<String, Object> value =
        Json.parseObject(
            " {\"a\": \"x\\u00e9\\n\\\"\\\\\\/\\b\\f\\r\\t\", \"b\": [1, -2.5, true, false, null, {}],"
                + " \"c\": {\"d\": 3e2}, \"e\": []} ");
    assertThat(value.get("a")).isEqualTo("xé\n\"\\/\b\f\r\t");
    @SuppressWarnings("unchecked")
    List<Object> list = (List<Object>) value.get("b");
    assertThat(list).hasSize(6);
    assertThat(list.get(0)).isEqualTo(1L);
    assertThat(list.get(1)).isEqualTo(-2.5);
    assertThat(list.get(2)).isEqualTo(true);
    assertThat(list.get(3)).isEqualTo(false);
    assertThat(list.get(4)).isNull();
    assertThat(list.get(5)).isEqualTo(Map.of());
    assertThat(((Map<?, ?>) value.get("c")).get("d")).isEqualTo(300.0);
    assertThat(value.get("e")).isEqualTo(List.of());
    assertThat(Json.parse("\"plain\"")).isEqualTo("plain");
    assertThat(Json.parse("42")).isEqualTo(42L);
  }

  @Test
  void rejectsMalformedInput() {
    assertThatThrownBy(() -> Json.parse("{\"a\" 1}")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parse("[1 2]")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parse("{\"a\":1}x")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parse("\"open")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parse("tru")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parse("\"\\q\"")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parse("")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parse("?")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Json.parseObject("[1]")).isInstanceOf(IllegalArgumentException.class);
  }
}
