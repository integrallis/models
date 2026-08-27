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
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.chat.ToolSyntax;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;

/** Builds token constraints for finite tool-call output spaces. */
public final class ToolCallTokenConstraints {

  private static final int MAX_ALTERNATIVES = 256;

  private ToolCallTokenConstraints() {}

  public static Optional<TokenConstraint> compile(
      Tokenizer tokenizer,
      ToolSyntax syntax,
      List<ToolSpec> tools,
      Function<ToolSpec, List<String>> argumentAlternatives) {
    Objects.requireNonNull(tokenizer, "tokenizer");
    Objects.requireNonNull(syntax, "syntax");
    Objects.requireNonNull(tools, "tools");
    Objects.requireNonNull(argumentAlternatives, "argumentAlternatives");
    if (syntax.mode() != ToolSyntax.Mode.TAG_WITH_JSON
        && syntax.mode() != ToolSyntax.Mode.JSON_NATIVE) {
      return Optional.empty();
    }
    // Array-wrapped families generate a reasoning span before a possibly parallel call array.
    // A finite single-call alternative would skip that trained prefix and emit the wrong shape.
    if (syntax.arrayWrapped()) {
      return Optional.empty();
    }
    List<String> alternatives = new ArrayList<>();
    for (ToolSpec tool : tools) {
      List<String> arguments = argumentAlternatives.apply(tool);
      if (arguments == null || arguments.isEmpty()) {
        return Optional.empty();
      }
      for (String argument : arguments) {
        alternatives.add(renderCall(syntax, tool.name(), argument));
        if (alternatives.size() > MAX_ALTERNATIVES) {
          return Optional.empty();
        }
      }
    }
    return Optional.of(new StringAlternativesTokenConstraint(tokenizer, alternatives));
  }

  private static String renderCall(ToolSyntax syntax, String name, String arguments) {
    String call =
        "{"
            + quoteJson(syntax.nameField())
            + ":"
            + quoteJson(name)
            + ","
            + quoteJson(syntax.argsField())
            + ":"
            + arguments
            + "}";
    return syntax.mode() == ToolSyntax.Mode.TAG_WITH_JSON
        ? syntax.sectionStart() + call + syntax.sectionEnd()
        : call;
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
