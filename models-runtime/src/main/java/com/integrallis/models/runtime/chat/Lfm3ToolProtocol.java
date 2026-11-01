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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import java.util.*;

/** LFM3 text/tool protocol. Tagged argument values are recovered using the declared schemas. */
final class Lfm3ToolProtocol {
  private Lfm3ToolProtocol() {}

  static ModelPrompt render(List<ChatMessage> messages, List<ToolSpec> tools, boolean thinking) {
    var out = ModelPrompt.builder().control("<|startoftext|>");
    int first = messages.getFirst().role() == ChatRole.SYSTEM ? 1 : 0, lastUser = -1;
    String system = first == 1 ? messages.getFirst().text() : "";
    if (!system.isEmpty() || !tools.isEmpty()) {
      out.control("<|im_start|>system\n").text(system);
      if (!tools.isEmpty()) {
        if (!system.isEmpty()) out.text("\n\n");
        out.control("<tools>\n");
        for (int i = 0; i < tools.size(); i++) {
          if (i > 0) out.text("\n");
          var tool = tools.get(i);
          var fn = new LinkedHashMap<String, Object>();
          fn.put("name", tool.name());
          fn.put("description", tool.description());
          fn.put("parameters", object(FlintToolProtocol.parseJson(tool.inputSchema())));
          out.text(FlintToolProtocol.json(fn, true));
        }
        out.control("\n</tools>");
      }
      out.control("<|im_end|>\n");
    }
    for (int i = first; i < messages.size(); i++)
      if (messages.get(i).role() == ChatRole.USER) lastUser = i;
    for (int i = first; i < messages.size(); i++) {
      var message = messages.get(i);
      if (message.hasToolCalls() && message.role() != ChatRole.ASSISTANT)
        throw new IllegalArgumentException("Only assistants can call LFM3 tools");
      out.control("<|im_start|>" + message.role().templateName() + "\n");
      if (message.role() == ChatRole.ASSISTANT) {
        String text = message.text();
        if (i < lastUser) {
          int end = text.lastIndexOf("</think>");
          if (end >= 0) text = text.substring(end + 8).strip();
          out.control("<think></think>");
        } else if (!text.contains("</think>")) out.control("<think></think>");
        appendAssistant(out, text);
        for (int j = 0; j < message.toolCalls().size(); j++) {
          if (j > 0) out.text("\n\n");
          var call = message.toolCalls().get(j);
          identifier(call.name());
          out.control("<tool_call>\n<function=").text(call.name()).control(">\n");
          for (var arg : object(FlintToolProtocol.parseJson(call.argumentsJson())).entrySet()) {
            identifier(arg.getKey());
            out.control("<parameter=").text(arg.getKey()).control(">\n");
            out.text(
                    arg.getValue() instanceof String v
                        ? v
                        : FlintToolProtocol.json(arg.getValue(), true))
                .control("\n</parameter>\n");
          }
          out.control("</function>\n</tool_call>");
        }
      } else if (message.role() == ChatRole.TOOL) {
        // Results remain TEXT, even if their characters resemble model controls.
        if (message.text().contains("<tool_response>")) out.text(message.text());
        else out.control("<tool_response>\n").text(message.text()).control("\n</tool_response>");
      } else out.text(message.text());
      out.control("<|im_end|>\n");
    }
    return out.control("<|im_start|>assistant\n" + (thinking ? "<think>" : "<think></think>"))
        .build();
  }

  private static void appendAssistant(ModelPrompt.Builder out, String text) {
    int cursor = 0;
    while (cursor < text.length()) {
      int a = text.indexOf("<think>", cursor), b = text.indexOf("</think>", cursor);
      int next = a < 0 ? b : b < 0 ? a : Math.min(a, b);
      if (next < 0) {
        out.text(text.substring(cursor));
        break;
      }
      out.text(text.substring(cursor, next));
      String marker = next == a ? "<think>" : "</think>";
      out.control(marker);
      cursor = next + marker.length();
    }
  }

  static ToolCallScanner.Result scan(String generated, List<ToolSpec> tools) {
    if (generated == null || generated.isEmpty()) return ToolCallScanner.Result.plainText("");
    try {
      var declarations = new LinkedHashMap<String, ToolSpec>();
      for (var tool : tools)
        if (declarations.putIfAbsent(tool.name(), tool) != null) throw malformed();
      var calls = new ArrayList<ToolCall>();
      var prose = new StringBuilder();
      int cursor = 0;
      while (cursor < generated.length()) {
        int start = generated.indexOf("<tool_call>", cursor),
            thought = generated.indexOf("<think>", cursor);
        if (thought >= 0 && (start < 0 || thought < start)) {
          int end = generated.indexOf("</think>", thought + 7);
          if (end < 0) throw malformed();
          prose.append(generated, cursor, end + 8);
          cursor = end + 8;
          continue;
        }
        if (start < 0) {
          prose.append(generated, cursor, generated.length());
          break;
        }
        prose.append(generated, cursor, start);
        int end = generated.indexOf("</tool_call>", start + 11);
        if (end < 0) throw malformed();
        String body = generated.substring(start + 11, end).strip();
        if (!body.startsWith("<function=") || !body.endsWith("</function>")) throw malformed();
        int nameEnd = body.indexOf('>', 10);
        if (nameEnd < 0) throw malformed();
        String name = body.substring(10, nameEnd);
        identifier(name);
        var declaration = declarations.get(name);
        if (declaration == null) throw malformed();
        var schema = object(FlintToolProtocol.parseJson(declaration.inputSchema()));
        var properties =
            schema.containsKey("properties")
                ? object(schema.get("properties"))
                : Map.<String, Object>of();
        var arguments = new LinkedHashMap<String, Object>();
        int at = nameEnd + 1, limit = body.length() - 11;
        while (at < limit) {
          while (at < limit && Character.isWhitespace(body.charAt(at))) at++;
          if (at == limit) break;
          if (!body.startsWith("<parameter=", at)) throw malformed();
          int keyEnd = body.indexOf('>', at + 11);
          if (keyEnd < 0) throw malformed();
          String key = body.substring(at + 11, keyEnd);
          identifier(key);
          int valueEnd = body.indexOf("</parameter>", keyEnd + 1);
          if (valueEnd < 0
              || valueEnd > limit
              || arguments.containsKey(key)
              || !properties.containsKey(key)) throw malformed();
          String raw = body.substring(keyEnd + 1, valueEnd);
          // Exactly one formatting newline belongs to each delimiter; preserve user whitespace.
          if (raw.startsWith("\n")) raw = raw.substring(1);
          if (raw.endsWith("\n")) raw = raw.substring(0, raw.length() - 1);
          if (raw.contains("<parameter=")
              || raw.contains("<tool_call>")
              || raw.contains("</function>")) throw malformed();
          var property = object(properties.get(key));
          Object type = property.get("type");
          boolean isString =
              "string".equals(type)
                  || (type instanceof List<?> list
                      && list.contains("string")
                      && !raw.equals("null"));
          Object value = isString ? raw : FlintToolProtocol.parseJson(raw);
          if (!FlintToolProtocol.matches(value, property)) throw malformed();
          arguments.put(key, value);
          at = valueEnd + 12;
        }
        if (!FlintToolProtocol.matches(arguments, schema)) throw malformed();
        calls.add(ToolCall.of(calls.size(), name, FlintToolProtocol.json(arguments, false)));
        cursor = end + 12;
      }
      return calls.isEmpty()
          ? ToolCallScanner.Result.plainText(generated)
          : new ToolCallScanner.Result(prose.toString().strip(), calls);
    } catch (IllegalArgumentException malformed) {
      return ToolCallScanner.Result.plainText(generated);
    }
  }

  private static void identifier(String name) {
    if (name.isEmpty()
        || name.indexOf('<') >= 0
        || name.indexOf('>') >= 0
        || name.indexOf('\n') >= 0
        || name.indexOf('\r') >= 0) throw malformed();
  }

  private static Map<String, Object> object(Object value) {
    if (!(value instanceof Map<?, ?> m)) throw malformed();
    var result = new LinkedHashMap<String, Object>();
    for (var entry : m.entrySet()) {
      if (!(entry.getKey() instanceof String name)) throw malformed();
      result.put(name, entry.getValue());
    }
    return result;
  }

  private static IllegalArgumentException malformed() {
    return new IllegalArgumentException("Malformed or unsupported LFM3 tool call");
  }
}
