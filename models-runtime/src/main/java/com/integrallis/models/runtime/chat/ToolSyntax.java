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

import java.util.Objects;

/**
 * How a model family expresses tool calls, in both directions.
 *
 * <p>Rendering and parsing read this same descriptor, which is why the runtime needs no parser per
 * model family. llama.cpp reached the same conclusion from the opposite direction: it collapsed
 * roughly twenty hand-written family parsers into a small inferred taxonomy. It infers the values
 * by rendering the model's Jinja template with probe inputs; this runtime has no Jinja engine by
 * design, so the values are filled in per {@link ChatTemplate} constant instead.
 *
 * <p>Only verified families are populated. Anything unconfirmed maps to {@link #NONE}, so adapters
 * refuse tools outright rather than emitting a format the model was never trained on.
 */
public record ToolSyntax(
    Mode mode,
    String sectionStart,
    String sectionEnd,
    String nameField,
    String argsField,
    boolean arrayWrapped,
    boolean parallelCalls,
    ResultStyle resultStyle,
    String resultStart,
    String resultEnd) {

  /** Shape of an emitted call. */
  public enum Mode {
    /** The family has no trained tool-call format. */
    NONE,
    /** A bare JSON object, with no surrounding delimiter (Llama 3.x). */
    JSON_NATIVE,
    /** A JSON object fenced by delimiters (Qwen, Hermes, SmolLM3). */
    TAG_WITH_JSON,
    /** OpenAI Harmony: the function name is in the message recipient and the body is JSON. */
    HARMONY,
    /** Tagged key/value arguments rather than JSON (Qwen3-Coder, Functionary). */
    TAG_WITH_TAGGED
  }

  /** How a tool result is fed back into the conversation. */
  public enum ResultStyle {
    /** Wrapped in delimiters inside a {@code user} turn, coalescing consecutive results (Qwen). */
    USER_WRAPPED,
    /** A first-class {@code tool} turn (Hermes, Granite). */
    TOOL_ROLE,
    /** Llama 3.x's {@code ipython} role. */
    IPYTHON,
    /** A normal {@code user} turn with no wrapper around the serialized result (Needle 2). */
    USER_PLAIN,
    /** A named Harmony tool author, such as {@code functions.get_weather}. */
    HARMONY_NAMED
  }

  /** No trained tool-call format; tools must be refused rather than approximated. */
  public static final ToolSyntax NONE =
      new ToolSyntax(
          Mode.NONE, "", "", "name", "arguments", false, false, ResultStyle.TOOL_ROLE, "", "");

  /**
   * Qwen2.5 / Qwen3. Calls are fenced by {@code <tool_call>}; results come back through a {@code
   * user} turn wrapped in {@code <tool_response>}, with consecutive results coalesced.
   */
  public static final ToolSyntax QWEN =
      new ToolSyntax(
          Mode.TAG_WITH_JSON,
          "<tool_call>",
          "</tool_call>",
          "name",
          "arguments",
          false,
          true,
          ResultStyle.USER_WRAPPED,
          "<tool_response>",
          "</tool_response>");

  /**
   * Hermes 2/3 and SmolLM3. Call syntax is identical to {@link #QWEN} — Qwen adopted it — but
   * results use a real {@code tool} role rather than a wrapped user turn.
   */
  public static final ToolSyntax HERMES =
      new ToolSyntax(
          Mode.TAG_WITH_JSON,
          "<tool_call>",
          "</tool_call>",
          "name",
          "arguments",
          false,
          true,
          ResultStyle.TOOL_ROLE,
          "",
          "");

  /**
   * SmolLM3. Calls use Qwen's tagged JSON shape, but tool results return as plain user turns rather
   * than inside {@code <tool_response>} delimiters.
   */
  public static final ToolSyntax SMOLLM3 =
      new ToolSyntax(
          Mode.TAG_WITH_JSON,
          "<tool_call>",
          "</tool_call>",
          "name",
          "arguments",
          false,
          true,
          ResultStyle.USER_PLAIN,
          "",
          "");

  /**
   * Llama 3.1 / 3.3. The only major family keyed on {@code parameters} rather than {@code
   * arguments}, and its template raises outright on more than one call per turn.
   */
  public static final ToolSyntax LLAMA3 =
      new ToolSyntax(
          Mode.JSON_NATIVE,
          "",
          "",
          "name",
          "parameters",
          false,
          false,
          ResultStyle.IPYTHON,
          "",
          "");

  /**
   * Needle 2. Calls are emitted as one JSON array inside {@code <tool_call>}; tool results return
   * as ordinary user text on the next turn.
   */
  public static final ToolSyntax NEEDLE2 =
      new ToolSyntax(
          Mode.TAG_WITH_JSON,
          "<tool_call>",
          "</tool_call>",
          "name",
          "arguments",
          true,
          true,
          ResultStyle.USER_PLAIN,
          "",
          "");

  /** GPT-OSS Harmony calls route JSON arguments to a named function on the commentary channel. */
  public static final ToolSyntax HARMONY =
      new ToolSyntax(
          Mode.HARMONY,
          "<|start|>assistant<|channel|>commentary to=functions.",
          "<|call|>",
          "name",
          "arguments",
          false,
          true,
          ResultStyle.HARMONY_NAMED,
          "<|start|>functions.",
          "<|end|>");

  /**
   * Gemma 4. Emits {@code <|tool_call>call:name{key:value,...}<tool_call|>} — note the delimiters
   * are asymmetric, with the pipe moving from the leading to the trailing position.
   *
   * <p>Arguments are tagged key/value pairs rather than JSON, and strings are delimited by {@code
   * <|"|>}. llama.cpp gives this family a bespoke grammar rather than handling it generically, so
   * {@link #parsable()} is false here: the format exists and is recorded, but recovering calls from
   * it needs more than a delimiter scan.
   */
  public static final ToolSyntax GEMMA4 =
      new ToolSyntax(
          Mode.TAG_WITH_TAGGED,
          "<|tool_call>",
          "<tool_call|>",
          "name",
          "arguments",
          false,
          true,
          ResultStyle.TOOL_ROLE,
          "<|tool_response>",
          "<tool_response|>");

  /**
   * MiniCPM5. Emits XML: {@code <function name="foo"><param name="bar">value</param></function>},
   * with results returned through a {@code user} turn wrapped in {@code <tool_response>}.
   *
   * <p>Tagged arguments carry no type information, so turning {@code <param name="n">1</param>}
   * into JSON requires the declared tool schema to know whether {@code 1} is a number or a string.
   * That is why {@link #parsable()} is false: the scanner is given generated text, not schemas.
   */
  public static final ToolSyntax MINICPM5 =
      new ToolSyntax(
          Mode.TAG_WITH_TAGGED,
          "<function name=\"",
          "</function>",
          "name",
          "arguments",
          false,
          true,
          ResultStyle.USER_WRAPPED,
          "<tool_response>",
          "</tool_response>");

  public ToolSyntax {
    Objects.requireNonNull(mode, "mode");
    Objects.requireNonNull(resultStyle, "resultStyle");
    sectionStart = sectionStart == null ? "" : sectionStart;
    sectionEnd = sectionEnd == null ? "" : sectionEnd;
    resultStart = resultStart == null ? "" : resultStart;
    resultEnd = resultEnd == null ? "" : resultEnd;
    if (nameField == null || nameField.isBlank()) {
      throw new IllegalArgumentException("nameField must not be blank");
    }
    if (argsField == null || argsField.isBlank()) {
      throw new IllegalArgumentException("argsField must not be blank");
    }
    if (mode == Mode.TAG_WITH_JSON || mode == Mode.TAG_WITH_TAGGED || mode == Mode.HARMONY) {
      if (sectionStart.isBlank()) {
        throw new IllegalArgumentException("sectionStart is required for mode " + mode);
      }
      if (sectionEnd.isBlank()) {
        throw new IllegalArgumentException("sectionEnd is required for mode " + mode);
      }
    }
    if (resultStyle == ResultStyle.USER_WRAPPED) {
      if (resultStart.isBlank()) {
        throw new IllegalArgumentException("resultStart is required for " + resultStyle);
      }
      if (resultEnd.isBlank()) {
        throw new IllegalArgumentException("resultEnd is required for " + resultStyle);
      }
    }
  }

  /** Whether this family has a trained tool-call format at all. */
  public boolean supportsTools() {
    return mode != Mode.NONE;
  }

  /**
   * Whether {@link ToolCallScanner} can recover calls in this format from generated text alone.
   *
   * <p>Distinct from {@link #supportsTools()}. JSON-shaped calls are self-describing, so a
   * delimiter scan plus brace matching recovers them. Tagged calls are not: {@code <param
   * name="n">1</param>} carries no type, so producing the JSON both frameworks expect requires the
   * declared tool schema, which the scanner is not given. llama.cpp reaches the same split — it
   * builds a parser per tool from {@code inputs.tools} for tagged formats, and hand-writes a
   * grammar for Gemma 4.
   *
   * <p>Callers that only have generated text should refuse tools when this is false. Framework
   * adapters also have the declared schemas and should use {@link #parsableWithSchemas()}.
   */
  public boolean parsable() {
    return mode == Mode.JSON_NATIVE || mode == Mode.TAG_WITH_JSON || mode == Mode.HARMONY;
  }

  /** Whether calls can be recovered when the request's declared JSON Schemas are available. */
  public boolean parsableWithSchemas() {
    return parsable() || this.equals(MINICPM5);
  }
}
