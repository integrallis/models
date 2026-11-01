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
package com.integrallis.models.backend.onnx;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

/**
 * One node attribute, decoded from {@code AttributeProto}.
 *
 * <p>Only the value slot named by {@link #type()} is meaningful; the others hold their empty
 * defaults. Sub-graph attributes are not decoded because no supported operator carries one.
 *
 * @param name the attribute name
 * @param type which value slot is populated
 * @param f the single float value
 * @param i the single integer value
 * @param s the single string value
 * @param t the single tensor value, or {@code null}
 * @param floats the float list
 * @param ints the integer list
 * @param strings the string list
 */
public record OnnxAttribute(
    String name,
    Type type,
    float f,
    long i,
    String s,
    Tensor t,
    float[] floats,
    long[] ints,
    List<String> strings) {

  /** The {@code AttributeProto.AttributeType} codes this module decodes. */
  public enum Type {
    /** No value. */
    UNDEFINED(0),
    /** Single float. */
    FLOAT(1),
    /** Single integer. */
    INT(2),
    /** Single string. */
    STRING(3),
    /** Single tensor. */
    TENSOR(4),
    /** Sub-graph; recorded but not decoded. */
    GRAPH(5),
    /** Float list. */
    FLOATS(6),
    /** Integer list. */
    INTS(7),
    /** String list. */
    STRINGS(8);

    private final int code;

    Type(int code) {
      this.code = code;
    }

    static Type fromCode(long code) {
      for (Type type : values()) {
        if (type.code == code) {
          return type;
        }
      }
      return UNDEFINED;
    }
  }

  /** Validating constructor that normalizes {@code null} lists to empty ones. */
  public OnnxAttribute {
    floats = floats == null ? new float[0] : floats.clone();
    ints = ints == null ? new long[0] : ints.clone();
    strings = strings == null ? List.of() : List.copyOf(strings);
  }

  /** The float list as a copy. */
  @Override
  public float[] floats() {
    return floats.clone();
  }

  /** The integer list as a copy. */
  @Override
  public long[] ints() {
    return ints.clone();
  }

  static String utf8(byte[] bytes) {
    return new String(bytes, StandardCharsets.UTF_8);
  }

  @Override
  public String toString() {
    return switch (type) {
      case FLOAT -> name + "=" + f;
      case INT -> name + "=" + i;
      case STRING -> name + "=" + s;
      case TENSOR -> name + "=" + t;
      case FLOATS -> name + "=" + Arrays.toString(floats);
      case INTS -> name + "=" + Arrays.toString(ints);
      case STRINGS -> name + "=" + strings;
      default -> name + ":" + type;
    };
  }
}
