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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * A decoded {@code ModelProto}.
 *
 * @param irVersion the ONNX IR version
 * @param producerName the exporting tool
 * @param producerVersion the exporting tool's version
 * @param opsets operator-set versions by domain; the default domain is recorded as {@code ai.onnx}
 * @param graph the main graph
 */
public record OnnxModel(
    long irVersion,
    String producerName,
    String producerVersion,
    Map<String, Long> opsets,
    OnnxGraph graph) {

  /** Defensive copy. */
  public OnnxModel {
    opsets = Map.copyOf(opsets);
  }

  /** Decodes a serialized {@code ModelProto}. */
  public static OnnxModel parse(byte[] bytes) {
    return OnnxParser.parseModel(bytes);
  }

  /** Reads and decodes an {@code .onnx} file. */
  public static OnnxModel load(Path path) throws IOException {
    return parse(Files.readAllBytes(path));
  }

  /** The {@code ai.onnx} operator-set version, or {@code -1} when the file declares none. */
  public long defaultOpset() {
    return opsets.getOrDefault("ai.onnx", -1L);
  }
}
