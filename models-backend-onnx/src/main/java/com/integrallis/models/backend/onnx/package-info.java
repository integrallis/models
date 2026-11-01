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

/**
 * In-process, pure-Java ONNX graph execution; never an ONNX Runtime wrapper.
 *
 * <p>{@link com.integrallis.models.backend.onnx.OnnxModel} decodes an ONNX protobuf file with the
 * module's own wire-format reader, and the graph is executed eagerly, node by node, with Java
 * kernels. The first artifact this module exists for is Paradee-8M, an 8M-parameter text-to-speech
 * model published only as ONNX.
 */
package com.integrallis.models.backend.onnx;
