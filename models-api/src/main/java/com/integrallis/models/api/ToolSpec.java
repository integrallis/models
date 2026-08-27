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

import java.util.Objects;

/**
 * A tool the model may call, as declared by the application.
 *
 * <p>{@code inputSchema} is JSON Schema text carried verbatim by this dependency-free API module. A
 * runtime may parse it to compile a decoding constraint. Spring AI supplies the schema as a string
 * already; LangChain4j models it as a typed tree, so its adapter serialises before constructing a
 * {@code ToolSpec}.
 */
public record ToolSpec(String name, String description, String inputSchema) {

  public ToolSpec {
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("name must not be blank");
    }
    Objects.requireNonNull(inputSchema, "inputSchema");
    description = description == null ? "" : description;
  }
}
