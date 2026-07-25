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
package com.integrallis.models.rag;

import java.util.Arrays;
import java.util.Locale;
import java.util.stream.Collectors;

/** Versioned RAG workloads available to the controlled benchmark. */
public enum RagWorkload {
  GENERAL("general", "/rag/documents.json", "/rag/cases.json"),
  CODING("coding", "/rag/coding/documents.json", "/rag/coding/cases.json"),
  FINANCE("finance", "/rag/finance/documents.json", "/rag/finance/cases.json"),
  HEALTHCARE("healthcare", "/rag/healthcare/documents.json", "/rag/healthcare/cases.json"),
  LEGAL("legal", "/rag/legal/documents.json", "/rag/legal/cases.json"),
  MATH("math", "/rag/math/documents.json", "/rag/math/cases.json"),
  MULTILINGUAL("multilingual", "/rag/multilingual/documents.json", "/rag/multilingual/cases.json"),
  SQL("sql", "/rag/sql/documents.json", "/rag/sql/cases.json"),
  TRANSPORTATION(
      "transportation", "/rag/transportation/documents.json", "/rag/transportation/cases.json");

  private final String id;
  private final String documentsResource;
  private final String casesResource;

  RagWorkload(String id, String documentsResource, String casesResource) {
    this.id = id;
    this.documentsResource = documentsResource;
    this.casesResource = casesResource;
  }

  /** Stable CLI and report identifier. */
  public String id() {
    return id;
  }

  String documentsResource() {
    return documentsResource;
  }

  String casesResource() {
    return casesResource;
  }

  /** Resolves a CLI identifier. */
  public static RagWorkload parse(String value) {
    for (RagWorkload workload : values()) {
      if (workload.id.equals(value.toLowerCase(Locale.ROOT))) {
        return workload;
      }
    }
    String supported =
        Arrays.stream(values()).map(RagWorkload::id).collect(Collectors.joining(", "));
    throw new IllegalArgumentException("workload must be one of " + supported);
  }
}
