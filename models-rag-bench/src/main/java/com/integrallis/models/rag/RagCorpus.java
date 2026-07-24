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

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** The versioned documents and evaluation cases used by every RAG runner. */
public record RagCorpus(List<RagDocument> documents, List<RagCase> cases) {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  public RagCorpus {
    documents = List.copyOf(Objects.requireNonNull(documents, "documents"));
    cases = List.copyOf(Objects.requireNonNull(cases, "cases"));
    requireUniqueDocuments(documents);
    requireValidCases(documents, cases);
  }

  /** Loads the controlled classpath corpus. */
  public static RagCorpus loadDefault() {
    return new RagCorpus(
        read("/rag/documents.json", new TypeReference<List<RagDocument>>() {}),
        read("/rag/cases.json", new TypeReference<List<RagCase>>() {}));
  }

  /** Returns a stable SHA-256 over the parsed corpus content. */
  public String fingerprint() {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      digest.update(MAPPER.writeValueAsString(documents).getBytes(StandardCharsets.UTF_8));
      digest.update((byte) '\n');
      digest.update(MAPPER.writeValueAsString(cases).getBytes(StandardCharsets.UTF_8));
      return HexFormat.of().formatHex(digest.digest());
    } catch (IOException | NoSuchAlgorithmException failure) {
      throw new IllegalStateException("cannot fingerprint RAG corpus", failure);
    }
  }

  private static <T> T read(String resource, TypeReference<T> type) {
    try (InputStream input = RagCorpus.class.getResourceAsStream(resource)) {
      if (input == null) {
        throw new IllegalStateException("missing RAG resource: " + resource);
      }
      return MAPPER.readValue(input, type);
    } catch (IOException failure) {
      throw new IllegalStateException("cannot read RAG resource: " + resource, failure);
    }
  }

  private static void requireUniqueDocuments(List<RagDocument> documents) {
    Set<String> ids = new HashSet<>();
    for (RagDocument document : documents) {
      if (!ids.add(document.id())) {
        throw new IllegalArgumentException("duplicate document id: " + document.id());
      }
    }
  }

  private static void requireValidCases(List<RagDocument> documents, List<RagCase> cases) {
    Set<String> documentIds = new HashSet<>();
    documents.forEach(document -> documentIds.add(document.id()));
    Set<String> caseIds = new HashSet<>();
    for (RagCase testCase : cases) {
      if (!caseIds.add(testCase.id())) {
        throw new IllegalArgumentException("duplicate case id: " + testCase.id());
      }
      if (!documentIds.containsAll(testCase.relevantDocumentIds())) {
        throw new IllegalArgumentException("case references an unknown document: " + testCase.id());
      }
    }
  }
}
