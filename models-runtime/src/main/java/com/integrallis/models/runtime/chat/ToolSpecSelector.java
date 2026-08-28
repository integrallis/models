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

import com.integrallis.models.api.AuxiliaryTextGenerationModel;
import com.integrallis.models.api.EmbeddingBackend;
import com.integrallis.models.api.ToolSpec;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Selects a compact prompt-time tool set with a model's trained contrastive head.
 *
 * <p>Small declarations pass through unchanged. Larger declarations are embedded and ranked, with
 * the tool-schema vectors cached independently from per-request query vectors. The bounded cache
 * prevents applications that generate schemas dynamically from retaining them without limit.
 */
public final class ToolSpecSelector {

  /** Number of declared tools the Needle 2 training and qualification path places in a prompt. */
  public static final int DEFAULT_TOOL_LIMIT = 5;

  private static final int DEFAULT_INDEX_CACHE_SIZE = 32;

  private final AuxiliaryTextGenerationModel model;
  private final EmbeddingBackend embeddings;
  private final int toolLimit;
  private final Map<List<ToolSpec>, ToolSpecRetriever> indexes;

  /** Creates a selector using the qualified five-tool prompt policy. */
  public ToolSpecSelector(AuxiliaryTextGenerationModel model) {
    this(model, DEFAULT_TOOL_LIMIT, DEFAULT_INDEX_CACHE_SIZE);
  }

  /** Creates a selector with explicit prompt and cache bounds. */
  public ToolSpecSelector(AuxiliaryTextGenerationModel model, int toolLimit, int indexCacheSize) {
    this.model = Objects.requireNonNull(model, "model");
    if (!model.supportsContrastiveEncoding()) {
      throw new IllegalArgumentException("model does not expose a contrastive encoding head");
    }
    if (toolLimit <= 0) {
      throw new IllegalArgumentException("toolLimit must be > 0: " + toolLimit);
    }
    if (indexCacheSize <= 0) {
      throw new IllegalArgumentException("indexCacheSize must be > 0: " + indexCacheSize);
    }
    this.toolLimit = toolLimit;
    this.embeddings = new ModelEmbeddings(model);
    this.indexes =
        new LinkedHashMap<>(16, 0.75f, true) {
          @Override
          protected boolean removeEldestEntry(Map.Entry<List<ToolSpec>, ToolSpecRetriever> eldest) {
            return size() > indexCacheSize;
          }
        };
  }

  /**
   * Returns the tools to place in the prompt, ranked by relevance when the declaration exceeds the
   * configured prompt limit.
   */
  public List<ToolSpec> select(String query, List<ToolSpec> tools) {
    Objects.requireNonNull(query, "query");
    List<ToolSpec> declared = List.copyOf(Objects.requireNonNull(tools, "tools"));
    if (declared.size() <= toolLimit) {
      return declared;
    }
    return retriever(declared).select(query, toolLimit).stream()
        .map(ToolSpecRetriever.Match::tool)
        .toList();
  }

  private ToolSpecRetriever retriever(List<ToolSpec> tools) {
    synchronized (indexes) {
      ToolSpecRetriever existing = indexes.get(tools);
      if (existing != null) {
        return existing;
      }
      ToolSpecRetriever created = new ToolSpecRetriever(embeddings, tools);
      indexes.put(tools, created);
      return created;
    }
  }

  private record ModelEmbeddings(AuxiliaryTextGenerationModel model) implements EmbeddingBackend {

    @Override
    public int dimension() {
      return model.contrastiveDimension();
    }

    @Override
    public float[] embed(String text) {
      return model.encodeContrastive(text);
    }
  }
}
