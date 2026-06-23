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
 * Bridge from {@code models} embedders into {@code java-vectors} storage.
 *
 * <p>{@link com.integrallis.models.embedding.EmbeddingBackend} is the minimal SPI an embedding
 * model implements. {@link com.integrallis.models.embedding.VectorCollectionEmbeddingSink} consumes
 * that SPI and routes the produced {@code float[]} directly into a {@link
 * com.integrallis.vectors.db.VectorCollection}. Closes ROADMAP II.12 F4 — every framework adapter
 * used to roll its own bridge; this consolidates the surface.
 */
package com.integrallis.models.embedding;
