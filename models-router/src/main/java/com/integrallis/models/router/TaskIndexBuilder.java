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
package com.integrallis.models.router;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.VectorCollection;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Builds the persistent index {@link PretrainedTaskClassifier} searches.
 *
 * <p>Runs at build time, not at startup. Embedding a few thousand prompts takes long enough that
 * doing it per process would dominate the cost of routing, and the result is identical every time,
 * so it belongs in the artifact rather than in the application's boot path.
 *
 * <p>Vectors are only comparable within the model that produced them. The manifest written
 * alongside the index records which model and dimension it was built with, and the classifier
 * refuses a query embedding that does not match — an index searched with the wrong model returns
 * confident nonsense rather than an error.
 */
public final class TaskIndexBuilder {

  /** Metadata key holding an entry's task label. */
  public static final String TASK_FIELD = "task";

  /** Metadata key holding the dataset an entry's prompt came from. */
  public static final String SOURCE_FIELD = "source";

  /** Name of the manifest written beside the index. */
  public static final String MANIFEST = "task-index.properties";

  private TaskIndexBuilder() {}

  /**
   * Embeds a corpus's training split and writes it as a persistent collection.
   *
   * @param exemplars the labelled corpus
   * @param embedder embeds prompts; must be the model the classifier will later query with
   * @param modelId identifier of that model, recorded in the manifest
   * @param directory an empty or absent directory to write the index into
   * @return how many prompts were indexed
   * @throws IllegalArgumentException if the corpus is empty or the embedder returns nothing
   */
  public static int build(
      TaskExemplars exemplars, TaskEmbedder embedder, String modelId, Path directory) {
    return build(exemplars, embedder, modelId, directory, QuantizerKind.NONE);
  }

  /**
   * Embeds a corpus's training split and writes it as a persistent collection.
   *
   * @param exemplars the labelled corpus
   * @param embedder embeds prompts; must be the model the classifier will later query with
   * @param modelId identifier of that model, recorded in the manifest
   * @param directory an empty or absent directory to write the index into
   * @param quantizer how stored vectors are compressed; recorded in the manifest so the index is
   *     reopened the way it was written
   * @return how many prompts were indexed
   * @throws IllegalArgumentException if the corpus is empty or the embedder returns nothing
   */
  public static int build(
      TaskExemplars exemplars,
      TaskEmbedder embedder,
      String modelId,
      Path directory,
      QuantizerKind quantizer) {
    return build(exemplars, embedder, modelId, directory, quantizer, null, null);
  }

  /**
   * Embeds a corpus's training split and writes it as a persistent collection, prefixing each
   * exemplar with {@code documentPrefix}.
   *
   * <p>Instruction-tuned embedders expect a task prefix, and the two sides of a comparison must be
   * prefixed the way the model was trained or the vectors land in different regions of the space.
   * Both prefixes are recorded in the manifest so the classifier applies the query side itself
   * rather than depending on every caller to remember: an index that was built with a prefix and
   * queried without one is not an error anyone observes, just a quieter one.
   *
   * @param exemplars the labelled corpus
   * @param embedder embeds prompts; must be the model the classifier will later query with
   * @param modelId identifier of that model, recorded in the manifest
   * @param directory an empty or absent directory to write the index into
   * @param quantizer how stored vectors are compressed
   * @param documentPrefix prefix applied to every exemplar before embedding, or null for none
   * @param queryPrefix prefix the classifier must apply to queries, or null for none
   * @return how many prompts were indexed
   * @throws IllegalArgumentException if the corpus is empty, the embedder returns nothing, or only
   *     one of the two prefixes is supplied
   */
  public static int build(
      TaskExemplars exemplars,
      TaskEmbedder embedder,
      String modelId,
      Path directory,
      QuantizerKind quantizer,
      String documentPrefix,
      String queryPrefix) {
    Objects.requireNonNull(exemplars, "exemplars");
    // One side alone is worse than neither: it embeds exemplars and queries into different regions.
    if ((documentPrefix == null) != (queryPrefix == null)) {
      throw new IllegalArgumentException(
          "documentPrefix and queryPrefix must be supplied together (got documentPrefix="
              + documentPrefix
              + ", queryPrefix="
              + queryPrefix
              + ")");
    }
    Objects.requireNonNull(quantizer, "quantizer");
    Objects.requireNonNull(embedder, "embedder");
    Objects.requireNonNull(modelId, "modelId");
    Objects.requireNonNull(directory, "directory");

    List<TaskExemplars.Labelled> prompts = new ArrayList<>();
    exemplars.trainingPrompts().values().forEach(prompts::addAll);
    if (prompts.isEmpty()) {
      throw new IllegalArgumentException("corpus has no training prompts");
    }

    float[][] vectors = new float[prompts.size()][];
    for (int index = 0; index < prompts.size(); index++) {
      String text = prompts.get(index).prompt();
      float[] vector = embedder.embed(documentPrefix == null ? text : documentPrefix + text);
      if (vector == null || vector.length == 0) {
        throw new IllegalArgumentException("embedder returned no vector for prompt " + index);
      }
      vectors[index] = vector;
    }
    int dimension = vectors[0].length;
    for (int index = 1; index < vectors.length; index++) {
      if (vectors[index].length != dimension) {
        throw new IllegalArgumentException(
            "embedder returned "
                + vectors[index].length
                + " dimensions for prompt "
                + index
                + " but "
                + dimension
                + " for the first");
      }
    }

    try {
      Files.createDirectories(directory);
    } catch (IOException e) {
      throw new UncheckedIOException("cannot create " + directory, e);
    }

    // Flat scan rather than an approximate index: a few thousand vectors search in well under a
    // millisecond exactly, and an approximate index would make classification non-deterministic
    // for no gain at this size.
    try (VectorCollection collection =
        VectorCollection.builder()
            .dimension(dimension)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .quantizer(quantizer)
            // Without this a quantizer only adds a compressed copy beside the full-precision
            // vectors and makes the artifact bigger. This index ships inside a jar and is never
            // read for its scores, only for which label won, so the exact copy is dead weight.
            .quantizedOnly(quantizer != QuantizerKind.NONE)
            .storagePath(directory.toAbsolutePath())
            .build()) {
      List<Document> documents = new ArrayList<>(prompts.size());
      for (int index = 0; index < prompts.size(); index++) {
        TaskExemplars.Labelled labelled = prompts.get(index);
        documents.add(
            new Document(
                labelled.task() + "#" + index,
                vectors[index],
                null,
                Map.of(
                    TASK_FIELD, new MetadataValue.Str(labelled.task()),
                    SOURCE_FIELD, new MetadataValue.Str(labelled.source()))));
      }
      collection.addAll(documents);
      collection.commit();
    }

    writeManifest(
        directory,
        modelId,
        dimension,
        prompts.size(),
        exemplars,
        quantizer,
        documentPrefix,
        queryPrefix);
    return prompts.size();
  }

  /**
   * Digest of the training prompts this index was built from.
   *
   * <p>Lets a build detect an index left behind by a corpus edit. A prompt count alone would miss
   * an edit that replaced one prompt with another, which is exactly the change most likely to be
   * made by hand and forgotten.
   *
   * @param exemplars the corpus
   * @return a hex SHA-256 over every training prompt, in task then insertion order
   */
  public static String corpusDigest(TaskExemplars exemplars) {
    Objects.requireNonNull(exemplars, "exemplars");
    MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 is required by the platform", e);
    }
    exemplars
        .trainingPrompts()
        .forEach(
            (task, prompts) ->
                prompts.forEach(
                    labelled -> {
                      digest.update(task.getBytes(StandardCharsets.UTF_8));
                      digest.update((byte) 0);
                      digest.update(labelled.prompt().getBytes(StandardCharsets.UTF_8));
                      digest.update((byte) 0);
                    }));
    return HexFormat.of().formatHex(digest.digest());
  }

  /**
   * Escapes a prefix for {@code java.util.Properties} text.
   *
   * <p>The prefixes carry {@code :} and {@code =} (EmbeddingGemma's {@code task: classification |
   * query: }) and end in a significant trailing space, all three of which Properties would
   * otherwise eat on read.
   *
   * @param value the raw prefix
   * @return the escaped form
   */
  private static String escape(String value) {
    StringBuilder out = new StringBuilder(value.length() + 8);
    for (int index = 0; index < value.length(); index++) {
      char character = value.charAt(index);
      switch (character) {
        case ':', '=', '#', '!', '\\' -> out.append('\\').append(character);
        case ' ' -> out.append("\\ ");
        default -> out.append(character);
      }
    }
    return out.toString();
  }

  private static void writeManifest(
      Path directory,
      String modelId,
      int dimension,
      int count,
      TaskExemplars exemplars,
      QuantizerKind quantizer,
      String documentPrefix,
      String queryPrefix) {
    String text =
        "# Written by TaskIndexBuilder. Describes the index in this directory.\n"
            + "embeddingModelId="
            + modelId
            + "\ndimension="
            + dimension
            + "\nprompts="
            + count
            + "\ncorpusSha256="
            + corpusDigest(exemplars)
            + "\nquantizer="
            + quantizer.name()
            + "\ntasks="
            + String.join(",", exemplars.taskNames())
            + "\n"
            // Omitted entirely when absent, so an index built without prefixes reads back as one.
            + (documentPrefix == null
                ? ""
                : "documentPrefix="
                    + escape(documentPrefix)
                    + "\nqueryPrefix="
                    + escape(queryPrefix)
                    + "\n");
    try {
      Files.writeString(directory.resolve(MANIFEST), text, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new UncheckedIOException("cannot write " + MANIFEST, e);
    }
  }

  /**
   * Turns a prompt into a vector.
   *
   * <p>Kept as a one-method interface so this module does not depend on a particular embedding
   * backend: the build wires in whichever ModelJar the index is pinned to.
   */
  @FunctionalInterface
  public interface TaskEmbedder {

    /**
     * Embeds one prompt.
     *
     * @param text the prompt
     * @return its embedding
     */
    float[] embed(String text);
  }
}
