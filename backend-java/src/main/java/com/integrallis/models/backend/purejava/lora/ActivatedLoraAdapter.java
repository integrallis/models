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
package com.integrallis.models.backend.purejava.lora;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.StreamReadFeature;
import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.ActivatedAdapterMetadata.Provenance;
import com.integrallis.models.api.ActivatedAdapterMetadata.TrainingProvenance;
import com.integrallis.models.api.ActivatedAdapterMetadata.TrainingSelection;
import com.integrallis.models.api.ActivatedAdapterMetadata.TrainingSource;
import com.integrallis.models.api.ActivatedAdapterMetadata.UpstreamProvenance;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsDtype;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsTensor;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;

/** A pinned PEFT Activated-LoRA adapter executed in Java over owned F32 execution matrices. */
public final class ActivatedLoraAdapter {

  private static final JsonFactory JSON =
      JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION).build();
  private static final String METADATA_FILE = "models-activated-lora.json";
  private static final String ADAPTER_KIND = "activated-lora-tool-specialist";

  /** Transformer projections supported by the activated adapter. */
  public enum Projection {
    QUERY("q_proj", "self_attn.q_proj"),
    KEY("k_proj", "self_attn.k_proj"),
    VALUE("v_proj", "self_attn.v_proj"),
    ATTENTION_OUTPUT("o_proj", "self_attn.o_proj"),
    FFN_GATE("gate_proj", "mlp.gate_proj"),
    FFN_UP("up_proj", "mlp.up_proj"),
    FFN_DOWN("down_proj", "mlp.down_proj");

    private final String targetModule;
    private final String tensorModule;

    Projection(String targetModule, String tensorModule) {
      this.targetModule = targetModule;
      this.tensorModule = tensorModule;
    }

    /** Name used by PEFT's {@code target_modules} metadata. */
    public String targetModule() {
      return targetModule;
    }

    private static Projection fromTargetModule(String targetModule) {
      for (Projection projection : values()) {
        if (projection.targetModule.equals(targetModule)) return projection;
      }
      throw new IllegalArgumentException(
          "unsupported activated adapter target module: " + targetModule);
    }
  }

  /** Base-model dimensions needed to reject adapters built for a different graph. */
  public record Architecture(
      int layers,
      int embeddingDimension,
      int queryDimension,
      int keyDimension,
      int valueDimension,
      int attentionOutputDimension,
      int hiddenDimension) {

    public Architecture {
      positive("layers", layers);
      positive("embeddingDimension", embeddingDimension);
      positive("queryDimension", queryDimension);
      positive("keyDimension", keyDimension);
      positive("valueDimension", valueDimension);
      positive("attentionOutputDimension", attentionOutputDimension);
      positive("hiddenDimension", hiddenDimension);
    }

    int inputDimension(Projection projection) {
      return switch (projection) {
        case QUERY, KEY, VALUE, FFN_GATE, FFN_UP -> embeddingDimension;
        case ATTENTION_OUTPUT -> attentionOutputDimension;
        case FFN_DOWN -> hiddenDimension;
      };
    }

    int outputDimension(Projection projection) {
      return switch (projection) {
        case QUERY -> queryDimension;
        case KEY -> keyDimension;
        case VALUE -> valueDimension;
        case ATTENTION_OUTPUT, FFN_DOWN -> embeddingDimension;
        case FFN_GATE, FFN_UP -> hiddenDimension;
      };
    }
  }

  private final String baseModel;
  private final String baseRevision;
  private final String baseArtifactSha256;
  private final Map<String, String> tokenizerFileSha256;
  private final String adapterSha256;
  private final int rank;
  private final int alpha;
  private final int[] invocationTokens;
  private final Provenance provenance;
  private final Architecture architecture;
  private final LoraProjection[][] layers;

  private ActivatedLoraAdapter(
      String baseModel,
      String baseRevision,
      String baseArtifactSha256,
      Map<String, String> tokenizerFileSha256,
      String adapterSha256,
      int rank,
      int alpha,
      int[] invocationTokens,
      Provenance provenance,
      Architecture architecture,
      LoraProjection[][] layers) {
    this.baseModel = baseModel;
    this.baseRevision = baseRevision;
    this.baseArtifactSha256 = baseArtifactSha256;
    this.tokenizerFileSha256 = Map.copyOf(tokenizerFileSha256);
    this.adapterSha256 = adapterSha256;
    this.rank = rank;
    this.alpha = alpha;
    this.invocationTokens = invocationTokens;
    this.provenance = provenance;
    this.architecture = architecture;
    this.layers = layers;
  }

  /** Opens and validates the adapter, its hash, exact base-artifact binding, and every tensor. */
  public static ActivatedLoraAdapter open(
      Path directory, Arena arena, Path baseArtifact, Architecture architecture)
      throws IOException {
    Objects.requireNonNull(baseArtifact, "baseArtifact");
    return open(directory, arena, sha256(baseArtifact), architecture);
  }

  /** Opens and validates the adapter against a previously computed base-artifact hash. */
  public static ActivatedLoraAdapter open(
      Path directory, Arena arena, String expectedBaseArtifactSha256, Architecture architecture)
      throws IOException {
    Objects.requireNonNull(directory, "directory");
    Objects.requireNonNull(arena, "arena");
    Objects.requireNonNull(architecture, "architecture");
    String expectedBaseHash = requireSha256(expectedBaseArtifactSha256, "expected base artifact");
    Path root = directory.toAbsolutePath().normalize();
    if (!Files.isDirectory(root))
      throw new IOException("adapter directory does not exist: " + root);
    Metadata metadata = readMetadata(root.resolve(METADATA_FILE));
    if (!metadata.baseArtifactSha256.equals(expectedBaseHash)) {
      throw new IllegalArgumentException(
          "base artifact SHA-256 differs from activated adapter metadata: expected "
              + metadata.baseArtifactSha256
              + ", got "
              + expectedBaseHash);
    }
    Path weights = resolveRootFile(root, metadata.adapterFile);
    String actualHash = sha256(weights);
    if (!actualHash.equals(metadata.adapterSha256)) {
      throw new IOException(
          "adapter SHA-256 differs from metadata: expected "
              + metadata.adapterSha256
              + ", got "
              + actualHash);
    }

    SafetensorsBundle tensors = SafetensorsBundle.open(weights, arena);
    Set<String> expectedNames = new HashSet<>();
    LoraProjection[][] projections =
        new LoraProjection[architecture.layers()][Projection.values().length];
    float scale = (float) metadata.alpha / metadata.rank;
    Set<Projection> declaredProjections = metadata.declaredProjections();
    for (int layer = 0; layer < architecture.layers(); layer++) {
      for (Projection projection : declaredProjections) {
        String prefix =
            "base_model.model.model.layers." + layer + "." + projection.tensorModule + ".lora_";
        String aName = prefix + "A.weight";
        String bName = prefix + "B.weight";
        expectedNames.add(aName);
        expectedNames.add(bName);
        int inputDimension = architecture.inputDimension(projection);
        int outputDimension = architecture.outputDimension(projection);
        SafetensorsTensor a = requireTensor(tensors, aName, metadata.rank, inputDimension);
        SafetensorsTensor b = requireTensor(tensors, bName, outputDimension, metadata.rank);
        projections[layer][projection.ordinal()] =
            new LoraProjection(
                a.data(), b.data(), inputDimension, outputDimension, metadata.rank, scale);
      }
    }
    Set<String> actualNames = Set.copyOf(tensors.tensorNames());
    if (!actualNames.equals(expectedNames)) {
      Set<String> missing = new HashSet<>(expectedNames);
      missing.removeAll(actualNames);
      Set<String> unexpected = new HashSet<>(actualNames);
      unexpected.removeAll(expectedNames);
      throw new IllegalArgumentException(
          "activated adapter tensor set differs from the supported graph; missing="
              + missing
              + ", unexpected="
              + unexpected);
    }
    return new ActivatedLoraAdapter(
        metadata.baseModel,
        metadata.baseRevision,
        metadata.baseArtifactSha256,
        metadata.tokenizerFileSha256,
        metadata.adapterSha256,
        metadata.rank,
        metadata.alpha,
        metadata.invocationTokens.clone(),
        metadata.provenance,
        architecture,
        projections);
  }

  /** Adds one activated low-rank projection update to an already-computed base projection. */
  public void addTo(
      int layer,
      Projection projection,
      float[] output,
      int outputOffset,
      float[] input,
      int inputOffset) {
    if (layer < 0 || layer >= layers.length) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
    Objects.requireNonNull(projection, "projection");
    LoraProjection loraProjection = layers[layer][projection.ordinal()];
    if (loraProjection != null) {
      loraProjection.addTo(output, outputOffset, input, inputOffset);
    }
  }

  public String baseModel() {
    return baseModel;
  }

  public String baseRevision() {
    return baseRevision;
  }

  public String baseArtifactSha256() {
    return baseArtifactSha256;
  }

  public String adapterSha256() {
    return adapterSha256;
  }

  public Map<String, String> tokenizerFileSha256() {
    return tokenizerFileSha256;
  }

  public Provenance provenance() {
    return provenance;
  }

  public int rank() {
    return rank;
  }

  public int alpha() {
    return alpha;
  }

  public int[] invocationTokens() {
    return invocationTokens.clone();
  }

  public int inputDimension(Projection projection) {
    return architecture.inputDimension(Objects.requireNonNull(projection, "projection"));
  }

  public int outputDimension(Projection projection) {
    return architecture.outputDimension(Objects.requireNonNull(projection, "projection"));
  }

  private static SafetensorsTensor requireTensor(
      SafetensorsBundle tensors, String name, int rows, int columns) {
    SafetensorsTensor tensor = tensors.tensor(name);
    if (tensor.info().dtype() != SafetensorsDtype.F32) {
      throw new IllegalArgumentException(
          name + " dtype must be F32: " + tensor.info().dtype().code());
    }
    long[] expectedShape = {rows, columns};
    if (!Arrays.equals(tensor.info().shape(), expectedShape)) {
      throw new IllegalArgumentException(
          name
              + " shape must be "
              + Arrays.toString(expectedShape)
              + ": "
              + Arrays.toString(tensor.info().shape()));
    }
    return tensor;
  }

  private static Metadata readMetadata(Path path) throws IOException {
    if (!Files.isRegularFile(path))
      throw new IOException("adapter metadata does not exist: " + path);
    Metadata fields = new Metadata();
    try (JsonParser parser = JSON.createParser(path.toFile())) {
      require(parser.nextToken(), JsonToken.START_OBJECT, "metadata root");
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "metadata field");
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        switch (name) {
          case "schemaVersion" -> fields.schemaVersion = readInt(parser, value, name);
          case "kind" -> fields.kind = readString(parser, value, name);
          case "base" -> readBase(parser, value, fields);
          case "tokenizer" -> readTokenizer(parser, value, fields);
          case "adapter" -> readAdapter(parser, value, fields);
          case "invocation" -> readInvocation(parser, value, fields);
          case "training" -> readTraining(parser, value, fields);
          case "upstream" -> readUpstream(parser, value, fields);
          default -> throw new IOException("unsupported adapter metadata field: " + name);
        }
      }
      if (parser.nextToken() != null)
        throw new IOException("content follows adapter metadata root");
    }
    fields.validate();
    return fields;
  }

  private static void readTokenizer(JsonParser parser, JsonToken token, Metadata fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "tokenizer");
    boolean foundFiles = false;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      if (!"files".equals(name)) {
        throw new IOException("unsupported tokenizer metadata field: " + name);
      }
      foundFiles = true;
      require(value, JsonToken.START_ARRAY, "tokenizer.files");
      TreeMap<String, String> hashes = new TreeMap<>();
      while (parser.nextToken() != JsonToken.END_ARRAY) {
        require(parser.currentToken(), JsonToken.START_OBJECT, "tokenizer file");
        String filename = null;
        String sha256 = null;
        while (parser.nextToken() != JsonToken.END_OBJECT) {
          String field = parser.currentName();
          JsonToken fieldValue = parser.nextToken();
          switch (field) {
            case "name" -> filename = readString(parser, fieldValue, "tokenizer file name");
            case "sha256" -> sha256 = readString(parser, fieldValue, "tokenizer file sha256");
            default -> throw new IOException("unsupported tokenizer file field: " + field);
          }
        }
        if (filename == null || sha256 == null) {
          throw new IOException("tokenizer file requires name and sha256");
        }
        if (hashes.put(filename, sha256) != null) {
          throw new IOException("duplicate tokenizer file: " + filename);
        }
      }
      fields.tokenizerFileSha256 = Map.copyOf(hashes);
    }
    if (!foundFiles) {
      throw new IOException("tokenizer.files is required");
    }
  }

  private static void readBase(JsonParser parser, JsonToken token, Metadata fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "base");
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "model" -> fields.baseModel = readString(parser, value, "base.model");
        case "revision" -> fields.baseRevision = readString(parser, value, "base.revision");
        case "artifactSha256" ->
            fields.baseArtifactSha256 = readString(parser, value, "base.artifactSha256");
        default -> throw new IOException("unsupported base metadata field: " + name);
      }
    }
  }

  private static void readAdapter(JsonParser parser, JsonToken token, Metadata fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "adapter");
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "file" -> fields.adapterFile = readString(parser, value, "adapter.file");
        case "sha256" -> fields.adapterSha256 = readString(parser, value, "adapter.sha256");
        case "rank" -> fields.rank = readInt(parser, value, "adapter.rank");
        case "alpha" -> fields.alpha = readInt(parser, value, "adapter.alpha");
        case "targetModules" -> fields.targetModules = readStrings(parser, value, name);
        default -> throw new IOException("unsupported adapter metadata field: " + name);
      }
    }
  }

  private static void readInvocation(JsonParser parser, JsonToken token, Metadata fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "invocation");
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      if (!"tokens".equals(name)) {
        throw new IOException("unsupported invocation metadata field: " + name);
      }
      fields.invocationTokens = readInts(parser, value, "invocation.tokens");
    }
  }

  private static void readTraining(JsonParser parser, JsonToken token, Metadata fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "training");
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "sources" -> fields.trainingSources = readTrainingSources(parser, value);
        case "dataset" -> fields.dataset = readString(parser, value, "training.dataset");
        case "datasetRevision" ->
            fields.datasetRevision = readString(parser, value, "training.datasetRevision");
        case "sourceFile" -> fields.sourceFile = readString(parser, value, "training.sourceFile");
        case "sourceSha256" ->
            fields.sourceSha256 = readString(parser, value, "training.sourceSha256");
        case "preparedSchemaVersion" ->
            fields.preparedSchemaVersion = readInt(parser, value, "training.preparedSchemaVersion");
        case "preparedManifestSha256" ->
            fields.preparedManifestSha256 =
                readString(parser, value, "training.preparedManifestSha256");
        case "trainSha256" ->
            fields.trainSha256 = readString(parser, value, "training.trainSha256");
        case "validationSha256" ->
            fields.validationSha256 = readString(parser, value, "training.validationSha256");
        case "trainSelection" ->
            fields.trainSelection = readTrainingSelection(parser, value, "training.trainSelection");
        case "validationSelection" ->
            fields.validationSelection =
                readTrainingSelection(parser, value, "training.validationSelection");
        case "formatter" -> fields.formatter = readString(parser, value, "training.formatter");
        case "formatterSha256" ->
            fields.formatterSha256 = readString(parser, value, "training.formatterSha256");
        default -> throw new IOException("unsupported training metadata field: " + name);
      }
    }
  }

  private static void readUpstream(JsonParser parser, JsonToken token, Metadata fields)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "upstream");
    String publisher = null;
    String repository = null;
    String revision = null;
    String modelCardSha256 = null;
    String adapterConfigSha256 = null;
    String license = null;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "publisher" -> publisher = readString(parser, value, "upstream.publisher");
        case "repository" -> repository = readString(parser, value, "upstream.repository");
        case "revision" -> revision = readString(parser, value, "upstream.revision");
        case "modelCardSha256" ->
            modelCardSha256 = readString(parser, value, "upstream.modelCardSha256");
        case "adapterConfigSha256" ->
            adapterConfigSha256 = readString(parser, value, "upstream.adapterConfigSha256");
        case "license" -> license = readString(parser, value, "upstream.license");
        default -> throw new IOException("unsupported upstream metadata field: " + name);
      }
    }
    if (publisher == null
        || repository == null
        || revision == null
        || modelCardSha256 == null
        || adapterConfigSha256 == null
        || license == null) {
      throw new IOException(
          "upstream provenance requires publisher, repository, revision, modelCardSha256, adapterConfigSha256, and license");
    }
    fields.upstreamProvenance =
        new UpstreamProvenance(
            publisher, repository, revision, modelCardSha256, adapterConfigSha256, license);
  }

  private static TrainingSelection readTrainingSelection(
      JsonParser parser, JsonToken token, String context) throws IOException {
    require(token, JsonToken.START_OBJECT, context);
    Integer usable = null;
    Integer skipped = null;
    Integer noCall = null;
    Integer multipleCall = null;
    String sourceLinesSha256 = null;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      String name = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (name) {
        case "usable" -> usable = readInt(parser, value, context + ".usable");
        case "skippedOverMaxLength" ->
            skipped = readInt(parser, value, context + ".skippedOverMaxLength");
        case "noCall" -> noCall = readInt(parser, value, context + ".noCall");
        case "multipleCall" -> multipleCall = readInt(parser, value, context + ".multipleCall");
        case "sourceLinesSha256" ->
            sourceLinesSha256 = readString(parser, value, context + ".sourceLinesSha256");
        default -> throw new IOException("unsupported " + context + " field: " + name);
      }
    }
    if (usable == null
        || skipped == null
        || noCall == null
        || multipleCall == null
        || sourceLinesSha256 == null) {
      throw new IOException(context + " must contain every selection field");
    }
    try {
      return new TrainingSelection(usable, skipped, noCall, multipleCall, sourceLinesSha256);
    } catch (RuntimeException invalid) {
      throw new IOException("invalid " + context + ": " + invalid.getMessage(), invalid);
    }
  }

  private static List<TrainingSource> readTrainingSources(JsonParser parser, JsonToken token)
      throws IOException {
    require(token, JsonToken.START_ARRAY, "training.sources");
    List<TrainingSource> sources = new ArrayList<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      require(parser.currentToken(), JsonToken.START_OBJECT, "training source");
      String role = null;
      String dataset = null;
      String revision = null;
      String sourceFile = null;
      String sourceSha256 = null;
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        switch (name) {
          case "role" -> role = readString(parser, value, "training source role");
          case "dataset" -> dataset = readString(parser, value, "training source dataset");
          case "datasetRevision" ->
              revision = readString(parser, value, "training source revision");
          case "sourceFile" -> sourceFile = readString(parser, value, "training source file");
          case "sourceSha256" ->
              sourceSha256 = readString(parser, value, "training source SHA-256");
          default -> throw new IOException("unsupported training source field: " + name);
        }
      }
      try {
        sources.add(new TrainingSource(role, dataset, revision, sourceFile, sourceSha256));
      } catch (RuntimeException invalid) {
        throw new IOException("invalid training source: " + invalid.getMessage(), invalid);
      }
    }
    return List.copyOf(sources);
  }

  private static List<String> readStrings(JsonParser parser, JsonToken token, String name)
      throws IOException {
    require(token, JsonToken.START_ARRAY, name);
    List<String> values = new ArrayList<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      values.add(readString(parser, parser.currentToken(), name));
    }
    return List.copyOf(values);
  }

  private static int[] readInts(JsonParser parser, JsonToken token, String name)
      throws IOException {
    require(token, JsonToken.START_ARRAY, name);
    List<Integer> values = new ArrayList<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      values.add(readInt(parser, parser.currentToken(), name));
    }
    return values.stream().mapToInt(Integer::intValue).toArray();
  }

  private static String readString(JsonParser parser, JsonToken token, String name)
      throws IOException {
    require(token, JsonToken.VALUE_STRING, name);
    return parser.getText();
  }

  private static int readInt(JsonParser parser, JsonToken token, String name) throws IOException {
    require(token, JsonToken.VALUE_NUMBER_INT, name);
    return parser.getIntValue();
  }

  private static void require(JsonToken actual, JsonToken expected, String name)
      throws IOException {
    if (actual != expected)
      throw new IOException(name + " must be " + expected + "; got " + actual);
  }

  private static Path resolveRootFile(Path root, String filename) throws IOException {
    if (!"adapter_model.safetensors".equals(filename)) {
      throw new IOException("adapter file must be adapter_model.safetensors: " + filename);
    }
    Path resolved = root.resolve(filename).normalize();
    if (!root.equals(resolved.getParent()) || !Files.isRegularFile(resolved)) {
      throw new IOException("adapter file does not exist in adapter root: " + filename);
    }
    return resolved;
  }

  private static String sha256(Path path) throws IOException {
    MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException impossible) {
      throw new AssertionError("SHA-256 is required by the Java platform", impossible);
    }
    try (var input = Files.newInputStream(path)) {
      byte[] buffer = new byte[64 * 1024];
      for (int read; (read = input.read(buffer)) >= 0; ) digest.update(buffer, 0, read);
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static String requireSha256(String value, String name) {
    Objects.requireNonNull(value, name);
    String normalized = value.toLowerCase(java.util.Locale.ROOT);
    if (!normalized.matches("[0-9a-f]{64}")) {
      throw new IllegalArgumentException(name + " SHA-256 must be 64 lowercase hex characters");
    }
    return normalized;
  }

  private static void positive(String name, int value) {
    if (value <= 0) throw new IllegalArgumentException(name + " must be > 0: " + value);
  }

  private static final class Metadata {
    int schemaVersion;
    String kind;
    String baseModel;
    String baseRevision;
    String baseArtifactSha256;
    Map<String, String> tokenizerFileSha256 = Map.of();
    String adapterFile;
    String adapterSha256;
    int rank;
    int alpha;
    List<String> targetModules = List.of();
    int[] invocationTokens = new int[0];
    String dataset;
    String datasetRevision;
    String sourceFile;
    String sourceSha256;
    int preparedSchemaVersion;
    String preparedManifestSha256;
    String trainSha256;
    String validationSha256;
    TrainingSelection trainSelection;
    TrainingSelection validationSelection;
    String formatter;
    String formatterSha256;
    List<TrainingSource> trainingSources = List.of();
    TrainingProvenance trainingProvenance;
    UpstreamProvenance upstreamProvenance;
    Provenance provenance;

    void validate() throws IOException {
      if (schemaVersion != 2 && schemaVersion != 3 && schemaVersion != 4 && schemaVersion != 5)
        throw new IOException("unsupported adapter schemaVersion: " + schemaVersion);
      if (!ADAPTER_KIND.equals(kind)) throw new IOException("unsupported adapter kind: " + kind);
      if (baseModel == null || baseModel.isBlank()) throw new IOException("base.model is required");
      if (baseRevision == null || !baseRevision.matches("[0-9a-f]{40}")) {
        throw new IOException("base.revision must be a pinned 40-character commit hash");
      }
      try {
        baseArtifactSha256 = requireSha256(baseArtifactSha256, "base artifact");
        adapterSha256 = requireSha256(adapterSha256, "adapter");
      } catch (RuntimeException invalid) {
        throw new IOException(invalid.getMessage(), invalid);
      }
      if (rank <= 0 || alpha <= 0) throw new IOException("adapter rank and alpha must be > 0");
      try {
        declaredProjections();
      } catch (IllegalArgumentException unsupported) {
        throw new IOException(unsupported.getMessage(), unsupported);
      }
      if (invocationTokens.length == 0)
        throw new IOException("invocation.tokens must not be empty");
      for (int token : invocationTokens) {
        if (token < 0) throw new IOException("invocation token must be >= 0: " + token);
      }
      try {
        if (schemaVersion == 5) {
          if (upstreamProvenance == null || trainingSources.size() != 0 || dataset != null) {
            throw new IllegalArgumentException(
                "schema 5 requires upstream provenance and must not contain training metadata");
          }
          provenance = upstreamProvenance;
        } else if (upstreamProvenance != null) {
          throw new IllegalArgumentException("upstream provenance requires schemaVersion 5");
        } else if (schemaVersion >= 3) {
          if (trainingSources.isEmpty()) {
            throw new IllegalArgumentException("training.sources must not be empty");
          }
          if (preparedSchemaVersion <= 0) {
            throw new IllegalArgumentException("training.preparedSchemaVersion must be > 0");
          }
          if (dataset != null
              || datasetRevision != null
              || sourceFile != null
              || sourceSha256 != null) {
            throw new IllegalArgumentException(
                "schema 3 training metadata must not mix sources with legacy source fields");
          }
          if (schemaVersion == 4) {
            if (trainSelection == null || validationSelection == null) {
              throw new IllegalArgumentException(
                  "schema 4 requires training.trainSelection and training.validationSelection");
            }
            trainingProvenance =
                new TrainingProvenance(
                    trainingSources,
                    preparedSchemaVersion,
                    preparedManifestSha256,
                    trainSha256,
                    validationSha256,
                    Optional.of(trainSelection),
                    Optional.of(validationSelection),
                    formatter,
                    formatterSha256);
          } else {
            if (trainSelection != null || validationSelection != null) {
              throw new IllegalArgumentException(
                  "schema 3 training metadata does not support selection fields");
            }
            trainingProvenance =
                new TrainingProvenance(
                    trainingSources,
                    preparedSchemaVersion,
                    preparedManifestSha256,
                    trainSha256,
                    validationSha256,
                    formatter,
                    formatterSha256);
          }
        } else {
          if (!trainingSources.isEmpty()) {
            throw new IllegalArgumentException(
                "schema 2 training metadata does not support training.sources");
          }
          if (trainSelection != null || validationSelection != null) {
            throw new IllegalArgumentException(
                "schema 2 training metadata does not support selection fields");
          }
          trainingProvenance =
              new TrainingProvenance(
                  dataset,
                  datasetRevision,
                  sourceFile,
                  sourceSha256,
                  preparedManifestSha256,
                  trainSha256,
                  validationSha256,
                  formatter,
                  formatterSha256);
        }
        if (provenance == null) provenance = trainingProvenance;
        new ActivatedAdapterMetadata(
            baseModel,
            baseRevision,
            baseArtifactSha256,
            tokenizerFileSha256,
            adapterSha256,
            rank,
            alpha,
            Arrays.stream(invocationTokens).boxed().toList(),
            provenance);
      } catch (RuntimeException invalid) {
        throw new IOException(
            "invalid activated adapter provenance: " + invalid.getMessage(), invalid);
      }
    }

    Set<Projection> declaredProjections() {
      if (targetModules.isEmpty()) {
        throw new IllegalArgumentException("adapter targetModules must not be empty");
      }
      Set<Projection> projections = new HashSet<>();
      for (String targetModule : targetModules) {
        if (!projections.add(Projection.fromTargetModule(targetModule))) {
          throw new IllegalArgumentException(
              "adapter targetModules contains duplicate or aliased module: " + targetModule);
        }
      }
      return Set.copyOf(projections);
    }
  }
}
