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
package com.integrallis.models.backend.purejava;

import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.AuxiliaryInferenceBackend;
import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.GroupedDecisionBackend;
import com.integrallis.models.api.HiddenStateInferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.api.ResumableInferenceBackend;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.bert.BertConfig;
import com.integrallis.models.backend.purejava.bert.BertForwardPass;
import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.cact.CactFile;
import com.integrallis.models.backend.purejava.cact.CactHeader;
import com.integrallis.models.backend.purejava.cact.CactNeedle2Layout;
import com.integrallis.models.backend.purejava.cact.CactParser;
import com.integrallis.models.backend.purejava.cact.CactTokenizer;
import com.integrallis.models.backend.purejava.cact.Needle2Weights;
import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliffs;
import com.integrallis.models.backend.purejava.gemma4.Gemma4Config;
import com.integrallis.models.backend.purejava.gemma4.Gemma4Decoder;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gptoss.GptOssForwardPass;
import com.integrallis.models.backend.purejava.gptoss.GptOssHuggingFaceConfig;
import com.integrallis.models.backend.purejava.huggingface.HuggingFaceEndOfGeneration;
import com.integrallis.models.backend.purejava.huggingface.Qwen2HuggingFaceConfig;
import com.integrallis.models.backend.purejava.internal.ModelMemoryArena;
import com.integrallis.models.backend.purejava.llama.DenseProjectionHead;
import com.integrallis.models.backend.purejava.llama.EncoderForwardPass;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaForwardPass;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.lora.ActivatedLoraAdapter;
import com.integrallis.models.backend.purejava.lora.ActivatedLoraAdapter.Architecture;
import com.integrallis.models.backend.purejava.mobilemoe.MobileMoeForwardPass;
import com.integrallis.models.backend.purejava.mobilemoe.MobileMoeHuggingFaceConfig;
import com.integrallis.models.backend.purejava.plan.ExecutionPlanner;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import com.integrallis.models.backend.purejava.plan.PureJavaExecutionPlan;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.backend.purejava.qwen35.Qwen35Config;
import com.integrallis.models.backend.purejava.qwen35.Qwen35ForwardPass;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.spi.PureJavaBackendProvider;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import com.integrallis.models.backend.purejava.tokenizer.HuggingFaceTokenizer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * Pure Java inference backend that loads a GGUF model and runs Llama-family forward passes without
 * any native dependencies.
 */
public final class PureJavaBackend
    implements GroupedDecisionBackend,
        ResumableInferenceBackend,
        SpeculativeInferenceBackend,
        AuxiliaryInferenceBackend,
        HiddenStateInferenceBackend,
        SharedPrefixInferenceBackend {

  public static final String MAX_CONTEXT_LENGTH_PROPERTY = "models.purejava.maxContextLength";
  private static final System.Logger LOGGER = System.getLogger(PureJavaBackend.class.getName());

  private final ModelMemoryArena arenaOwner;
  private final Tokenizer tokenizer;
  private final PureJavaDecoder decoder;
  private final ModelMetadata modelMetadata;
  private final int contextCapacity;
  private final PureJavaExecutionPlan executionPlan;
  private final BackendDiagnostics diagnostics;
  private final GgufBatchedMatrixKernel batchedMatrixKernel;
  private final BatchedCausalAttentionKernel batchedAttentionKernel;
  private final Set<PureJavaInferenceSession> activeSessions =
      Collections.newSetFromMap(new IdentityHashMap<>());
  private PureJavaDecoder.Session[] sessionBatch = new PureJavaDecoder.Session[0];
  private boolean closed;

  private record LoadedDecoder(
      PureJavaDecoder decoder,
      ModelMetadata metadata,
      int contextCapacity,
      PureJavaExecutionPlan executionPlan) {}

  private static final class PureJavaInferenceSession implements InferenceSession {
    private final PureJavaBackend owner;
    private PureJavaDecoder.Session delegate;
    private boolean closed;

    private PureJavaInferenceSession(PureJavaBackend owner, PureJavaDecoder.Session delegate) {
      this.owner = owner;
      this.delegate = delegate;
    }

    @Override
    public int checkpoint() {
      owner.requireOpen(this);
      return delegate.checkpoint();
    }

    @Override
    public boolean isClosed() {
      return closed || owner.closed;
    }

    @Override
    public java.util.OptionalLong allocatedStateBytes() {
      owner.requireOpen(this);
      return delegate.allocatedStateBytes();
    }

    @Override
    public void close() {
      owner.closeSession(this);
    }
  }

  private static final class PureJavaSharedPrefix implements SharedInferencePrefix {
    private final PureJavaBackend owner;
    private final PureJavaDecoder.SharedPrefix delegate;

    private PureJavaSharedPrefix(PureJavaBackend owner, PureJavaDecoder.SharedPrefix delegate) {
      this.owner = owner;
      this.delegate = delegate;
    }

    @Override
    public int checkpoint() {
      owner.checkOpen();
      return delegate.checkpoint();
    }

    @Override
    public long sharedBytes() {
      owner.checkOpen();
      return delegate.sharedBytes();
    }
  }

  private PureJavaBackend(
      ModelMemoryArena arenaOwner,
      Tokenizer tokenizer,
      PureJavaDecoder decoder,
      ModelMetadata modelMetadata,
      int contextCapacity,
      PureJavaExecutionPlan executionPlan,
      BackendDiagnostics diagnostics,
      GgufBatchedMatrixKernel batchedMatrixKernel,
      BatchedCausalAttentionKernel batchedAttentionKernel) {
    this.arenaOwner = arenaOwner;
    this.tokenizer = tokenizer;
    this.decoder = decoder;
    this.modelMetadata = modelMetadata;
    this.contextCapacity = contextCapacity;
    this.executionPlan = executionPlan;
    this.diagnostics = diagnostics;
    this.batchedMatrixKernel = batchedMatrixKernel;
    this.batchedAttentionKernel = batchedAttentionKernel;
  }

  /** Loads a GGUF model file and returns a ready-to-use backend. */
  public static PureJavaBackend load(Path modelPath) {
    return load(
        modelPath,
        ModelMemoryArena.create(),
        BackendConfiguration.empty(),
        GgufBatchedMatrixKernel.none(),
        BatchedCausalAttentionKernel.none());
  }

  /** Loads a GGUF model with registry-neutral, artifact-qualified recommendations. */
  public static PureJavaBackend load(Path modelPath, BackendConfiguration backendConfiguration) {
    return load(
        modelPath,
        ModelMemoryArena.create(),
        Objects.requireNonNull(backendConfiguration, "backendConfiguration"),
        GgufBatchedMatrixKernel.none(),
        BatchedCausalAttentionKernel.none());
  }

  /**
   * Loads a causal GGUF model and a pinned Activated-LoRA adapter executed entirely in Java.
   *
   * <p>The adapter metadata must bind to the SHA-256 of {@code modelPath}, declare every supported
   * projection, and match the loaded model dimensions. The returned backend can fork base and
   * activated branches from one physically shared KV-cache prefix.
   */
  public static PureJavaBackend loadActivatedAdapter(Path modelPath, Path adapterDirectory) {
    return loadActivatedAdapter(modelPath, adapterDirectory, BackendConfiguration.empty());
  }

  /** Loads a pinned Activated-LoRA adapter with registry-neutral backend recommendations. */
  public static PureJavaBackend loadActivatedAdapter(
      Path modelPath, Path adapterDirectory, BackendConfiguration backendConfiguration) {
    return loadActivatedAdapter(
        modelPath, adapterDirectory, backendConfiguration, GgufBatchedMatrixKernel.none());
  }

  /**
   * Loads a pinned Activated-LoRA adapter with an injected batched projection kernel.
   *
   * <p>The kernel only replaces the base matrix products. The transformer plan, the adapter delta,
   * the activation boundary, and the physically shared KV prefix remain the Java implementation, so
   * a kernel arm must be token-identical to the pure-Java arm to count as the same model. The
   * returned backend owns and closes {@code batchedMatrixKernel}.
   */
  public static PureJavaBackend loadActivatedAdapter(
      Path modelPath,
      Path adapterDirectory,
      BackendConfiguration backendConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    return load(
        modelPath,
        ModelMemoryArena.create(),
        Objects.requireNonNull(backendConfiguration, "backendConfiguration"),
        Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel"),
        BatchedCausalAttentionKernel.none(),
        Objects.requireNonNull(adapterDirectory, "adapterDirectory"));
  }

  /**
   * Loads an optional in-process accelerator when one qualifies this artifact and machine, then
   * falls back to the ordinary Java Vector API backend.
   */
  public static PureJavaBackend loadAutomatic(Path modelPath) {
    return loadAutomatic(modelPath, BackendConfiguration.empty());
  }

  /**
   * Loads an optional in-process accelerator with artifact-qualified recommendations, then falls
   * back to the ordinary Java Vector API backend.
   */
  public static PureJavaBackend loadAutomatic(
      Path modelPath, BackendConfiguration backendConfiguration) {
    Objects.requireNonNull(modelPath, "modelPath");
    Objects.requireNonNull(backendConfiguration, "backendConfiguration");
    try {
      for (PureJavaBackendProvider provider :
          ServiceLoader.load(
              PureJavaBackendProvider.class, PureJavaBackend.class.getClassLoader())) {
        try {
          Optional<PureJavaBackend> accelerated = provider.tryLoad(modelPath, backendConfiguration);
          if (accelerated.isPresent()) {
            return accelerated.orElseThrow();
          }
        } catch (LinkageError | RuntimeException failure) {
          LOGGER.log(
              System.Logger.Level.WARNING,
              "Models accelerator provider {0} failed; using the next available path: {1}",
              provider.getClass().getName(),
              failure.getClass().getSimpleName());
        }
      }
    } catch (ServiceConfigurationError failure) {
      LOGGER.log(
          System.Logger.Level.WARNING,
          "Models accelerator discovery failed; using the Vector API: {0}",
          failure.getClass().getSimpleName());
    }
    return load(modelPath, backendConfiguration);
  }

  /**
   * Loads a GGUF model with an injected batched projection implementation.
   *
   * <p>The returned backend owns and closes {@code batchedMatrixKernel}.
   */
  public static PureJavaBackend load(Path modelPath, GgufBatchedMatrixKernel batchedMatrixKernel) {
    return load(
        modelPath,
        ModelMemoryArena.create(),
        BackendConfiguration.empty(),
        Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel"),
        BatchedCausalAttentionKernel.none());
  }

  /**
   * Loads a GGUF model with injected batched projection and causal-attention implementations.
   *
   * <p>The returned backend owns and closes both kernels.
   */
  public static PureJavaBackend load(
      Path modelPath,
      GgufBatchedMatrixKernel batchedMatrixKernel,
      BatchedCausalAttentionKernel batchedAttentionKernel) {
    return load(
        modelPath,
        ModelMemoryArena.create(),
        BackendConfiguration.empty(),
        Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel"),
        Objects.requireNonNull(batchedAttentionKernel, "batchedAttentionKernel"));
  }

  /**
   * Loads a GGUF model with an injected projection kernel and registry-neutral recommendations.
   *
   * <p>The returned backend owns and closes {@code batchedMatrixKernel}.
   */
  public static PureJavaBackend load(
      Path modelPath,
      BackendConfiguration backendConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    return load(
        modelPath,
        ModelMemoryArena.create(),
        Objects.requireNonNull(backendConfiguration, "backendConfiguration"),
        Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel"),
        BatchedCausalAttentionKernel.none());
  }

  /**
   * Loads a GGUF model with injected projection and causal-attention kernels and a qualified
   * backend configuration.
   *
   * <p>The returned backend owns and closes both kernels.
   */
  public static PureJavaBackend load(
      Path modelPath,
      BackendConfiguration backendConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel,
      BatchedCausalAttentionKernel batchedAttentionKernel) {
    return load(
        modelPath,
        ModelMemoryArena.create(),
        Objects.requireNonNull(backendConfiguration, "backendConfiguration"),
        Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel"),
        Objects.requireNonNull(batchedAttentionKernel, "batchedAttentionKernel"));
  }

  static PureJavaBackend load(Path modelPath, Arena arena) {
    return load(
        modelPath,
        ModelMemoryArena.owning(arena),
        BackendConfiguration.empty(),
        GgufBatchedMatrixKernel.none(),
        BatchedCausalAttentionKernel.none());
  }

  private static PureJavaBackend load(
      Path modelPath,
      ModelMemoryArena arenaOwner,
      BackendConfiguration backendConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel,
      BatchedCausalAttentionKernel batchedAttentionKernel) {
    return load(
        modelPath,
        arenaOwner,
        backendConfiguration,
        batchedMatrixKernel,
        batchedAttentionKernel,
        null);
  }

  private static PureJavaBackend load(
      Path modelPath,
      ModelMemoryArena arenaOwner,
      BackendConfiguration backendConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel,
      BatchedCausalAttentionKernel batchedAttentionKernel,
      Path activatedAdapterDirectory) {
    Objects.requireNonNull(arenaOwner, "arenaOwner");
    Objects.requireNonNull(backendConfiguration, "backendConfiguration");
    Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel");
    Objects.requireNonNull(batchedAttentionKernel, "batchedAttentionKernel");
    LoadedDecoder loaded = null;
    try {
      Objects.requireNonNull(modelPath, "modelPath");
      JvmLaunchSupport.requireOptimizingCompiler();
      RuntimeFingerprint runtime = RuntimeFingerprint.capture();
      Map<String, String> recommendations =
          recommendations(backendConfiguration, batchedMatrixKernel);
      PureJavaPlanConfiguration planConfiguration =
          PureJavaPlanConfiguration.fromSystemProperties(recommendations);
      Tokenizer tokenizer;
      Arena arena = arenaOwner.arena();
      if (Files.isDirectory(modelPath)) {
        requireGgufAdapterModel(activatedAdapterDirectory, modelPath);
        Path configPath = modelPath.resolve("config.json");
        if (MobileMoeHuggingFaceConfig.matches(configPath)) {
          MobileMoeHuggingFaceConfig config = MobileMoeHuggingFaceConfig.parse(configPath);
          tokenizer =
              HuggingFaceTokenizer.fromMobileMoe(
                  modelPath.resolve("tokenizer.json"),
                  modelPath.resolve("tokenizer_config.json"),
                  config,
                  Set.copyOf(HuggingFaceEndOfGeneration.tokenIds(modelPath)));
          loaded =
              loadHuggingFaceMobileMoe(
                  modelPath,
                  config,
                  new SafetensorsTensorSource(SafetensorsBundle.open(modelPath, arena)),
                  arena,
                  runtime,
                  planConfiguration,
                  batchedMatrixKernel);
        } else if (GptOssHuggingFaceConfig.matches(configPath)) {
          GptOssHuggingFaceConfig config = GptOssHuggingFaceConfig.parse(configPath);
          tokenizer =
              HuggingFaceTokenizer.fromGptOss(
                  modelPath.resolve("tokenizer.json"),
                  modelPath.resolve("tokenizer_config.json"),
                  config,
                  Set.copyOf(HuggingFaceEndOfGeneration.tokenIds(modelPath)));
          loaded =
              loadHuggingFaceGptOss(
                  modelPath,
                  config,
                  new SafetensorsTensorSource(SafetensorsBundle.open(modelPath, arena)),
                  runtime,
                  planConfiguration,
                  batchedMatrixKernel);
        } else {
          Qwen2HuggingFaceConfig config = Qwen2HuggingFaceConfig.parse(configPath);
          tokenizer =
              HuggingFaceTokenizer.fromQwen2(
                  modelPath.resolve("tokenizer.json"),
                  modelPath.resolve("tokenizer_config.json"),
                  config,
                  Set.copyOf(HuggingFaceEndOfGeneration.tokenIds(modelPath)));
          loaded =
              loadHuggingFaceQwen2(
                  modelPath,
                  config,
                  new SafetensorsTensorSource(SafetensorsBundle.open(modelPath, arena)),
                  runtime,
                  planConfiguration,
                  batchedMatrixKernel,
                  batchedAttentionKernel);
        }
      } else if (CactParser.matches(modelPath)) {
        requireGgufAdapterModel(activatedAdapterDirectory, modelPath);
        CactFile file = CactParser.parse(modelPath, arena);
        CactNeedle2Layout layout = CactNeedle2Layout.from(file);
        CactTokenizer cactTokenizer = CactTokenizer.from(file);
        tokenizer = cactTokenizer;
        loaded =
            loadNeedle2(
                modelPath, layout, cactTokenizer, runtime, planConfiguration, batchedMatrixKernel);
      } else {
        GgufFile file = GgufParser.parse(modelPath, arena);
        tokenizer = GgufTokenizer.fromMetadata(file.metadata());
        String modelFamily = file.metadata().getString("general.architecture").orElse("llama");
        if ("gemma4".equals(modelFamily)) {
          requireLlamaAdapterModel(activatedAdapterDirectory, modelFamily);
          loaded =
              loadGemma4(
                  modelPath,
                  file,
                  modelFamily,
                  runtime,
                  planConfiguration,
                  backendConfiguration,
                  batchedMatrixKernel);
        } else if ("qwen35".equals(modelFamily)) {
          requireLlamaAdapterModel(activatedAdapterDirectory, modelFamily);
          loaded = loadQwen35(modelPath, file, runtime, planConfiguration, batchedMatrixKernel);
        } else if ("bert".equals(modelFamily) || "nomic-bert".equals(modelFamily)) {
          requireLlamaAdapterModel(activatedAdapterDirectory, modelFamily);
          loaded = loadBert(modelPath, file, runtime, planConfiguration, batchedMatrixKernel);
        } else {
          loaded =
              loadLlama(
                  modelPath,
                  file,
                  modelFamily,
                  runtime,
                  planConfiguration,
                  batchedMatrixKernel,
                  batchedAttentionKernel,
                  activatedAdapterDirectory,
                  arena);
        }
      }

      BackendDiagnostics diagnostics =
          endOfGenerationDiagnostics(
              backendConfiguration.enrich(
                  architectureDiagnostics(loaded.executionPlan().diagnostics(), loaded.decoder())),
              tokenizer);

      return new PureJavaBackend(
          arenaOwner,
          tokenizer,
          loaded.decoder(),
          loaded.metadata(),
          loaded.contextCapacity(),
          loaded.executionPlan(),
          diagnostics,
          batchedMatrixKernel,
          batchedAttentionKernel);
    } catch (IOException e) {
      closeAfterFailure(null, arenaOwner, batchedMatrixKernel, batchedAttentionKernel, e);
      throw new UncheckedIOException("Failed to load model: " + modelPath, e);
    } catch (RuntimeException | Error e) {
      closeAfterFailure(
          loaded == null ? null : loaded.decoder(),
          arenaOwner,
          batchedMatrixKernel,
          batchedAttentionKernel,
          e);
      throw e;
    }
  }

  private static LoadedDecoder loadNeedle2(
      Path modelPath,
      CactNeedle2Layout layout,
      CactTokenizer tokenizer,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    CactHeader config = layout.header();
    int queryWidth = Math.multiplyExact(config.queryHeadCount(), config.headWidth());
    int kvWidth = Math.multiplyExact(config.kvHeadCount(), config.headWidth());
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(
            runtime,
            ModelTopology.mappedArchitecture(
                "needle2", queryWidth, kvWidth, kvWidth, config.layerCount()),
            planConfiguration,
            batchedMatrixKernel);
    int contextCapacity = runtimeContextLength(config.maximumSequenceLength(), planConfiguration);
    ModelMetadata metadata =
        new ModelMetadata(
            "needle2",
            modelPath.getFileName().toString(),
            config.maximumSequenceLength(),
            config.vocabularySize(),
            config.modelWidth(),
            config.layerCount(),
            config.queryHeadCount(),
            config.kvHeadCount());
    return new LoadedDecoder(
        new Needle2DecoderAdapter(
            Needle2Weights.load(layout), contextCapacity, tokenizer.tokenId("</tools>")),
        metadata,
        contextCapacity,
        executionPlan);
  }

  private static LoadedDecoder loadBert(
      Path modelPath,
      GgufFile file,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    BertConfig config = BertConfig.fromMetadata(file.metadata());
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(
            runtime,
            ModelTopology.mappedArchitecture(
                "bert",
                config.embeddingDim(),
                config.embeddingDim(),
                config.embeddingDim(),
                config.numLayers()),
            planConfiguration,
            batchedMatrixKernel);
    int contextCapacity = runtimeContextLength(config.contextLength(), planConfiguration);
    ModelMetadata metadata =
        new ModelMetadata(
            config.architecture(),
            modelPath.getFileName().toString(),
            config.contextLength(),
            config.vocabSize(),
            config.embeddingDim(),
            config.numLayers(),
            config.numHeads(),
            config.numHeads());
    return new LoadedDecoder(
        new EncoderDecoderAdapter(BertForwardPass.fromGgufFile(file, config)),
        metadata,
        contextCapacity,
        executionPlan);
  }

  private static LoadedDecoder loadHuggingFaceQwen2(
      Path modelPath,
      Qwen2HuggingFaceConfig huggingFaceConfig,
      SafetensorsTensorSource tensors,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel,
      BatchedCausalAttentionKernel batchedAttentionKernel) {
    LlamaConfig config = huggingFaceConfig.model();
    LlamaWeights weights = LlamaWeights.fromQwen2Safetensors(tensors, huggingFaceConfig);
    String modelFamily = "qwen2";
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(
            runtime,
            ModelTopology.from(modelFamily, config, weights),
            planConfiguration,
            batchedMatrixKernel);
    int contextCapacity = runtimeContextLength(config.contextLength(), planConfiguration);
    KvCache cache =
        new KvCache(config.numLayers(), contextCapacity, config.keyDim(), config.valueDim());
    PureJavaDecoder decoder =
        new LlamaDecoder(
            new LlamaForwardPass(
                config,
                weights,
                cache,
                executionPlan,
                batchedMatrixKernel,
                batchedAttentionKernel));
    ModelMetadata metadata =
        new ModelMetadata(
            modelFamily,
            modelPath.getFileName().toString(),
            config.contextLength(),
            config.vocabSize(),
            config.embeddingDim(),
            config.numLayers(),
            config.numHeads(),
            config.numKvHeads());
    return new LoadedDecoder(decoder, metadata, contextCapacity, executionPlan);
  }

  private static LoadedDecoder loadHuggingFaceGptOss(
      Path modelPath,
      GptOssHuggingFaceConfig config,
      SafetensorsTensorSource tensors,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    String modelFamily = "gpt-oss";
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(
            runtime,
            ModelTopology.mappedArchitecture(
                modelFamily,
                config.queryDimension(),
                config.keyValueDimension(),
                config.keyValueDimension(),
                config.numLayers()),
            planConfiguration,
            batchedMatrixKernel);
    int contextCapacity = runtimeContextLength(config.maxPosition(), planConfiguration);
    PureJavaDecoder decoder =
        new GptOssDecoderAdapter(GptOssForwardPass.load(config, tensors, contextCapacity));
    ModelMetadata metadata =
        new ModelMetadata(
            modelFamily,
            modelPath.getFileName().toString(),
            config.maxPosition(),
            config.vocabSize(),
            config.hiddenSize(),
            config.numLayers(),
            config.numHeads(),
            config.numKvHeads());
    return new LoadedDecoder(decoder, metadata, contextCapacity, executionPlan);
  }

  private static LoadedDecoder loadHuggingFaceMobileMoe(
      Path modelPath,
      MobileMoeHuggingFaceConfig config,
      SafetensorsTensorSource tensors,
      Arena arena,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    String modelFamily = "mobilemoe";
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(
            runtime,
            ModelTopology.mappedArchitecture(
                modelFamily,
                config.queryDimension(),
                config.keyValueDimension(),
                config.keyValueDimension(),
                config.numLayers()),
            planConfiguration,
            batchedMatrixKernel);
    int contextCapacity = runtimeContextLength(config.contextLength(), planConfiguration);
    PureJavaDecoder decoder =
        new MobileMoeDecoderAdapter(
            MobileMoeForwardPass.load(config, tensors, contextCapacity, arena));
    ModelMetadata metadata =
        new ModelMetadata(
            modelFamily,
            modelPath.getFileName().toString(),
            config.contextLength(),
            config.vocabSize(),
            config.hiddenSize(),
            config.numLayers(),
            config.numHeads(),
            config.numKvHeads());
    return new LoadedDecoder(decoder, metadata, contextCapacity, executionPlan);
  }

  private static LoadedDecoder loadLlama(
      Path modelPath,
      GgufFile file,
      String modelFamily,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel,
      BatchedCausalAttentionKernel batchedAttentionKernel,
      Path activatedAdapterDirectory,
      Arena arena)
      throws IOException {
    LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
    LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(
            runtime,
            ModelTopology.from(modelFamily, config, weights),
            planConfiguration,
            batchedMatrixKernel);
    int contextCapacity = runtimeContextLength(config.contextLength(), planConfiguration);
    PureJavaDecoder decoder;
    if (config.usesBidirectionalAttention()) {
      requireLlamaAdapterModel(activatedAdapterDirectory, config.architecture().toString());
      // An encoder holds no KV cache: every position depends on every other, so there is nothing
      // from a previous step to reuse.
      decoder =
          new EncoderDecoderAdapter(
              new EncoderForwardPass(
                  config,
                  weights,
                  DenseProjectionHead.load(file, modelFamily, config.embeddingDim()).orElse(null)));
    } else {
      ActivatedLoraAdapter activatedAdapter =
          activatedAdapterDirectory == null
              ? null
              : ActivatedLoraAdapter.open(
                  activatedAdapterDirectory,
                  arena,
                  modelPath,
                  new Architecture(
                      config.numLayers(),
                      config.embeddingDim(),
                      config.queryDim(),
                      config.keyDim(),
                      config.valueDim(),
                      config.attentionOutputDim(),
                      config.hiddenDim(),
                      config.numHeads(),
                      config.numKvHeads(),
                      !config.usesNeoxRope()));
      KvCache cache =
          new KvCache(config.numLayers(), contextCapacity, config.keyDim(), config.valueDim());
      LlamaForwardPass forwardPass =
          activatedAdapter == null
              ? new LlamaForwardPass(
                  config,
                  weights,
                  cache,
                  executionPlan,
                  batchedMatrixKernel,
                  batchedAttentionKernel)
              : new LlamaForwardPass(
                  config,
                  weights,
                  cache,
                  executionPlan,
                  batchedMatrixKernel,
                  batchedAttentionKernel,
                  activatedAdapter);
      decoder = new LlamaDecoder(forwardPass);
    }
    ModelMetadata metadata =
        new ModelMetadata(
            modelFamily,
            modelName(modelPath, file),
            config.contextLength(),
            config.vocabSize(),
            config.embeddingDim(),
            config.numLayers(),
            config.numHeads(),
            config.numKvHeads());
    return new LoadedDecoder(decoder, metadata, contextCapacity, executionPlan);
  }

  private static void requireGgufAdapterModel(Path adapterDirectory, Path modelPath) {
    if (adapterDirectory != null) {
      throw new IllegalArgumentException(
          "activated adapters require a causal Llama-family GGUF model, not " + modelPath);
    }
  }

  private static void requireLlamaAdapterModel(Path adapterDirectory, String modelFamily) {
    if (adapterDirectory != null) {
      throw new IllegalArgumentException(
          "activated adapters require a causal Llama-family graph; got " + modelFamily);
    }
  }

  private static LoadedDecoder loadGemma4(
      Path modelPath,
      GgufFile file,
      String modelFamily,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      BackendConfiguration backendConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel)
      throws IOException {
    Gemma4Config config = Gemma4Config.fromMetadata(file.metadata());
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(
            runtime, gemma4Topology(file, config), planConfiguration, batchedMatrixKernel);
    ModelMetadata metadata =
        new ModelMetadata(
            modelFamily,
            modelName(modelPath, file),
            config.contextLength(),
            config.vocabSize(),
            config.embeddingDim(),
            config.numLayers(),
            config.numHeads(),
            config.numKvHeads(0));
    int contextCapacity = runtimeContextLength(config.contextLength(), planConfiguration);
    Gemma4Decoder decoder = Gemma4Decoder.load(file, contextCapacity, batchedMatrixKernel);
    return new LoadedDecoder(
        new Gemma4DecoderAdapter(decoder), metadata, contextCapacity, executionPlan);
  }

  private static LoadedDecoder loadQwen35(
      Path modelPath,
      GgufFile file,
      RuntimeFingerprint runtime,
      PureJavaPlanConfiguration planConfiguration,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    Qwen35Config config = Qwen35Config.fromMetadata(file.metadata());
    Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(file, batchedMatrixKernel);
    PureJavaExecutionPlan executionPlan =
        ExecutionPlanner.plan(runtime, graph.topology(), planConfiguration, batchedMatrixKernel);
    int contextCapacity = runtimeContextLength(config.contextLength(), planConfiguration);
    ModelMetadata metadata =
        new ModelMetadata(
            "qwen35",
            modelName(modelPath, file),
            config.contextLength(),
            config.vocabSize(),
            config.embeddingDim(),
            config.numLayers(),
            config.numHeads(),
            config.numKvHeads());
    PureJavaDecoder decoder =
        new Qwen35DecoderAdapter(graph.withExecutionPlan(executionPlan), contextCapacity);
    return new LoadedDecoder(decoder, metadata, contextCapacity, executionPlan);
  }

  @Override
  public String name() {
    return "pure-java";
  }

  @Override
  public ModelMetadata metadata() {
    return modelMetadata;
  }

  @Override
  public int contextCapacity() {
    return contextCapacity;
  }

  /** Returns the immutable execution plan selected while loading this model. */
  public PureJavaExecutionPlan executionPlan() {
    return executionPlan;
  }

  @Override
  public BackendDiagnostics diagnostics() {
    return PerformanceCliffs.enrich(diagnostics);
  }

  @Override
  public Tokenizer tokenizer() {
    return tokenizer;
  }

  @Override
  public int maxBatchSize() {
    checkOpen();
    return decoder.maxBatchSize();
  }

  @Override
  public float[] forward(int token, int position) {
    checkOpen();
    return decoder.forward(token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    checkOpen();
    return decoder.forwardTransient(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    checkOpen();
    return decoder.prefill(tokens, startPosition);
  }

  /**
   * Runs the stack over a sequence and returns the final position's hidden state.
   *
   * <p>Skips the vocabulary projection, which is the widest matmul in the pass — so producing an
   * embedding costs less per token than generating one.
   *
   * <p>The array is backend-owned scratch, valid until the next call. Copy it to keep it.
   */
  @Override
  public float[] prefillHiddenState(int[] tokens, int startPosition) {
    checkOpen();
    return decoder.prefillHiddenState(tokens, startPosition);
  }

  /**
   * Runs one step and returns its hidden state instead of logits.
   *
   * <p>Needed for mean pooling, which reduces over every position rather than only the last.
   *
   * <p>The array is backend-owned scratch, valid until the next call.
   */
  public float[] hiddenState(int token, int position) {
    checkOpen();
    return decoder.hiddenState(token, position);
  }

  /** Whether this model's architecture exposes hidden states for embedding. */
  @Override
  public boolean supportsHiddenState() {
    return decoder.supportsHiddenState();
  }

  /**
   * Encodes a whole sequence into one vector using the model's own pooling and projection.
   *
   * <p>Available only where the model owns its embedding pipeline; check {@link
   * #supportsSequenceEmbedding()} first.
   *
   * @param tokens the tokenized text
   * @return a newly allocated pooled embedding, not normalized
   */
  public float[] embedSequence(int[] tokens) {
    checkOpen();
    return decoder.embedSequence(tokens);
  }

  /** Whether this model encodes whole sequences rather than exposing per-position states. */
  public boolean supportsSequenceEmbedding() {
    return decoder.supportsSequenceEmbedding();
  }

  @Override
  public boolean supportsContrastiveEncoding() {
    return decoder.supportsContrastiveEncoding();
  }

  @Override
  public int contrastiveDimension() {
    checkOpen();
    return decoder.contrastiveDimension();
  }

  @Override
  public float[] encodeContrastive(int[] tokens) {
    checkOpen();
    return decoder.encodeContrastive(tokens);
  }

  @Override
  public boolean supportsConfidenceScoring() {
    return decoder.supportsConfidenceScoring();
  }

  @Override
  public float scoreConfidence(int[] tokens) {
    checkOpen();
    return decoder.scoreConfidence(tokens);
  }

  @Override
  public InferenceSession openSession() {
    checkOpen();
    return registerSession(decoder.openSession());
  }

  @Override
  public boolean supportsSharedPrefixes() {
    checkOpen();
    return decoder.supportsSharedPrefixes();
  }

  @Override
  public boolean supportsActivatedBranch() {
    checkOpen();
    return decoder.supportsActivatedBranch();
  }

  @Override
  public Optional<ActivatedAdapterMetadata> activatedAdapter() {
    checkOpen();
    return decoder.activatedAdapter();
  }

  @Override
  public void activateAdapter(InferenceSession session) {
    PureJavaInferenceSession state = requireOpen(session);
    if (!decoder.supportsActivatedBranch()) {
      throw new IllegalStateException("loaded model has no activated adapter");
    }
    decoder.activateAdapter(state.delegate);
  }

  @Override
  public SharedInferencePrefix freezePrefix(InferenceSession source) {
    PureJavaInferenceSession session = requireOpen(source);
    if (!decoder.supportsSharedPrefixes()) {
      throw new UnsupportedOperationException(
          "model family " + modelMetadata.modelFamily() + " cannot share KV-cache prefixes");
    }
    PureJavaDecoder.SharedPrefix prefix = decoder.freezePrefix(session.delegate);
    session.closed = true;
    activeSessions.remove(session);
    session.delegate = null;
    return new PureJavaSharedPrefix(this, prefix);
  }

  @Override
  public InferenceSession fork(SharedInferencePrefix prefix, Branch branch) {
    checkOpen();
    Objects.requireNonNull(branch, "branch");
    PureJavaSharedPrefix ownedPrefix = requirePrefix(prefix);
    boolean activated = branch == Branch.ACTIVATED_ADAPTER;
    if (activated && !decoder.supportsActivatedBranch()) {
      throw new IllegalStateException("loaded model has no activated adapter");
    }
    return registerSession(decoder.fork(ownedPrefix.delegate, activated));
  }

  @Override
  public boolean sharesPrefixStorage(InferenceSession first, InferenceSession second) {
    return decoder.sharesPrefixStorage(requireOpen(first).delegate, requireOpen(second).delegate);
  }

  @Override
  public float[] forward(InferenceSession session, int token, int position) {
    return decoder.forward(requireOpen(session).delegate, token, position);
  }

  @Override
  public float[] forwardTransient(InferenceSession session, int token, int position) {
    return decoder.forwardTransient(requireOpen(session).delegate, token, position);
  }

  @Override
  public float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
    return decoder.prefill(requireOpen(session).delegate, tokens, startPosition);
  }

  @Override
  public float[] forwardHiddenState(int token, int position) {
    return decoder.hiddenState(token, position).clone();
  }

  @Override
  public float[] forwardHiddenStateTransient(int token, int position) {
    return decoder.hiddenState(token, position);
  }

  @Override
  public float[] forwardHiddenState(InferenceSession session, int token, int position) {
    return decoder.hiddenState(requireOpen(session).delegate, token, position);
  }

  @Override
  public float[] forwardHiddenStateTransient(InferenceSession session, int token, int position) {
    return decoder.hiddenStateTransient(requireOpen(session).delegate, token, position);
  }

  @Override
  public float[] prefillHiddenState(InferenceSession session, int[] tokens, int startPosition) {
    return decoder.prefillHiddenState(requireOpen(session).delegate, tokens, startPosition);
  }

  @Override
  public boolean supportsRaggedPrefillBatch() {
    return decoder.supportsRaggedPrefillBatch();
  }

  @Override
  public LogitBatch prefillBatch(InferenceSession[] sessions, int[][] tokenBatches) {
    if (!decoder.supportsRaggedPrefillBatch()) {
      return SharedPrefixInferenceBackend.super.prefillBatch(sessions, tokenBatches);
    }
    return decoder.prefillBatch(unwrapSessions(sessions), tokenBatches);
  }

  /** Whether several sessions can be prefilled together and their hidden states returned. */
  public boolean supportsBatchedHiddenStates() {
    return decoder.supportsBatchedHiddenStates();
  }

  /**
   * Prefills independent sessions together and returns one final normalized hidden state each.
   *
   * <p>Falls back to prefilling them one at a time where the decoder cannot batch, so a caller gets
   * the same answers either way and only the cost changes.
   */
  public float[][] prefillBatchHiddenStates(
      InferenceSession[] sessions, int[][] tokenBatches, int[] startPositions) {
    Objects.requireNonNull(sessions, "sessions");
    Objects.requireNonNull(tokenBatches, "tokenBatches");
    Objects.requireNonNull(startPositions, "startPositions");
    if (sessions.length != tokenBatches.length || sessions.length != startPositions.length) {
      throw new IllegalArgumentException(
          "sessions "
              + sessions.length
              + ", tokenBatches "
              + tokenBatches.length
              + " and startPositions "
              + startPositions.length
              + " must agree");
    }
    if (!decoder.supportsBatchedHiddenStates()) {
      // The same answers by a slower route, so a caller never has to ask which path it got.
      float[][] states = new float[sessions.length][];
      for (int index = 0; index < sessions.length; index++) {
        states[index] =
            prefillHiddenState(sessions[index], tokenBatches[index], startPositions[index]);
      }
      return states;
    }
    return decoder.prefillBatchHiddenStates(unwrapSessions(sessions), tokenBatches);
  }

  @Override
  public LogitBatch prefillBatchTransient(InferenceSession[] sessions, int[][] tokenBatches) {
    if (!decoder.supportsRaggedPrefillBatch()) {
      return SharedPrefixInferenceBackend.super.prefillBatchTransient(sessions, tokenBatches);
    }
    return decoder.prefillBatchTransient(unwrapSessions(sessions), tokenBatches);
  }

  @Override
  public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
    return decoder.forwardBatch(unwrapSessions(sessions), tokens);
  }

  @Override
  public LogitBatch forwardBatchTransient(InferenceSession[] sessions, int[] tokens) {
    return decoder.forwardBatchTransient(unwrapSessions(sessions), tokens);
  }

  @Override
  public void rewind(InferenceSession session, int checkpoint) {
    decoder.rewind(requireOpen(session).delegate, checkpoint);
  }

  @Override
  public void reset(InferenceSession session) {
    decoder.reset(requireOpen(session).delegate);
  }

  @Override
  public int checkpoint() {
    checkOpen();
    return decoder.checkpoint();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    checkOpen();
    return decoder.verify(tokens, startPosition);
  }

  @Override
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
    checkOpen();
    return decoder.verifyTransient(tokens, startPosition);
  }

  @Override
  public boolean supportsResumption() {
    return decoder.supportsResumption();
  }

  @Override
  public boolean supportsGroupedDecisions() {
    return decoder.supportsGroupedDecisions();
  }

  @Override
  public int maximumGroupSize() {
    return decoder.maximumGroupSize();
  }

  @Override
  public boolean groupedDecisionsMatchSingleDecisions() {
    // Yes, and measured rather than assumed. MEASURED 2026-09-24 on the shipped Qwen3.5-4B at
    // Q4_K_M: four questions over 120 tokens of shared evidence, answered as a group and then each
    // asked on its own, agreed to the bit -- 0.000e+00 on every logit of all four. The reason is
    // that this decoder's batched projection computes each row exactly as its single-row projection
    // would, so the number of rows in a call cannot reach the result. Splitting one prefill into
    // several batches is likewise bit-identical.
    return true;
  }

  @Override
  public int groupedDecisionBreakEven() {
    // Every question here ends with a single-token step through the whole of the weights in Java,
    // which is the most expensive step in a decision and the one grouping replaces with one step
    // for the group. MEASURED 2026-09-24 on a Hetzner CCX33 with the native decode kernel off:
    // 1.21x at two questions, 1.49x at five, 1.69x at twenty. Worth it from two.
    return 2;
  }

  @Override
  public float[][] decideGrouped(int[][] suffixes) {
    return decoder.decideGrouped(suffixes);
  }

  @Override
  public Resumption capture() {
    int position = decoder.checkpoint();
    Object point = decoder.captureResumption();
    return new DecoderResumption(position, point);
  }

  @Override
  public void resume(Resumption point) {
    Objects.requireNonNull(point, "point");
    if (!(point instanceof DecoderResumption resumption)) {
      throw new IllegalArgumentException("resumption was not issued by this backend");
    }
    decoder.resume(resumption.point());
  }

  private record DecoderResumption(int position, Object point) implements Resumption {}

  @Override
  public void rewind(int checkpoint) {
    checkOpen();
    decoder.rewind(checkpoint);
  }

  @Override
  public void reset() {
    checkOpen();
    decoder.reset();
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    for (PureJavaInferenceSession session : Set.copyOf(activeSessions)) {
      session.closed = true;
      session.delegate = null;
    }
    activeSessions.clear();
    Arrays.fill(sessionBatch, null);
    RuntimeException closeFailure = null;
    try {
      decoder.close();
    } catch (IOException failure) {
      closeFailure = new UncheckedIOException("Failed to close decoder resources", failure);
    } catch (RuntimeException failure) {
      closeFailure = failure;
    }
    try {
      batchedMatrixKernel.close();
    } catch (RuntimeException failure) {
      closeFailure = combineCloseFailures(closeFailure, failure);
    }
    try {
      batchedAttentionKernel.close();
    } catch (RuntimeException failure) {
      closeFailure = combineCloseFailures(closeFailure, failure);
    }
    try {
      arenaOwner.close();
    } catch (RuntimeException failure) {
      closeFailure = combineCloseFailures(closeFailure, failure);
    }
    if (closeFailure != null) {
      throw closeFailure;
    }
  }

  private PureJavaDecoder.Session[] unwrapSessions(InferenceSession[] sessions) {
    Objects.requireNonNull(sessions, "sessions");
    if (sessionBatch.length != sessions.length) {
      sessionBatch = new PureJavaDecoder.Session[sessions.length];
    }
    for (int index = 0; index < sessions.length; index++) {
      sessionBatch[index] = requireOpen(sessions[index]).delegate;
    }
    return sessionBatch;
  }

  private PureJavaInferenceSession requireOpen(InferenceSession session) {
    checkOpen();
    Objects.requireNonNull(session, "session");
    if (!(session instanceof PureJavaInferenceSession pureJavaSession)
        || pureJavaSession.owner != this) {
      throw new IllegalArgumentException("session belongs to a different backend");
    }
    return requireOpen(pureJavaSession);
  }

  private PureJavaSharedPrefix requirePrefix(SharedInferencePrefix prefix) {
    Objects.requireNonNull(prefix, "prefix");
    if (!(prefix instanceof PureJavaSharedPrefix pureJavaPrefix) || pureJavaPrefix.owner != this) {
      throw new IllegalArgumentException("prefix belongs to a different backend");
    }
    return pureJavaPrefix;
  }

  private PureJavaInferenceSession requireOpen(PureJavaInferenceSession session) {
    checkOpen();
    if (session.closed || session.delegate == null) {
      throw new IllegalStateException("session is closed");
    }
    return session;
  }

  private PureJavaInferenceSession registerSession(PureJavaDecoder.Session delegate) {
    PureJavaInferenceSession session = new PureJavaInferenceSession(this, delegate);
    activeSessions.add(session);
    return session;
  }

  private void closeSession(PureJavaInferenceSession session) {
    if (session.closed) {
      return;
    }
    PureJavaDecoder.Session delegate = session.delegate;
    session.closed = true;
    activeSessions.remove(session);
    try {
      if (!closed && delegate != null) {
        decoder.reset(delegate);
      }
    } finally {
      session.delegate = null;
      for (int index = 0; index < sessionBatch.length; index++) {
        if (sessionBatch[index] == delegate) {
          sessionBatch[index] = null;
        }
      }
    }
  }

  private void checkOpen() {
    if (closed) {
      throw new IllegalStateException("backend is closed");
    }
  }

  /**
   * The context a session is sized for.
   *
   * <p>A ModelJar can recommend a bound through the plan configuration, which is how a model that
   * knows its prompts are short avoids being sized for a context it will never reach: at Qwen3.5's
   * 262,144 tokens the key and value cache alone is about 17 GB, so a default heap dies before the
   * first answer. A deployment setting still wins over the recommendation.
   */
  private static int runtimeContextLength(
      int modelContextLength, PureJavaPlanConfiguration planConfiguration) {
    int recommended = planConfiguration.maxContextLength();
    return recommended == PureJavaPlanConfiguration.MODEL_MAXIMUM_CONTEXT
        ? modelContextLength
        : Math.min(modelContextLength, recommended);
  }

  private static ModelTopology gemma4Topology(GgufFile file, Gemma4Config config) {
    List<ModelTopology.LayerTopology> layers = new ArrayList<>(config.numLayers());
    int queryRows = 0;
    int keyRows = 0;
    int valueRows = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      String prefix = "blk." + layer + ".";
      GgufTensorType key = tensorType(file, prefix + "attn_k.weight");
      layers.add(
          new ModelTopology.LayerTopology(
              tensorType(file, prefix + "attn_q.weight"),
              key,
              config.usesSlidingWindow(layer) ? tensorType(file, prefix + "attn_v.weight") : key,
              tensorType(file, prefix + "attn_output.weight"),
              tensorType(file, prefix + "ffn_gate.weight"),
              tensorType(file, prefix + "ffn_up.weight"),
              tensorType(file, prefix + "ffn_down.weight")));
      queryRows = Math.max(queryRows, config.queryDim(layer));
      keyRows = Math.max(keyRows, config.keyDim(layer));
      valueRows = Math.max(valueRows, config.valueDim(layer));
    }
    return new ModelTopology("gemma4", queryRows, keyRows, valueRows, layers, true);
  }

  /**
   * Adds the resolved end-of-generation set ({@code end-of-generation-token-ids}) and, for GGUF
   * tokenizers, the rules behind each id ({@code end-of-generation.<id>}) and the chat template's
   * end-of-turn resolution ({@code end-of-generation.chat-template}).
   */
  private static BackendDiagnostics endOfGenerationDiagnostics(
      BackendDiagnostics diagnostics, Tokenizer tokenizer) {
    Map<String, String> environment = new LinkedHashMap<>(diagnostics.environment());
    environment.put(
        "end-of-generation-token-ids",
        java.util.Arrays.stream(tokenizer.endOfGenerationTokenIds())
            .mapToObj(Integer::toString)
            .collect(java.util.stream.Collectors.joining(",")));
    if (tokenizer instanceof GgufTokenizer gguf) {
      gguf.endOfGenerationSources()
          .forEach(
              (token, sources) ->
                  environment.put("end-of-generation." + token, String.join(",", sources)));
      environment.put("end-of-generation.chat-template", gguf.chatTemplateEndOfTurnResolution());
    }
    return new BackendDiagnostics(
        diagnostics.backend(), diagnostics.planVersion(), environment, diagnostics.optimizations());
  }

  private static BackendDiagnostics architectureDiagnostics(
      BackendDiagnostics diagnostics, PureJavaDecoder decoder) {
    if (decoder instanceof MobileMoeDecoderAdapter mobileMoe) {
      Map<String, String> environment = new LinkedHashMap<>(diagnostics.environment());
      environment.put("artifact-format", "safetensors");
      environment.put("weight-encoding", "packed-int4-g32");
      environment.put("runtime-weight-layout", mobileMoe.runtimeWeightLayout());
      environment.put(
          "architecture-prefill-batch-size", Integer.toString(mobileMoe.prefillBatchSize()));
      List<OptimizationDecision> optimizations = new ArrayList<>(diagnostics.optimizations());
      optimizations.removeIf(
          decision ->
              Set.of("grouped-projections", "batched-prefill", "mapped-model-weights")
                  .contains(decision.id()));
      optimizations.add(
          new OptimizationDecision(
              "mapped-model-weights",
              OptimizationStatus.ENABLED,
              "the source Safetensors checkpoint remains mapped while prepared runtime weights use backend-owned native memory",
              Map.of("artifact-format", "safetensors", "storage", "memory-segment")));
      boolean q8Runtime = "q8".equals(mobileMoe.runtimeWeightLayout());
      optimizations.add(
          new OptimizationDecision(
              "mobilemoe-runtime-weight-layout",
              q8Runtime ? OptimizationStatus.ENABLED : OptimizationStatus.DISABLED,
              q8Runtime
                  ? "packed group-32 INT4 checkpoint weights are prepared once as row-major Q8_0 for Java Vector API execution"
                  : "compact mode executes packed group-32 INT4 weights directly from the mapped checkpoint",
              Map.of(
                  "artifact-layout",
                  "packed-int4-g32",
                  "implementation",
                  "java-vector-api",
                  "runtime-layout",
                  mobileMoe.runtimeWeightLayout())));
      optimizations.add(
          new OptimizationDecision(
              "grouped-projections",
              q8Runtime ? OptimizationStatus.ENABLED : OptimizationStatus.UNSUPPORTED,
              q8Runtime
                  ? "MobileMoE QKV and shared gate/up projections reuse one quantized activation and one row dispatch"
                  : "the compact packed-INT4 fallback has no grouped projection kernel",
              Map.of("qkv", "grouped", "gate-up", "grouped")));
      int prefillBatchSize = mobileMoe.prefillBatchSize();
      optimizations.add(
          new OptimizationDecision(
              "batched-prefill",
              prefillBatchSize > 1 ? OptimizationStatus.ENABLED : OptimizationStatus.DISABLED,
              prefillBatchSize > 1
                  ? "MobileMoE attention, shared FFN, router, and routed experts execute prompt batches with retained weights"
                  : "MobileMoE prefill batching is disabled by configuration",
              Map.of("batch-size", Integer.toString(prefillBatchSize))));
      return new BackendDiagnostics(
          diagnostics.backend(), diagnostics.planVersion(), environment, optimizations);
    }
    if (decoder instanceof GptOssDecoderAdapter) {
      Map<String, String> environment = new LinkedHashMap<>(diagnostics.environment());
      environment.put("artifact-format", "safetensors");
      environment.put("weight-encoding", "mxfp4");
      List<OptimizationDecision> optimizations = new ArrayList<>(diagnostics.optimizations());
      optimizations.add(
          new OptimizationDecision(
              "gpt-oss-mxfp4",
              OptimizationStatus.ENABLED,
              "MXFP4 routed experts execute directly from the mapped artifact",
              Map.of("implementation", "java-vector-api")));
      return new BackendDiagnostics(
          diagnostics.backend(), diagnostics.planVersion(), environment, optimizations);
    }
    if (decoder instanceof Needle2DecoderAdapter needle2) {
      CactHeader header = needle2.header();
      Map<String, String> environment = new LinkedHashMap<>(diagnostics.environment());
      environment.put("artifact-format", "cact");
      environment.put("weight-encoding", "rotated-codebook");
      environment.put("kv-bits", Integer.toString(header.kvBits()));
      List<OptimizationDecision> optimizations = new ArrayList<>(diagnostics.optimizations());
      optimizations.add(
          new OptimizationDecision(
              "cact-rotated-codebook",
              OptimizationStatus.ENABLED,
              "packed CQ weights execute directly from the mapped artifact",
              Map.of("codebook-values", Integer.toString(header.codebookLength()))));
      return new BackendDiagnostics(
          diagnostics.backend(), diagnostics.planVersion(), environment, optimizations);
    }
    if (!(decoder instanceof Gemma4DecoderAdapter gemma4)) {
      return diagnostics;
    }
    int batchSize = gemma4.prefillBatchSize();
    Map<String, String> environment = new LinkedHashMap<>(diagnostics.environment());
    environment.put("architecture-prefill-batch-size", Integer.toString(batchSize));
    List<OptimizationDecision> optimizations = new ArrayList<>(diagnostics.optimizations());
    optimizations.add(
        new OptimizationDecision(
            "gemma4-batched-prefill",
            batchSize > 1 ? OptimizationStatus.ENABLED : OptimizationStatus.UNSUPPORTED,
            batchSize > 1
                ? "Gemma 4 attention, shared FFN, router, and routed experts use retained batched kernels"
                : "at least one Gemma 4 projection or routed-expert tensor lacks a retained batched kernel",
            Map.of("batch-size", Integer.toString(batchSize))));
    return new BackendDiagnostics(
        diagnostics.backend(), diagnostics.planVersion(), environment, optimizations);
  }

  private static GgufTensorType tensorType(GgufFile file, String name) {
    return file.getTensor(name).type();
  }

  private static String modelName(Path modelPath, GgufFile file) {
    return file.metadata().getString("general.name").orElse(modelPath.getFileName().toString());
  }

  private static Map<String, String> recommendations(
      BackendConfiguration backendConfiguration, GgufBatchedMatrixKernel batchedMatrixKernel) {
    Map<String, String> combined = new LinkedHashMap<>(backendConfiguration.recommendations());
    combined.putAll(batchedMatrixKernel.planRecommendations());
    return Map.copyOf(combined);
  }

  private static void closeAfterFailure(
      PureJavaDecoder decoder,
      ModelMemoryArena arenaOwner,
      GgufBatchedMatrixKernel batchedMatrixKernel,
      BatchedCausalAttentionKernel batchedAttentionKernel,
      Throwable failure) {
    if (decoder != null) {
      try {
        decoder.close();
      } catch (IOException | RuntimeException closeFailure) {
        failure.addSuppressed(closeFailure);
      }
    }
    try {
      batchedMatrixKernel.close();
    } catch (RuntimeException closeFailure) {
      failure.addSuppressed(closeFailure);
    }
    try {
      batchedAttentionKernel.close();
    } catch (RuntimeException closeFailure) {
      failure.addSuppressed(closeFailure);
    }
    try {
      arenaOwner.close();
    } catch (RuntimeException closeFailure) {
      failure.addSuppressed(closeFailure);
    }
  }

  private static RuntimeException combineCloseFailures(
      RuntimeException current, RuntimeException next) {
    if (current == null) {
      return next;
    }
    current.addSuppressed(next);
    return current;
  }
}
