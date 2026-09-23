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
package com.integrallis.models.backend.cuda;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Routes eligible K-quant projections and decode attention onto our own PTX kernels.
 *
 * <p>Implements the same SPI the TornadoVM and Vector API paths implement, so the forward passes
 * need no change: each of the per-projection dispatch helpers in {@code LlamaForwardPass} (and the
 * Gemma 4 and Qwen 3.5 equivalents) already asks {@code isEligible} and falls back to {@code
 * TensorOps} when the answer is no. Fallback is therefore automatic and per projection.
 *
 * <h2>What is on the device, and what is not</h2>
 *
 * Two kernels: the fused K-quant projection and decode attention. Model parsing, the tokenizer, the
 * graph, norms, RoPE, residuals, sampling and the generation loop stay in Java. So do activation
 * quantisation ({@link Q8KActivations}) and KV cache ownership.
 *
 * <h2>Weights are uploaded once, from the mapped file</h2>
 *
 * {@link #weightResidency} keys on the weight segment's address, so a tensor is uploaded the first
 * time it is projected and never again, whatever batch shape asks for it later. The upload reads
 * the mapped GGUF segment directly — there is no host-side duplicate at any point. The TornadoVM
 * arm cannot do this today: {@code ByteArray.fromSegment} copies into a fresh off-heap array and
 * prefill and decode are separate plans, which the large-model analysis costs at about 33 GB of
 * host memory for a 26B model. That difference is measured by {@link
 * CudaRoutingCounters#weightUploadBytes()}, not asserted.
 *
 * <h2>KV cache ownership, and gate G6</h2>
 *
 * The KV cache stays in Java. The bandwidth arithmetic does not force it onto the device for the
 * decode step: one step attends {@code positions * (keyDim + valueDim) * 4} bytes for a single
 * layer, which at a 1,024-slot sliding window and Gemma 4's 2,048-wide keys is about 16 MB — but
 * the host already holds those rows contiguously, and {@code cuMemcpyHtoD} of a contiguous span is
 * bandwidth-bound rather than latency-bound. Mirroring would save that copy at the cost of owning
 * per-sequence device state.
 *
 * <p>That cost is the reason not to pay it yet. Our published physical KV prefix-sharing claim
 * rests on two sequences forked from one prefix pointing at the <em>same</em> host rows. A
 * device-resident mirror cannot preserve that without a sequence identity the current SPI does not
 * carry on this branch, and a mirror per fork silently loses the sharing rather than failing. So
 * this kernel {@linkplain #isAttentionEligible refuses the accelerator} whenever the attended
 * window is not a single contiguous span — which is exactly the shape a shared prefix produces —
 * and the refusal is counted. Sharing is preserved by construction, and the refusal is observable
 * rather than assumed.
 */
public final class CudaGgufBatchedMatrixKernel implements GgufBatchedMatrixKernel {

  /** System property selecting the CUDA device ordinal. */
  public static final String DEVICE_ORDINAL_PROPERTY = "models.cuda.device";

  /** System property that disables this backend outright. */
  public static final String DISABLED_PROPERTY = "models.cuda.disabled";

  /**
   * System property that refuses only the decode-attention kernel, leaving the projections routed.
   *
   * <p>A per-stage ablation switch, not a feature flag. The projections are bit-exact by
   * construction; attention carries a stated 2.0e-5 relative-L2 contract because of {@code expf}
   * (see {@code UPSTREAM.md} CU-005). So when G1 diverges at a token where both stages routed,
   * there are two live suspects and no way to tell them apart from the report alone. {@code
   * CudaParityRun} literally tells the reader to "re-run with attention refused"; this is the
   * switch that makes that re-run possible without recompiling anything.
   *
   * <p>The refusal is <b>counted</b>, under the reason {@code ablated-by-models.cuda.attention
   * .disabled}, once per attention operation diverted. An ablation whose only evidence is an
   * absence cannot be distinguished from a stage that was never reached, and a toggle that silently
   * does nothing reads in the results as "this stage does not matter".
   */
  public static final String ATTENTION_DISABLED_PROPERTY = "models.cuda.attention.disabled";

  private final CudaDriver driver;
  private final BundledPtxModule module;
  private final CudaRoutingCounters counters;
  private final MemorySegment q4kProjection;
  private final MemorySegment q6kProjection;
  private final MemorySegment gqaAttention;
  private final Arena arena;

  /** Device pointers for weight tensors, keyed on the mapped segment address. */
  private final Map<Long, DeviceBuffer> weightResidency = new HashMap<>();

  private DeviceBuffer quantScratch;
  private DeviceBuffer scaleScratch;
  private DeviceBuffer sumScratch;
  private DeviceBuffer outputScratch;
  private byte[] hostQuants = new byte[0];
  private float[] hostScales = new float[0];
  private short[] hostSums = new short[0];
  private boolean closed;

  private CudaGgufBatchedMatrixKernel(
      CudaDriver driver, BundledPtxModule module, CudaRoutingCounters counters) {
    this.driver = driver;
    this.module = module;
    this.counters = counters;
    this.arena = Arena.ofShared();
    MemorySegment loaded = driver.loadModule(module.ptx());
    this.q4kProjection = driver.function(loaded, CudaKernelAbi.Q4_K_PROJECTION);
    this.q6kProjection = driver.function(loaded, CudaKernelAbi.Q6_K_PROJECTION);
    this.gqaAttention = driver.function(loaded, CudaKernelAbi.GQA_ATTENTION);
  }

  /**
   * Opens the accelerator, or explains why it is unavailable.
   *
   * <p>Never throws for an absent, old or ineligible device. Gate G3 requires that a build with
   * this module present but ineligible performs within 5% of a build without it, so the failure
   * path must not allocate, retry, or log per operation.
   */
  public static Status open() {
    if (Boolean.getBoolean(DISABLED_PROPERTY)) {
      return Status.unavailable("disabled by " + DISABLED_PROPERTY);
    }
    BundledPtxModule module;
    try {
      module = BundledPtxModule.resolve();
    } catch (RuntimeException failure) {
      return Status.unavailable("PTX module unavailable: " + failure.getMessage());
    }
    int ordinal = Integer.getInteger(DEVICE_ORDINAL_PROPERTY, 0);
    CudaDriver.Result result = CudaDriver.open(ordinal);
    if (!result.isAvailable()) {
      return Status.unavailable(result.reason());
    }
    CudaDriver driver = result.driver().orElseThrow();
    if (driver.computeCapability() < module.minimumComputeCapability()) {
      String reason =
          "device compute capability "
              + driver.computeCapability()
              + " is below the PTX target "
              + module.target();
      driver.close();
      return Status.unavailable(reason);
    }
    try {
      CudaRoutingCounters counters = new CudaRoutingCounters();
      CudaGgufBatchedMatrixKernel kernel =
          new CudaGgufBatchedMatrixKernel(driver, module, counters);
      return Status.available(kernel, driver, module, counters);
    } catch (RuntimeException failure) {
      driver.close();
      return Status.unavailable("PTX module could not be loaded: " + failure.getMessage());
    }
  }

  @Override
  public String implementation() {
    return "cuda-ptx";
  }

  /** Per-format, per-stage routing counters for this kernel (gate G2). */
  public CudaRoutingCounters counters() {
    return counters;
  }

  /** The loaded module's SHA-256, reported as the kernel identity with every result. */
  public String kernelSha256() {
    return module.sha256();
  }

  @Override
  public boolean supports(GgufTensorType type) {
    return type == GgufTensorType.Q4_K || type == GgufTensorType.Q6_K;
  }

  @Override
  public boolean isEligible(GgufTensorType type, int batchSize, int rows, int cols) {
    if (closed || !supports(type)) {
      return false;
    }
    return isShapeEligible(batchSize, rows, cols);
  }

  /**
   * Whether a projection shape can run on the device.
   *
   * <p>Separate from {@link #isEligible} so the reasons are testable without a device. Note that no
   * minimum batch size is imposed. The TornadoVM kernel requires {@code batchSize >= 4}, and {@code
   * Gemma4ForwardPass.supportsBatchedPrefill} probes with {@code batchSize == 2}, so Gemma 4 fails
   * that probe for every tensor regardless of format. Accepting any positive batch size avoids
   * inheriting that coupling; decode, the case G4 measures, is {@code batchSize == 1}.
   */
  static boolean isShapeEligible(int batchSize, int rows, int cols) {
    if (batchSize < 1 || rows < 1 || cols < 1) {
      return false;
    }
    if (cols % CudaKernelAbi.SUPER_BLOCK_VALUES != 0) {
      return false;
    }
    int blocksPerRow = cols / CudaKernelAbi.SUPER_BLOCK_VALUES;
    // The device kernels index a fixed shared-memory scratch by super-block. A wider row is
    // refused here rather than overrunning it there.
    return blocksPerRow <= CudaKernelAbi.MAX_BLOCKS_PER_ROW;
  }

  @Override
  public boolean supportsTriple(
      GgufTensorType firstType, GgufTensorType secondType, GgufTensorType thirdType) {
    // Q4_K_M promotes the value projection to Q6_K while query and key stay Q4_K, so the fused
    // QKV group is literally Q4_K/Q4_K/Q6_K. Refusing mixed families would leave the attention
    // projections on the CPU for every Q4_K_M model, which is every large model in the catalog.
    return supports(firstType) && supports(secondType) && supports(thirdType);
  }

  @Override
  public boolean isTripleEligible(
      GgufTensorType firstType,
      int firstRows,
      GgufTensorType secondType,
      int secondRows,
      GgufTensorType thirdType,
      int thirdRows,
      int batchSize,
      int cols) {
    return supportsTriple(firstType, secondType, thirdType)
        && isShapeEligible(batchSize, firstRows, cols)
        && isShapeEligible(batchSize, secondRows, cols)
        && isShapeEligible(batchSize, thirdRows, cols);
  }

  @Override
  public void multiplyTriple(
      float[] firstOutput,
      MemorySegment firstWeights,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOutput,
      MemorySegment secondWeights,
      GgufTensorType secondType,
      int secondRows,
      float[] thirdOutput,
      MemorySegment thirdWeights,
      GgufTensorType thirdType,
      int thirdRows,
      float[] input,
      int batchSize,
      int cols) {
    // One activation quantisation, three launches against the resident weights. The three
    // projections share the staged activation buffers, which is the point of the fused form.
    stageActivations(input, batchSize, cols);
    project(firstOutput, firstWeights, firstType, firstRows, batchSize, cols, false);
    project(secondOutput, secondWeights, secondType, secondRows, batchSize, cols, false);
    project(thirdOutput, thirdWeights, thirdType, thirdRows, batchSize, cols, false);
  }

  @Override
  public void multiply(
      float[] output,
      float[] input,
      MemorySegment weights,
      GgufTensorType type,
      int batchSize,
      int rows,
      int cols) {
    stageActivations(input, batchSize, cols);
    project(output, weights, type, rows, batchSize, cols, false);
  }

  /**
   * Whether a decode attention step can run on the device.
   *
   * <p>The second span must be empty. A non-empty second span is what a physically shared KV prefix
   * looks like from here, and serving it would require device state this kernel does not own;
   * refusing keeps the sharing intact. See the class documentation and gate G6.
   */
  static boolean isAttentionEligible(
      int positionsA, int positionsB, int numHeads, int numKvHeads, int keyLength) {
    if (positionsB != 0) {
      return false;
    }
    return positionsA > 0
        && numHeads > 0
        && numKvHeads > 0
        && numHeads % numKvHeads == 0
        && keyLength > 0;
  }

  @Override
  public boolean supportsGroupedAttention() {
    if (closed) {
      return false;
    }
    if (Boolean.getBoolean(ATTENTION_DISABLED_PROPERTY)) {
      // Counted, not silent: see ATTENTION_DISABLED_PROPERTY. This is asked once per attention
      // operation, so the count is exactly how many the ablation diverted back to the CPU.
      counters.refused(CudaStage.DECODE_ATTENTION, "ablated-by-" + ATTENTION_DISABLED_PROPERTY);
      return false;
    }
    return true;
  }

  @Override
  public void groupedAttention(
      float[] query,
      int queryOffset,
      float[] keysA,
      int keysAOffset,
      float[] valuesA,
      int valuesAOffset,
      int positionsA,
      float[] keysB,
      int keysBOffset,
      float[] valuesB,
      int valuesBOffset,
      int positionsB,
      float[] output,
      int outputOffset,
      float[] scores,
      int keyDim,
      int valueDim,
      int keyLength,
      int valueLength,
      int numHeads,
      int numKvHeads,
      float scale) {
    if (Boolean.getBoolean(ATTENTION_DISABLED_PROPERTY)) {
      counters.refused(CudaStage.DECODE_ATTENTION, "ablated-by-" + ATTENTION_DISABLED_PROPERTY);
      throw new UnsupportedOperationException(
          "CUDA attention refused: ablated by " + ATTENTION_DISABLED_PROPERTY);
    }
    if (!isAttentionEligible(positionsA, positionsB, numHeads, numKvHeads, keyLength)) {
      counters.refused(
          CudaStage.DECODE_ATTENTION,
          positionsB != 0 ? "shared-kv-prefix-span" : "unsupported-attention-shape");
      throw new UnsupportedOperationException(
          "CUDA attention refused: the attended window is not a single contiguous span");
    }
    long queryBytes = (long) numHeads * keyLength * Float.BYTES;
    long keyBytes = (long) positionsA * keyDim * Float.BYTES;
    long valueBytes = (long) positionsA * valueDim * Float.BYTES;
    long scoreBytes = (long) numHeads * positionsA * Float.BYTES;
    long outputBytes = (long) numHeads * valueLength * Float.BYTES;

    try (Arena call = Arena.ofConfined()) {
      DeviceBuffer queryBuffer = upload(call, query, queryOffset, numHeads * keyLength);
      DeviceBuffer keyBuffer = upload(call, keysA, keysAOffset, positionsA * keyDim);
      DeviceBuffer valueBuffer = upload(call, valuesA, valuesAOffset, positionsA * valueDim);
      DeviceBuffer scoreBuffer = allocateTracked(scoreBytes);
      DeviceBuffer outputBuffer = allocateTracked(outputBytes);
      try {
        MemorySegment parameters =
            parameterArray(
                call,
                pointer(call, queryBuffer.address()),
                pointer(call, keyBuffer.address()),
                pointer(call, valueBuffer.address()),
                pointer(call, scoreBuffer.address()),
                pointer(call, outputBuffer.address()),
                integer(call, numHeads),
                integer(call, numKvHeads),
                integer(call, keyLength),
                integer(call, valueLength),
                integer(call, keyDim),
                integer(call, valueDim),
                integer(call, positionsA),
                floating(call, scale));
        driver.launch(gqaAttention, numHeads, CudaKernelAbi.BLOCK_THREADS, parameters);
        counters.launched();
        driver.synchronize();

        MemorySegment host = call.allocate(outputBytes);
        driver.copyToHost(host, outputBuffer.address(), outputBytes);
        counters.copiedToHost(outputBytes);
        MemorySegment.copy(
            host, ValueLayout.JAVA_FLOAT, 0, output, outputOffset, numHeads * valueLength);
        counters.accelerated(GgufTensorType.F32, CudaStage.DECODE_ATTENTION, numHeads);
      } finally {
        scoreBuffer.free(driver, counters);
        outputBuffer.free(driver, counters);
        queryBuffer.free(driver, counters);
        keyBuffer.free(driver, counters);
        valueBuffer.free(driver, counters);
      }
    }
    counters.copiedToDevice(queryBytes + keyBytes + valueBytes);
  }

  /** Quantises and uploads one activation plane, reusing the staging buffers across calls. */
  private void stageActivations(float[] input, int batchSize, int cols) {
    int quants = Q8KActivations.quantCount(batchSize, cols);
    int scales = Q8KActivations.scaleCount(batchSize, cols);
    int sums = Q8KActivations.sumCount(batchSize, cols);
    if (hostQuants.length < quants) {
      hostQuants = new byte[quants];
    }
    if (hostScales.length < scales) {
      hostScales = new float[scales];
    }
    if (hostSums.length < sums) {
      hostSums = new short[sums];
    }
    Q8KActivations.quantize(input, batchSize, cols, hostQuants, hostScales, hostSums);

    quantScratch = ensure(quantScratch, (long) quants);
    scaleScratch = ensure(scaleScratch, (long) scales * Float.BYTES);
    sumScratch = ensure(sumScratch, (long) sums * Short.BYTES);
    try (Arena call = Arena.ofConfined()) {
      MemorySegment staged = call.allocate((long) quants);
      MemorySegment.copy(hostQuants, 0, staged, ValueLayout.JAVA_BYTE, 0, quants);
      driver.copyToDevice(quantScratch.address(), staged, quants);

      MemorySegment stagedScales = call.allocate((long) scales * Float.BYTES);
      MemorySegment.copy(hostScales, 0, stagedScales, ValueLayout.JAVA_FLOAT, 0, scales);
      driver.copyToDevice(scaleScratch.address(), stagedScales, (long) scales * Float.BYTES);

      MemorySegment stagedSums = call.allocate((long) sums * Short.BYTES);
      MemorySegment.copy(hostSums, 0, stagedSums, ValueLayout.JAVA_SHORT, 0, sums);
      driver.copyToDevice(sumScratch.address(), stagedSums, (long) sums * Short.BYTES);
    }
    counters.copiedToDevice(
        (long) quants + (long) scales * Float.BYTES + (long) sums * Short.BYTES);
  }

  /** Launches one projection against already-staged activations. */
  private void project(
      float[] output,
      MemorySegment weights,
      GgufTensorType type,
      int rows,
      int batchSize,
      int cols,
      boolean unusedGrouped) {
    DeviceBuffer resident = resident(weights, type, rows, cols);
    long outputBytes = (long) batchSize * rows * Float.BYTES;
    outputScratch = ensure(outputScratch, outputBytes);
    CudaStage stage = batchSize == 1 ? CudaStage.DECODE_PROJECTION : CudaStage.PREFILL_PROJECTION;

    try (Arena call = Arena.ofConfined()) {
      for (int batch = 0; batch < batchSize; batch++) {
        MemorySegment parameters =
            type == GgufTensorType.Q4_K
                ? parameterArray(
                    call,
                    pointer(call, resident.address()),
                    pointer(call, quantScratch.address()),
                    pointer(call, scaleScratch.address()),
                    pointer(call, sumScratch.address()),
                    pointer(call, outputScratch.address()),
                    integer(call, rows),
                    integer(call, cols),
                    integer(call, batch))
                : parameterArray(
                    call,
                    pointer(call, resident.address()),
                    pointer(call, quantScratch.address()),
                    pointer(call, scaleScratch.address()),
                    pointer(call, outputScratch.address()),
                    integer(call, rows),
                    integer(call, cols),
                    integer(call, batch));
        driver.launch(
            type == GgufTensorType.Q4_K ? q4kProjection : q6kProjection,
            rows,
            CudaKernelAbi.BLOCK_THREADS,
            parameters);
        counters.launched();
      }
      driver.synchronize();
      MemorySegment host = call.allocate(outputBytes);
      driver.copyToHost(host, outputScratch.address(), outputBytes);
      MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, output, 0, batchSize * rows);
    }
    counters.copiedToHost(outputBytes);
    counters.accelerated(type, stage, 1);
    if (batchSize == 1) {
      // A projection, not a token. Only the measurement harness can see token boundaries; see
      // CudaRoutingCounters#decodeStep.
      counters.decodeProjection();
    }
  }

  /**
   * Returns the device copy of {@code weights}, uploading it the first time only.
   *
   * <p>Keyed on the mapped segment's address, so every batch shape shares one copy. The upload
   * reads the mapped file directly: no host-side duplicate is created here or anywhere.
   */
  private DeviceBuffer resident(MemorySegment weights, GgufTensorType type, int rows, int cols) {
    long key = weights.address();
    DeviceBuffer existing = weightResidency.get(key);
    if (existing != null) {
      return existing;
    }
    long bytes = (long) rows * (cols / type.blockSize()) * type.typeSize();
    DeviceBuffer buffer = allocateTracked(bytes);
    driver.copyToDevice(buffer.address(), weights, bytes);
    counters.uploadedWeights(bytes);
    weightResidency.put(key, buffer);
    return buffer;
  }

  private DeviceBuffer ensure(DeviceBuffer current, long bytes) {
    if (current != null && current.bytes() >= bytes) {
      return current;
    }
    if (current != null) {
      current.free(driver, counters);
    }
    return allocateTracked(bytes);
  }

  /** Allocates device memory and records it against the run's high-water mark. */
  private DeviceBuffer allocateTracked(long bytes) {
    long devicePointer = driver.allocate(bytes);
    counters.deviceAllocated(bytes);
    return new DeviceBuffer(devicePointer, bytes);
  }

  private DeviceBuffer upload(Arena call, float[] source, int offset, int elements) {
    long bytes = (long) elements * Float.BYTES;
    DeviceBuffer buffer = allocateTracked(bytes);
    MemorySegment staged = call.allocate(bytes);
    MemorySegment.copy(source, offset, staged, ValueLayout.JAVA_FLOAT, 0, elements);
    driver.copyToDevice(buffer.address(), staged, bytes);
    return buffer;
  }

  private static MemorySegment pointer(Arena arena, long value) {
    MemorySegment slot = arena.allocate(ValueLayout.JAVA_LONG);
    slot.set(ValueLayout.JAVA_LONG, 0, value);
    return slot;
  }

  private static MemorySegment integer(Arena arena, int value) {
    MemorySegment slot = arena.allocate(ValueLayout.JAVA_INT);
    slot.set(ValueLayout.JAVA_INT, 0, value);
    return slot;
  }

  private static MemorySegment floating(Arena arena, float value) {
    MemorySegment slot = arena.allocate(ValueLayout.JAVA_FLOAT);
    slot.set(ValueLayout.JAVA_FLOAT, 0, value);
    return slot;
  }

  private static MemorySegment parameterArray(Arena arena, MemorySegment... slots) {
    MemorySegment array = arena.allocate(ValueLayout.ADDRESS, slots.length);
    for (int index = 0; index < slots.length; index++) {
      array.setAtIndex(ValueLayout.ADDRESS, index, slots[index]);
    }
    return array;
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    weightResidency.values().forEach(buffer -> buffer.free(driver, counters));
    weightResidency.clear();
    for (DeviceBuffer buffer :
        new DeviceBuffer[] {quantScratch, scaleScratch, sumScratch, outputScratch}) {
      if (buffer != null) {
        buffer.free(driver, counters);
      }
    }
    arena.close();
    driver.close();
  }

  /** A device allocation and its size. */
  private record DeviceBuffer(long address, long bytes) {
    void free(CudaDriver driver, CudaRoutingCounters counters) {
      driver.free(address);
      counters.deviceFreed(bytes);
    }
  }

  /**
   * The outcome of opening the accelerator.
   *
   * <p>A value rather than an exception, because "no GPU on this host" is the ordinary case and the
   * caller must be able to take the Vector API path without paying for the attempt.
   */
  public record Status(
      Optional<CudaGgufBatchedMatrixKernel> kernel,
      String reason,
      String deviceName,
      int computeCapability,
      long deviceMemoryBytes,
      int driverVersion,
      String kernelSha256,
      String ptxTarget,
      String toolchain,
      Optional<CudaRoutingCounters> counters) {

    static Status available(
        CudaGgufBatchedMatrixKernel kernel,
        CudaDriver driver,
        BundledPtxModule module,
        CudaRoutingCounters counters) {
      return new Status(
          Optional.of(kernel),
          "",
          driver.deviceName(),
          driver.computeCapability(),
          driver.totalMemoryBytes(),
          driver.driverVersion(),
          module.sha256(),
          module.target(),
          module.toolchain(),
          Optional.of(counters));
    }

    static Status unavailable(String reason) {
      Objects.requireNonNull(reason, "reason");
      return new Status(Optional.empty(), reason, "none", 0, 0L, 0, "", "", "", Optional.empty());
    }

    /** Whether the accelerator is usable. */
    public boolean accelerated() {
      return kernel.isPresent();
    }
  }
}
