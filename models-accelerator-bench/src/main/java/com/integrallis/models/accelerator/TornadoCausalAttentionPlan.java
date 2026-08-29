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
package com.integrallis.models.accelerator;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** One fixed-shape causal-attention plan with a persistent device KV cache. */
final class TornadoCausalAttentionPlan implements AutoCloseable {
  private final int executionBatchSize;
  private final int queryDim;
  private final int keyDim;
  private final int valueDim;
  private final int outputDim;
  private final int maxSequenceLength;
  private final float[] paddedQuery;
  private final float[] paddedKey;
  private final float[] paddedValue;
  private final IntArray state;
  private final FloatArray deviceQuery;
  private final FloatArray deviceKey;
  private final FloatArray deviceValue;
  private final FloatArray deviceOutput;
  private final TornadoExecutionPlan plan;
  private long calls;
  private long totalNanos;

  TornadoCausalAttentionPlan(
      String name,
      int executionBatchSize,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int maxSequenceLength,
      int slidingWindow) {
    if (executionBatchSize < 1
        || numHeads < 1
        || numKvHeads < 1
        || numHeads % numKvHeads != 0
        || keyLength < 1
        || valueLength != keyLength
        || maxSequenceLength < executionBatchSize
        || slidingWindow != 0) {
      throw new IllegalArgumentException("invalid causal-attention shape");
    }
    this.executionBatchSize = executionBatchSize;
    this.queryDim = Math.multiplyExact(numHeads, keyLength);
    this.keyDim = Math.multiplyExact(numKvHeads, keyLength);
    this.valueDim = Math.multiplyExact(numKvHeads, valueLength);
    this.outputDim = Math.multiplyExact(numHeads, valueLength);
    this.maxSequenceLength = maxSequenceLength;
    this.paddedQuery = new float[Math.multiplyExact(executionBatchSize, queryDim)];
    this.paddedKey = new float[Math.multiplyExact(executionBatchSize, keyDim)];
    this.paddedValue = new float[Math.multiplyExact(executionBatchSize, valueDim)];
    this.state = new IntArray(2);
    this.deviceQuery = new FloatArray(paddedQuery.length);
    this.deviceKey = new FloatArray(paddedKey.length);
    this.deviceValue = new FloatArray(paddedValue.length);
    FloatArray deviceKeyCache = new FloatArray(Math.multiplyExact(maxSequenceLength, keyDim));
    FloatArray deviceValueCache = new FloatArray(Math.multiplyExact(maxSequenceLength, valueDim));
    this.deviceOutput = new FloatArray(Math.multiplyExact(executionBatchSize, outputDim));
    KernelContext context = new KernelContext();
    TaskGraph graph =
        new TaskGraph(name)
            .transferToDevice(
                DataTransferMode.FIRST_EXECUTION, context, deviceKeyCache, deviceValueCache)
            .transferToDevice(
                DataTransferMode.EVERY_EXECUTION, state, deviceQuery, deviceKey, deviceValue)
            .task(
                "store-kv",
                CausalAttentionKernel::store,
                state,
                deviceKey,
                deviceValue,
                deviceKeyCache,
                deviceValueCache,
                keyDim,
                valueDim)
            .task(
                "attention",
                CausalAttentionKernel::attendTiled,
                context,
                state,
                deviceQuery,
                deviceKeyCache,
                deviceValueCache,
                deviceOutput,
                numHeads,
                keyLength,
                keyDim,
                numHeads / numKvHeads,
                queryDim)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, deviceOutput);
    int localSize = optimalLocalSize(keyLength);
    int globalSize =
        Math.multiplyExact(Math.multiplyExact(executionBatchSize, numHeads), localSize);
    WorkerGrid workerGrid = new WorkerGrid1D(globalSize);
    workerGrid.setGlobalWork(globalSize, 1, 1);
    workerGrid.setLocalWork(localSize, 1, 1);
    GridScheduler scheduler = new GridScheduler(name + ".attention", workerGrid);
    this.plan = new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(scheduler);
  }

  void execute(
      float[] query,
      float[] key,
      float[] value,
      float[] output,
      int actualBatchSize,
      int startPosition) {
    if (actualBatchSize < 1 || actualBatchSize > executionBatchSize) {
      throw new IllegalArgumentException("actualBatchSize exceeds execution shape");
    }
    if (startPosition < 0 || startPosition > maxSequenceLength - actualBatchSize) {
      throw new IllegalArgumentException("prompt chunk exceeds attention context");
    }
    stage(query, paddedQuery, actualBatchSize, queryDim, "query");
    stage(key, paddedKey, actualBatchSize, keyDim, "key");
    stage(value, paddedValue, actualBatchSize, valueDim, "value");
    if (output.length < Math.multiplyExact(actualBatchSize, outputDim)) {
      throw new IllegalArgumentException("output storage does not match attention shape");
    }
    deviceQuery.getSegment().copyFrom(MemorySegment.ofArray(paddedQuery));
    deviceKey.getSegment().copyFrom(MemorySegment.ofArray(paddedKey));
    deviceValue.getSegment().copyFrom(MemorySegment.ofArray(paddedValue));
    state.set(0, startPosition);
    state.set(1, actualBatchSize);
    long started = System.nanoTime();
    plan.execute();
    long byteSize = Math.multiplyExact((long) actualBatchSize * outputDim, Float.BYTES);
    MemorySegment.ofArray(output)
        .asSlice(0, byteSize)
        .copyFrom(deviceOutput.getSegment().asSlice(0, byteSize));
    totalNanos += System.nanoTime() - started;
    calls++;
  }

  long calls() {
    return calls;
  }

  double totalMillis() {
    return totalNanos / 1_000_000.0;
  }

  private static void stage(
      float[] source, float[] padded, int actualBatchSize, int dimension, String name) {
    int actualEntries = Math.multiplyExact(actualBatchSize, dimension);
    if (source.length < actualEntries) {
      throw new IllegalArgumentException(name + " storage does not match attention shape");
    }
    System.arraycopy(source, 0, padded, 0, actualEntries);
    Arrays.fill(padded, actualEntries, padded.length, 0.0f);
  }

  private static int optimalLocalSize(int dimension) {
    int maximum = Math.min(dimension, 64);
    for (int candidate = maximum; candidate >= 1; candidate--) {
      if (dimension % candidate == 0) {
        return candidate;
      }
    }
    return 1;
  }

  @Override
  public void close() {
    try {
      plan.close();
    } catch (Exception exception) {
      throw new IllegalStateException("could not close Tornado attention plan", exception);
    }
  }
}
