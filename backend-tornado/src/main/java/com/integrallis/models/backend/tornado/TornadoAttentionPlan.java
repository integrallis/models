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
package com.integrallis.models.backend.tornado;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * One fixed-shape TornadoVM attention plan holding a device-resident KV mirror for one layer.
 *
 * <p>The mirror buffers transfer once ({@link DataTransferMode#FIRST_EXECUTION}) and then stay on
 * the device for the life of the plan; every later execution moves only the staged rows, the query
 * block, and the output block. Backfill of rows the Java path cached before the device saw this
 * sequence runs through the same graph with the attend rows set to zero, so exactly one task graph
 * ever owns the mirror buffers — two graphs sharing them would depend on how the runtime associates
 * a buffer with a device across plans, which is not something this code should assume.
 */
final class TornadoAttentionPlan implements AttentionPlan {
  private final TornadoAttentionShape shape;
  private final float[] stagedKey;
  private final float[] stagedValue;
  private final float[] stagedQuery;
  private final IntArray state;
  private final FloatArray deviceKey;
  private final FloatArray deviceValue;
  private final FloatArray deviceQuery;
  private final FloatArray deviceOutput;
  private final TornadoExecutionPlan plan;

  TornadoAttentionPlan(String name, TornadoAttentionShape shape) {
    this.shape = shape;
    int keyDim = shape.keyDim();
    int valueDim = shape.valueDim();
    this.stagedKey = new float[Math.multiplyExact(shape.stagingRows(), keyDim)];
    this.stagedValue = new float[Math.multiplyExact(shape.stagingRows(), valueDim)];
    this.stagedQuery = new float[Math.multiplyExact(shape.executionBatchSize(), shape.queryDim())];
    this.state = new IntArray(TornadoAttentionKernel.STATE_LENGTH);
    this.deviceKey = new FloatArray(stagedKey.length);
    this.deviceValue = new FloatArray(stagedValue.length);
    this.deviceQuery = new FloatArray(stagedQuery.length);
    this.deviceOutput =
        new FloatArray(Math.multiplyExact(shape.executionBatchSize(), shape.outputDim()));
    FloatArray keyMirror = new FloatArray(Math.multiplyExact(shape.maxSequenceLength(), keyDim));
    FloatArray valueMirror =
        new FloatArray(Math.multiplyExact(shape.maxSequenceLength(), valueDim));
    FloatArray scores =
        new FloatArray(
            Math.multiplyExact(
                Math.multiplyExact(shape.executionBatchSize(), shape.numHeads()),
                shape.maxSequenceLength()));
    TornadoAttentionKernel.validate(
        deviceQuery,
        deviceOutput,
        scores,
        shape.executionBatchSize(),
        shape.numHeads(),
        shape.numKvHeads(),
        shape.keyLength(),
        shape.valueLength(),
        shape.maxSequenceLength());
    TaskGraph graph =
        new TaskGraph(name)
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, keyMirror, valueMirror, scores)
            .transferToDevice(
                DataTransferMode.EVERY_EXECUTION, state, deviceKey, deviceValue, deviceQuery)
            .task(
                "store",
                TornadoAttentionKernel::store,
                state,
                deviceKey,
                deviceValue,
                keyMirror,
                valueMirror,
                keyDim,
                valueDim)
            .task(
                "attend",
                TornadoAttentionKernel::attend,
                state,
                deviceQuery,
                keyMirror,
                valueMirror,
                scores,
                deviceOutput,
                shape.numHeads(),
                shape.groupSize(),
                shape.keyLength(),
                shape.valueLength(),
                keyDim,
                valueDim,
                shape.maxSequenceLength(),
                shape.slidingWindow(),
                shape.scale())
            .transferToHost(DataTransferMode.EVERY_EXECUTION, deviceOutput);
    this.plan = new TornadoExecutionPlan(graph.snapshot());
  }

  @Override
  public int mirrorChunkPositions() {
    return shape.stagingRows();
  }

  @Override
  public void mirror(
      int firstPosition,
      int positionCount,
      float[] keys,
      int keyOffset,
      int keyRowStride,
      float[] values,
      int valueOffset,
      int valueRowStride) {
    int keyDim = shape.keyDim();
    int valueDim = shape.valueDim();
    for (int row = 0; row < positionCount; row++) {
      System.arraycopy(keys, keyOffset + row * keyRowStride, stagedKey, row * keyDim, keyDim);
      System.arraycopy(
          values, valueOffset + row * valueRowStride, stagedValue, row * valueDim, valueDim);
    }
    deviceKey.getSegment().copyFrom(MemorySegment.ofArray(stagedKey));
    deviceValue.getSegment().copyFrom(MemorySegment.ofArray(stagedValue));
    state.set(TornadoAttentionKernel.STORE_POSITION_INDEX, firstPosition);
    state.set(TornadoAttentionKernel.STORE_ROWS_INDEX, positionCount);
    state.set(TornadoAttentionKernel.ATTEND_POSITION_INDEX, 0);
    state.set(TornadoAttentionKernel.ATTEND_ROWS_INDEX, 0);
    plan.execute();
  }

  @Override
  public void attend(
      float[] output, float[] query, float[] key, float[] value, int startPosition, int batchSize) {
    int keyDim = shape.keyDim();
    int valueDim = shape.valueDim();
    stage(key, stagedKey, Math.multiplyExact(batchSize, keyDim));
    stage(value, stagedValue, Math.multiplyExact(batchSize, valueDim));
    stage(query, stagedQuery, Math.multiplyExact(batchSize, shape.queryDim()));
    deviceKey.getSegment().copyFrom(MemorySegment.ofArray(stagedKey));
    deviceValue.getSegment().copyFrom(MemorySegment.ofArray(stagedValue));
    deviceQuery.getSegment().copyFrom(MemorySegment.ofArray(stagedQuery));
    state.set(TornadoAttentionKernel.STORE_POSITION_INDEX, startPosition);
    state.set(TornadoAttentionKernel.STORE_ROWS_INDEX, batchSize);
    state.set(TornadoAttentionKernel.ATTEND_POSITION_INDEX, startPosition);
    state.set(TornadoAttentionKernel.ATTEND_ROWS_INDEX, batchSize);
    plan.execute();
    long byteSize = Math.multiplyExact((long) batchSize * shape.outputDim(), (long) Float.BYTES);
    MemorySegment.ofArray(output)
        .asSlice(0, byteSize)
        .copyFrom(deviceOutput.getSegment().asSlice(0, byteSize));
  }

  private static void stage(float[] source, float[] staged, int entries) {
    System.arraycopy(source, 0, staged, 0, entries);
    Arrays.fill(staged, entries, staged.length, 0.0f);
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
