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

import java.util.Objects;

/**
 * What one model would place on an accelerator, in the terms the capacity gate can add up.
 *
 * <p>A request is either <em>coarse</em> or <em>detailed</em>. A coarse request is derived from the
 * GGUF file size alone: it knows the upper bound on uploaded weights and nothing else, so retained
 * plan count, largest single tensor, and device KV residency are all reported as unknown rather
 * than silently assumed to be zero. A detailed request is built from the tensor inventory and
 * carries all of those terms.
 *
 * @param modelLabel human-readable model identity used in ineligibility messages
 * @param weightBytes bytes of model weights that would be uploaded once, per retained shape
 * @param retainedShapes distinct batch shapes whose plans are held at once (prefill, or prefill and
 *     decode)
 * @param retainedPlanCount distinct {@code (weight, shape)} plans that would be retained; {@code 0}
 *     means unknown
 * @param planScratchBytes device activation, scale, and output buffers summed over every retained
 *     plan; {@code 0} means unknown
 * @param largestAllocationBytes largest single device allocation the plans would ask for, whether a
 *     weight tensor or a scratch buffer; {@code 0} means unknown
 * @param deviceKvCacheBytes KV cache bytes that would live on the device at the planned context
 *     length; {@code 0} means the KV cache stays in host memory
 * @param detailed whether the weight, plan, and tensor terms came from a real tensor inventory
 */
public record DeviceMemoryRequest(
    String modelLabel,
    long weightBytes,
    int retainedShapes,
    int retainedPlanCount,
    long planScratchBytes,
    long largestAllocationBytes,
    long deviceKvCacheBytes,
    boolean detailed) {

  public DeviceMemoryRequest {
    modelLabel = requireLabel(modelLabel);
    requirePositive("weightBytes", weightBytes);
    if (retainedShapes < 1) {
      throw new IllegalArgumentException("retainedShapes must be at least 1: " + retainedShapes);
    }
    requireNotNegative("retainedPlanCount", retainedPlanCount);
    requireNotNegative("planScratchBytes", planScratchBytes);
    requireNotNegative("largestAllocationBytes", largestAllocationBytes);
    requireNotNegative("deviceKvCacheBytes", deviceKvCacheBytes);
  }

  /**
   * Builds the coarse request the loader can form before parsing a model: the whole GGUF file is
   * treated as uploadable weights and nothing else is known.
   *
   * <p>This is what {@code TornadoBackend.open} has available, and it reproduces the budget the
   * published A16-2Q and A40-4Q qualification runs were admitted under.
   */
  public static DeviceMemoryRequest ofModelFile(
      String modelLabel, long fileBytes, boolean accelerateDecode) {
    return new DeviceMemoryRequest(
        modelLabel, fileBytes, accelerateDecode ? 2 : 1, 0, 0L, 0L, 0L, false);
  }

  /** Starts a detailed request built from a real tensor inventory. */
  public static Builder detailed(String modelLabel) {
    return new Builder(modelLabel);
  }

  /** Returns this request with a device-resident KV cache of the given size. */
  public DeviceMemoryRequest withDeviceKvCacheBytes(long bytes) {
    return new DeviceMemoryRequest(
        modelLabel,
        weightBytes,
        retainedShapes,
        retainedPlanCount,
        planScratchBytes,
        largestAllocationBytes,
        bytes,
        detailed);
  }

  /** Mutable assembly for a detailed request. */
  public static final class Builder {
    private final String modelLabel;
    private long weightBytes;
    private int retainedShapes = 1;
    private int retainedPlanCount;
    private long planScratchBytes;
    private long largestAllocationBytes;
    private long deviceKvCacheBytes;

    private Builder(String modelLabel) {
      this.modelLabel = requireLabel(modelLabel);
    }

    /** Sets the bytes of weights uploaded once, per retained shape. */
    public Builder weightBytes(long bytes) {
      this.weightBytes = bytes;
      return this;
    }

    /** Sets how many batch shapes are retained at once. */
    public Builder retainedShapes(int shapes) {
      this.retainedShapes = shapes;
      return this;
    }

    /** Sets how many distinct {@code (weight, shape)} plans are retained. */
    public Builder retainedPlanCount(int plans) {
      this.retainedPlanCount = plans;
      return this;
    }

    /** Sets the device activation, scale, and output buffers summed over every retained plan. */
    public Builder planScratchBytes(long bytes) {
      this.planScratchBytes = bytes;
      return this;
    }

    /** Sets the largest single device allocation the plans would ask for. */
    public Builder largestAllocationBytes(long bytes) {
      this.largestAllocationBytes = bytes;
      return this;
    }

    /** Sets the KV cache bytes that would live on the device at the planned context length. */
    public Builder deviceKvCacheBytes(long bytes) {
      this.deviceKvCacheBytes = bytes;
      return this;
    }

    /** Builds the detailed request. */
    public DeviceMemoryRequest build() {
      return new DeviceMemoryRequest(
          modelLabel,
          weightBytes,
          retainedShapes,
          retainedPlanCount,
          planScratchBytes,
          largestAllocationBytes,
          deviceKvCacheBytes,
          true);
    }
  }

  private static String requireLabel(String value) {
    Objects.requireNonNull(value, "modelLabel");
    if (value.isBlank()) {
      throw new IllegalArgumentException("modelLabel must not be blank");
    }
    return value.strip();
  }

  private static void requirePositive(String name, long value) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive: " + value);
    }
  }

  private static void requireNotNegative(String name, long value) {
    if (value < 0) {
      throw new IllegalArgumentException(name + " must not be negative: " + value);
    }
  }
}
