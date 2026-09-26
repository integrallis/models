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

/**
 * How retained TornadoVM execution plans hold model weights on the device.
 *
 * <p>Only {@link #PER_SHAPE_WHOLE_MODEL} is implemented today. The other constants exist so the
 * capacity gate can say precisely which unimplemented plan shape a model would have needed, instead
 * of reporting an anonymous shortfall.
 */
public enum PlanShapeStrategy {

  /**
   * One {@code TaskGraph} per (weight tensor, batch shape), each owning its own device weight
   * buffer and its own host copy.
   *
   * <p>This is what {@code TornadoGgufBatchedMatrixKernel} builds: the plan cache is keyed on the
   * weight address plus the execution batch size, and the constructor calls {@code
   * ByteArray.fromSegment(weights)}, which copies the tensor into a fresh off-heap array before the
   * task graph pins it with {@code DataTransferMode.FIRST_EXECUTION}. With decode acceleration on,
   * every weight is therefore held twice on the device and twice on the host.
   */
  PER_SHAPE_WHOLE_MODEL(true, null),

  /**
   * One device weight buffer per tensor, shared by every batch shape that reads it.
   *
   * <p>Halves both device and host weight residency when prefill and decode plans are retained
   * together.
   */
  SHARED_WEIGHT_UPLOAD(
      false,
      "TornadoGgufBatchedMatrixKernel allocates a new ByteArray per (weight, batch shape) plan;"
          + " sharing one device buffer across shapes requires hoisting the weight array out of"
          + " the plan constructor and keying it on the weight address alone"),

  /**
   * A bounded resident working set: weights stay on the host and only the tensors needed by the
   * current layer or routed experts are uploaded, under an LRU bound.
   *
   * <p>Device capacity stops being the binding constraint and interconnect bandwidth becomes it, so
   * this strategy is never selected by the capacity gate; it is named only so an ineligibility
   * message can point at it.
   */
  BOUNDED_RESIDENT_WORKING_SET(
      false,
      "no eviction path exists: plans are held in unbounded LinkedHashMaps and only released on"
          + " kernel close, and per-token re-upload is bounded by PCIe bandwidth rather than by"
          + " device capacity");

  private final boolean implemented;
  private final String limitation;

  PlanShapeStrategy(boolean implemented, String limitation) {
    this.implemented = implemented;
    this.limitation = limitation;
  }

  /** Whether the shipped kernel can actually build plans in this shape. */
  public boolean implemented() {
    return implemented;
  }

  /** What stops this strategy from being available, or {@code null} when it is implemented. */
  public String limitation() {
    return limitation;
  }

  /** Whether device capacity for this strategy can be computed from a request. */
  public boolean capacityComputable() {
    return this != BOUNDED_RESIDENT_WORKING_SET;
  }

  /** Device bytes of model weights retained under this strategy. */
  long deviceWeightBytes(DeviceMemoryRequest request) {
    return switch (this) {
      case PER_SHAPE_WHOLE_MODEL ->
          Math.multiplyExact(request.weightBytes(), request.retainedShapes());
      case SHARED_WEIGHT_UPLOAD -> request.weightBytes();
      case BOUNDED_RESIDENT_WORKING_SET ->
          throw new IllegalStateException(
              "BOUNDED_RESIDENT_WORKING_SET has no capacity-derived weight budget");
    };
  }

  /**
   * Host bytes of off-heap weight copies retained under this strategy.
   *
   * <p>{@code ByteArray.fromSegment} copies; the mapped GGUF stays mapped alongside these copies.
   */
  long hostWeightCopyBytes(DeviceMemoryRequest request) {
    return deviceWeightBytes(request);
  }
}
