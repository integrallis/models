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
package com.integrallis.models.backend.purejava.diagnostics;

import java.util.Objects;

/**
 * A named reason a fast path was not taken.
 *
 * <p>Each constant is reported from exactly the code branch that takes the slower path, at most
 * once per process (see {@link PerformanceCliffs#report(PerformanceCliff, String)}). The stable
 * {@link #id()} is the JFR event's {@code reason} field and the suffix of the {@code
 * performance-cliff.<id>} diagnostics key.
 */
public enum PerformanceCliff {
  /**
   * vectors-core selected its scalar provider (for example {@code -Dvectors.forceScalar=true}), so
   * the execution plan runs every vectors-core kernel without the Vector API.
   */
  VECTOR_API_UNAVAILABLE(
      "vector-api-unavailable", "vectors-core kernels run on the scalar provider"),
  /**
   * The active vector width is below the hardware-preferred width (for example {@code
   * -Dvectors.maxBits}).
   */
  VECTOR_WIDTH_CAPPED(
      "vector-width-capped", "the active vector species is narrower than the hardware preference"),
  /**
   * GGUF row parallelism is off ({@code -Dvectors.gguf.parallel=false}) on a host with more than
   * one processor, so projection rows run on the calling thread.
   */
  GGUF_PARALLEL_DISABLED(
      "gguf-parallel-disabled",
      "GGUF projection rows run serially although more than one processor is available"),
  /** vectors-core selected a GGUF row executor other than its persistent reusable workers. */
  PERSISTENT_EXECUTOR_NOT_USED(
      "persistent-executor-not-used", "GGUF rows do not run on persistent reusable workers"),
  /**
   * A pairwise Q4_0 kernel was requested but the active vector shape cannot execute it, so the
   * widened Q4 kernel runs instead.
   */
  Q4_PAIRWISE_KERNEL_UNSUPPORTED(
      "q4-pairwise-kernel-unsupported",
      "the requested pairwise Q4_0 kernel fell back to the widened kernel"),
  /**
   * At least one projection tensor type has no retained batched kernel, so the planned prefill
   * batch size falls to one token per step.
   */
  BATCHED_PREFILL_UNSUPPORTED_TENSOR_TYPE(
      "batched-prefill-unsupported-tensor-type",
      "a projection tensor type has no batched kernel, so prefill runs one token at a time"),
  /**
   * A batched Qwen 3.5 projection whose tensor type has no batched kernel is executed one row at a
   * time with per-call scratch allocation.
   */
  ROW_BY_ROW_PROJECTION(
      "row-by-row-projection",
      "a batched projection runs one row at a time because its tensor type has no batched kernel"),
  /**
   * Grouped-query attention runs head by head because the fused grouped kernel is wired only for
   * Granite, whose pinned numerics it was qualified against.
   */
  FUSED_GROUPED_ATTENTION_NOT_WIRED(
      "fused-grouped-attention-not-wired",
      "grouped-query attention runs head by head: the fused kernel is wired only for Granite"),
  /**
   * An injected native kernel is present on an architecture wired for native grouped attention, but
   * the kernel does not provide it (capability missing or disabled by property).
   */
  NATIVE_GROUPED_ATTENTION_UNAVAILABLE(
      "native-grouped-attention-unavailable",
      "the injected kernel has no native grouped attention; attention stays in Java"),
  /**
   * Native grouped attention is wired and available, but the sequence's KV cache view has more than
   * the two spans the kernel accepts (a prefix frozen from an already-forked branch), so that
   * attention row runs in Java.
   */
  NATIVE_GROUPED_ATTENTION_SPAN_LIMIT(
      "native-grouped-attention-span-limit",
      "the KV cache view has more than two spans; native grouped attention falls back to Java"),
  /**
   * An injected native kernel does not provide the Gated DeltaNet recurrence, so it runs in Java.
   */
  NATIVE_GATED_DELTA_NET_UNAVAILABLE(
      "native-gated-delta-net-unavailable",
      "the injected kernel has no Gated DeltaNet recurrence; it runs in Java"),
  /**
   * The loaded native kernel library lacks the poll-budget capability, so its workers keep the
   * library's built-in poll-before-park behaviour instead of the configured budget.
   */
  NATIVE_POLL_BUDGET_UNSUPPORTED(
      "native-poll-budget-unsupported",
      "the native library cannot apply the configured worker poll budget");

  private final String id;
  private final String description;

  PerformanceCliff(String id, String description) {
    this.id = Objects.requireNonNull(id, "id");
    this.description = Objects.requireNonNull(description, "description");
  }

  /** Stable kebab-case identifier used in JFR events and diagnostics keys. */
  public String id() {
    return id;
  }

  /** Human-readable summary of the slower path that was taken. */
  public String description() {
    return description;
  }
}
