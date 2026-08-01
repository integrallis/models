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
package com.integrallis.models.bench;

import java.lang.management.BufferPoolMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** JVM-owned memory observed at one inference boundary. */
record JvmMemorySnapshot(
    long heapUsedBytes,
    long heapCommittedBytes,
    long nonHeapUsedBytes,
    long nonHeapCommittedBytes,
    Map<String, BufferPoolUsage> bufferPools,
    NativeMemoryTracking.Summary nativeMemory) {

  JvmMemorySnapshot {
    bufferPools = Map.copyOf(bufferPools);
  }

  static JvmMemorySnapshot capture() {
    MemoryMXBean memory = ManagementFactory.getMemoryMXBean();
    List<BufferPoolUsage> bufferPools = new ArrayList<>();
    for (BufferPoolMXBean pool : ManagementFactory.getPlatformMXBeans(BufferPoolMXBean.class)) {
      bufferPools.add(
          new BufferPoolUsage(
              pool.getName(), pool.getCount(), pool.getMemoryUsed(), pool.getTotalCapacity()));
    }
    return from(
        memory.getHeapMemoryUsage(),
        memory.getNonHeapMemoryUsage(),
        bufferPools,
        NativeMemoryTracking.capture());
  }

  static JvmMemorySnapshot from(
      MemoryUsage heap,
      MemoryUsage nonHeap,
      Iterable<BufferPoolUsage> bufferPools,
      NativeMemoryTracking.Summary nativeMemory) {
    Map<String, BufferPoolUsage> poolsByName = new LinkedHashMap<>();
    for (BufferPoolUsage pool : bufferPools) {
      poolsByName.put(pool.name(), pool);
    }
    return new JvmMemorySnapshot(
        heap.getUsed(),
        heap.getCommitted(),
        nonHeap.getUsed(),
        nonHeap.getCommitted(),
        poolsByName,
        nativeMemory);
  }

  record BufferPoolUsage(String name, long count, long memoryUsedBytes, long totalCapacityBytes) {}
}
