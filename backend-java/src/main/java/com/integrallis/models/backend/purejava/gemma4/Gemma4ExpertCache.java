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
package com.integrallis.models.backend.purejava.gemma4;

import com.integrallis.models.backend.purejava.gemma4.Gemma4ExpertLoader.LoadedExpert;
import com.integrallis.models.backend.purejava.gemma4.Gemma4TensorLayout.ExpertWeights;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/** Bounded per-layer routed-expert cache with lease-safe LFU or LRU eviction. */
final class Gemma4ExpertCache implements AutoCloseable {

  enum CachePolicy {
    LFU,
    LRU
  }

  @FunctionalInterface
  interface ExpertResolver {
    ExpertWeights resolve(int layer, int expert);
  }

  record Stats(long hits, long misses, long bytesRead, long evictions, long waits) {}

  final class Lease implements AutoCloseable {
    private final Slot slot;
    private final LoadedExpert loaded;
    private final AtomicBoolean released = new AtomicBoolean();

    private Lease(Slot slot, LoadedExpert loaded) {
      this.slot = slot;
      this.loaded = loaded;
    }

    int layer() {
      requireLive();
      return loaded.weights().layer();
    }

    int expert() {
      requireLive();
      return loaded.weights().expert();
    }

    MemorySegment gateUp() {
      requireLive();
      return loaded.gateUp();
    }

    MemorySegment down() {
      requireLive();
      return loaded.down();
    }

    private void requireLive() {
      if (released.get()) {
        throw new IllegalStateException("Gemma 4 expert lease is closed");
      }
    }

    @Override
    public void close() {
      if (released.compareAndSet(false, true)) {
        release(slot);
      }
    }
  }

  private static final class LayerState {
    private final Slot[] slots;
    private final int[] expertUseCount;
    private final long slotBytes;

    private LayerState(int slotCount, int numExperts, long slotBytes) {
      this.slots = new Slot[slotCount];
      for (int slot = 0; slot < slotCount; slot++) {
        this.slots[slot] = new Slot();
      }
      this.expertUseCount = new int[numExperts];
      this.slotBytes = slotBytes;
    }
  }

  private static final class Slot {
    private MemorySegment storage;
    private LoadedExpert loaded;
    private int expert = -1;
    private int loadingExpert = -1;
    private int leases;
    private long lastUse;
    private boolean loading;
  }

  private final Gemma4ExpertLoader loader;
  private final ExpertResolver resolver;
  private final CachePolicy policy;
  private final int numExperts;
  private final LayerState[] layers;
  private final Arena arena = Arena.ofShared();
  private final ReentrantLock lock = new ReentrantLock();
  private final Condition stateChanged = lock.newCondition();

  private boolean closed;
  private int activeLoads;
  private long clock;
  private long hits;
  private long misses;
  private long bytesRead;
  private long evictions;
  private long waits;
  private long allocatedSlots;
  private long allocatedBytes;

  Gemma4ExpertCache(
      Gemma4ExpertLoader loader,
      int numLayers,
      int numExperts,
      int slotsPerLayer,
      ExpertResolver resolver,
      CachePolicy policy) {
    this.loader = Objects.requireNonNull(loader, "loader");
    this.resolver = Objects.requireNonNull(resolver, "resolver");
    this.policy = Objects.requireNonNull(policy, "policy");
    if (numLayers <= 0) {
      throw new IllegalArgumentException("numLayers must be > 0: " + numLayers);
    }
    if (numExperts <= 0) {
      throw new IllegalArgumentException("numExperts must be > 0: " + numExperts);
    }
    if (slotsPerLayer <= 0) {
      throw new IllegalArgumentException("slotsPerLayer must be > 0: " + slotsPerLayer);
    }
    this.numExperts = numExperts;
    this.layers = new LayerState[numLayers];
    for (int layer = 0; layer < numLayers; layer++) {
      ExpertWeights weights = requireResolved(layer, 0);
      this.layers[layer] = new LayerState(slotsPerLayer, numExperts, weights.totalBytes());
    }
  }

  Lease acquire(int layer, int expert) throws IOException {
    requireCoordinates(layer, expert);
    ExpertWeights weights = requireResolved(layer, expert);
    LayerState layerState = layers[layer];
    if (weights.totalBytes() != layerState.slotBytes) {
      throw new IllegalArgumentException(
          "Expert "
              + expert
              + " in layer "
              + layer
              + " requires "
              + weights.totalBytes()
              + " bytes; layer slots hold "
              + layerState.slotBytes);
    }

    Slot reserved;
    boolean replacesResidentExpert;
    lock.lock();
    try {
      while (true) {
        requireOpen();
        Slot hit = findHit(layerState, expert);
        if (hit != null) {
          recordUse(layerState, hit, expert);
          hit.leases++;
          hits++;
          return new Lease(hit, hit.loaded);
        }
        if (!isLoading(layerState, expert)) {
          reserved = selectEvictionSlot(layerState);
          if (reserved != null) {
            replacesResidentExpert = reserved.loaded != null;
            reserved.loaded = null;
            reserved.expert = -1;
            reserved.loading = true;
            reserved.loadingExpert = expert;
            activeLoads++;
            break;
          }
        }
        waits++;
        try {
          stateChanged.await();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          InterruptedIOException interrupted =
              new InterruptedIOException("Interrupted waiting for a Gemma 4 expert cache slot");
          interrupted.initCause(e);
          throw interrupted;
        }
      }
    } finally {
      lock.unlock();
    }

    try {
      MemorySegment storage = ensureStorage(reserved, layerState.slotBytes);
      LoadedExpert loaded = loader.loadInto(weights, storage);
      lock.lock();
      try {
        activeLoads--;
        if (closed) {
          clearReservation(reserved);
          stateChanged.signalAll();
          throw new IllegalStateException("Gemma 4 expert cache closed during a load");
        }
        reserved.loaded = loaded;
        reserved.expert = expert;
        reserved.loading = false;
        reserved.loadingExpert = -1;
        reserved.leases = 1;
        recordUse(layerState, reserved, expert);
        misses++;
        bytesRead = Math.addExact(bytesRead, weights.totalBytes());
        if (replacesResidentExpert) {
          evictions++;
        }
        stateChanged.signalAll();
        return new Lease(reserved, loaded);
      } finally {
        lock.unlock();
      }
    } catch (IOException | RuntimeException | Error failure) {
      rollbackLoad(reserved);
      throw failure;
    }
  }

  Stats stats() {
    lock.lock();
    try {
      return new Stats(hits, misses, bytesRead, evictions, waits);
    } finally {
      lock.unlock();
    }
  }

  long allocatedSlots() {
    lock.lock();
    try {
      return allocatedSlots;
    } finally {
      lock.unlock();
    }
  }

  long allocatedBytes() {
    lock.lock();
    try {
      return allocatedBytes;
    } finally {
      lock.unlock();
    }
  }

  private MemorySegment ensureStorage(Slot slot, long requiredBytes) {
    if (slot.storage != null) {
      return slot.storage;
    }
    MemorySegment allocated = arena.allocate(requiredBytes, 64);
    lock.lock();
    try {
      if (slot.storage == null) {
        slot.storage = allocated;
        allocatedSlots++;
        allocatedBytes = Math.addExact(allocatedBytes, requiredBytes);
        return allocated;
      }
      return slot.storage;
    } finally {
      lock.unlock();
    }
  }

  private Slot findHit(LayerState layer, int expert) {
    for (Slot slot : layer.slots) {
      if (!slot.loading && slot.expert == expert && slot.loaded != null) {
        return slot;
      }
    }
    return null;
  }

  private static boolean isLoading(LayerState layer, int expert) {
    for (Slot slot : layer.slots) {
      if (slot.loading && slot.loadingExpert == expert) {
        return true;
      }
    }
    return false;
  }

  private Slot selectEvictionSlot(LayerState layer) {
    Slot selected = null;
    for (Slot slot : layer.slots) {
      if (slot.loading || slot.leases != 0) {
        continue;
      }
      if (slot.loaded == null) {
        return slot;
      }
      if (selected == null || shouldEvictBefore(layer, slot, selected)) {
        selected = slot;
      }
    }
    return selected;
  }

  private boolean shouldEvictBefore(LayerState layer, Slot candidate, Slot current) {
    if (policy == CachePolicy.LFU) {
      int candidateCount = layer.expertUseCount[candidate.expert];
      int currentCount = layer.expertUseCount[current.expert];
      if (candidateCount != currentCount) {
        return candidateCount < currentCount;
      }
    }
    return candidate.lastUse < current.lastUse;
  }

  private void recordUse(LayerState layer, Slot slot, int expert) {
    if (layer.expertUseCount[expert] != Integer.MAX_VALUE) {
      layer.expertUseCount[expert]++;
    }
    slot.lastUse = ++clock;
  }

  private void release(Slot slot) {
    lock.lock();
    try {
      if (slot.leases <= 0) {
        throw new IllegalStateException("Gemma 4 expert slot has no live lease");
      }
      slot.leases--;
      stateChanged.signalAll();
    } finally {
      lock.unlock();
    }
  }

  private void rollbackLoad(Slot slot) {
    lock.lock();
    try {
      if (slot.loading) {
        activeLoads--;
        clearReservation(slot);
        stateChanged.signalAll();
      }
    } finally {
      lock.unlock();
    }
  }

  private static void clearReservation(Slot slot) {
    slot.loaded = null;
    slot.expert = -1;
    slot.loading = false;
    slot.loadingExpert = -1;
    slot.leases = 0;
  }

  private ExpertWeights requireResolved(int layer, int expert) {
    ExpertWeights weights =
        Objects.requireNonNull(resolver.resolve(layer, expert), "resolved expert weights");
    if (weights.layer() != layer || weights.expert() != expert) {
      throw new IllegalArgumentException(
          "Expert resolver returned layer="
              + weights.layer()
              + ", expert="
              + weights.expert()
              + " for layer="
              + layer
              + ", expert="
              + expert);
    }
    return weights;
  }

  private void requireCoordinates(int layer, int expert) {
    if (layer < 0 || layer >= layers.length) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
    if (expert < 0 || expert >= numExperts) {
      throw new IllegalArgumentException("expert out of range: " + expert);
    }
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("Gemma 4 expert cache is closed");
    }
  }

  @Override
  public void close() throws IOException {
    lock.lock();
    try {
      if (closed) {
        return;
      }
      closed = true;
      while (activeLoads != 0) {
        stateChanged.awaitUninterruptibly();
      }
      stateChanged.signalAll();
    } finally {
      lock.unlock();
    }

    IOException closeFailure = null;
    try {
      loader.close();
    } catch (IOException e) {
      closeFailure = e;
    }
    try {
      arena.close();
    } catch (RuntimeException e) {
      if (closeFailure != null) {
        closeFailure.addSuppressed(e);
      } else {
        throw e;
      }
    }
    if (closeFailure != null) {
      throw closeFailure;
    }
  }
}
