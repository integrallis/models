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

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.Optional;

/**
 * Panama FFM binding to the CUDA driver API, following {@code NativeKernelLibrary}'s shape.
 *
 * <p>The driver API, not the runtime API: {@code libcuda.so} ships with the NVIDIA display driver
 * and needs no CUDA toolkit installed, and it is the layer that loads PTX directly. Using it keeps
 * the policy boundary intact — there is no vendor math library here, no inference runtime, and no
 * third-party Rust CUDA crate. Java owns the host side outright, exactly as {@code backend-native}
 * states for the CPU shim.
 *
 * <p><strong>Weights are uploaded from the mapped GGUF with no host-side copy.</strong> {@link
 * #copyToDevice} takes a {@link MemorySegment}, and for a mapped tensor that segment's address is
 * passed straight to {@code cuMemcpyHtoD}. This is a concrete advantage over the TornadoVM arm,
 * where {@code ByteArray.fromSegment} allocates a fresh off-heap segment and copies into it — about
 * 33 GB of host memory for a 26B model before a byte reaches the device, per the analysis on {@code
 * feat/tornado-large-model-plan}. Here that term is zero.
 *
 * <p><strong>Everything fails closed.</strong> {@link #open} returns an empty {@link Optional} with
 * a stated reason when the driver is missing, too old, or reports no device, rather than throwing.
 * A host with no NVIDIA driver must cost nothing at all (gate G3).
 */
@SuppressWarnings("restricted")
public final class CudaDriver implements AutoCloseable {

  /** Compute capability below which this backend refuses a device. */
  public static final int MINIMUM_COMPUTE_CAPABILITY = 80;

  private static final int CUDA_SUCCESS = 0;
  private static final int ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
  private static final int ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
  private static final int DEVICE_NAME_BYTES = 256;

  private static final ValueLayout.OfInt C_INT = ValueLayout.JAVA_INT;
  private static final ValueLayout.OfLong C_LONG = ValueLayout.JAVA_LONG;
  private static final java.lang.foreign.AddressLayout C_POINTER = ValueLayout.ADDRESS;

  private final Arena arena;
  private final MethodHandle cuGetErrorString;
  private final MethodHandle cuDeviceGetName;
  private final MethodHandle cuDeviceGetAttribute;
  private final MethodHandle cuDeviceTotalMem;
  private final MethodHandle cuModuleLoadData;
  private final MethodHandle cuModuleGetFunction;
  private final MethodHandle cuMemAlloc;
  private final MethodHandle cuMemFree;
  private final MethodHandle cuMemcpyHtoD;
  private final MethodHandle cuMemcpyDtoH;
  private final MethodHandle cuLaunchKernel;
  private final MethodHandle cuCtxSynchronize;
  private final MethodHandle cuDevicePrimaryCtxRelease;

  private final int driverVersion;
  private final int device;
  private final String deviceName;
  private final int computeCapability;
  private final long totalMemoryBytes;
  private final MemorySegment context;
  private boolean closed;
  private int releaseStatus = CUDA_SUCCESS;

  private CudaDriver(
      Arena arena,
      MethodHandle cuGetErrorString,
      MethodHandle cuDeviceGetName,
      MethodHandle cuDeviceGetAttribute,
      MethodHandle cuDeviceTotalMem,
      MethodHandle cuModuleLoadData,
      MethodHandle cuModuleGetFunction,
      MethodHandle cuMemAlloc,
      MethodHandle cuMemFree,
      MethodHandle cuMemcpyHtoD,
      MethodHandle cuMemcpyDtoH,
      MethodHandle cuLaunchKernel,
      MethodHandle cuCtxSynchronize,
      MethodHandle cuDevicePrimaryCtxRelease,
      int driverVersion,
      int device,
      String deviceName,
      int computeCapability,
      long totalMemoryBytes,
      MemorySegment context) {
    this.arena = arena;
    this.cuGetErrorString = cuGetErrorString;
    this.cuDeviceGetName = cuDeviceGetName;
    this.cuDeviceGetAttribute = cuDeviceGetAttribute;
    this.cuDeviceTotalMem = cuDeviceTotalMem;
    this.cuModuleLoadData = cuModuleLoadData;
    this.cuModuleGetFunction = cuModuleGetFunction;
    this.cuMemAlloc = cuMemAlloc;
    this.cuMemFree = cuMemFree;
    this.cuMemcpyHtoD = cuMemcpyHtoD;
    this.cuMemcpyDtoH = cuMemcpyDtoH;
    this.cuLaunchKernel = cuLaunchKernel;
    this.cuCtxSynchronize = cuCtxSynchronize;
    this.cuDevicePrimaryCtxRelease = cuDevicePrimaryCtxRelease;
    this.driverVersion = driverVersion;
    this.device = device;
    this.deviceName = deviceName;
    this.computeCapability = computeCapability;
    this.totalMemoryBytes = totalMemoryBytes;
    this.context = context;
  }

  /**
   * Opens the first eligible CUDA device, or explains why none is usable.
   *
   * <p>Never throws for an absent or unusable device: an empty result with a reason is the fallback
   * path, and the caller is expected to take it silently.
   */
  public static Result open() {
    return open(0);
  }

  /** Opens CUDA device {@code ordinal}, or explains why it is not usable. */
  public static Result open(int ordinal) {
    Arena arena = Arena.ofShared();
    try {
      SymbolLookup lookup;
      try {
        lookup = SymbolLookup.libraryLookup(driverLibraryName(), arena);
      } catch (IllegalArgumentException | UnsatisfiedLinkError absent) {
        arena.close();
        return Result.unavailable(
            "no CUDA driver library (" + driverLibraryName() + ") on this host");
      }

      Linker linker = Linker.nativeLinker();
      MethodHandle cuInit = downcall(lookup, linker, "cuInit", FunctionDescriptor.of(C_INT, C_INT));
      MethodHandle cuDriverGetVersion =
          downcall(lookup, linker, "cuDriverGetVersion", FunctionDescriptor.of(C_INT, C_POINTER));
      MethodHandle cuDeviceGetCount =
          downcall(lookup, linker, "cuDeviceGetCount", FunctionDescriptor.of(C_INT, C_POINTER));
      MethodHandle cuDeviceGet =
          downcall(lookup, linker, "cuDeviceGet", FunctionDescriptor.of(C_INT, C_POINTER, C_INT));
      MethodHandle cuDeviceGetName =
          downcall(
              lookup,
              linker,
              "cuDeviceGetName",
              FunctionDescriptor.of(C_INT, C_POINTER, C_INT, C_INT));
      MethodHandle cuDeviceGetAttribute =
          downcall(
              lookup,
              linker,
              "cuDeviceGetAttribute",
              FunctionDescriptor.of(C_INT, C_POINTER, C_INT, C_INT));
      MethodHandle cuDeviceTotalMem =
          downcall(
              lookup,
              linker,
              "cuDeviceTotalMem_v2",
              FunctionDescriptor.of(C_INT, C_POINTER, C_INT));
      MethodHandle cuDevicePrimaryCtxRetain =
          downcall(
              lookup,
              linker,
              "cuDevicePrimaryCtxRetain",
              FunctionDescriptor.of(C_INT, C_POINTER, C_INT));
      MethodHandle cuDevicePrimaryCtxRelease =
          downcall(
              lookup, linker, "cuDevicePrimaryCtxRelease_v2", FunctionDescriptor.of(C_INT, C_INT));
      MethodHandle cuCtxSetCurrent =
          downcall(lookup, linker, "cuCtxSetCurrent", FunctionDescriptor.of(C_INT, C_POINTER));
      MethodHandle cuGetErrorString =
          downcall(
              lookup, linker, "cuGetErrorString", FunctionDescriptor.of(C_INT, C_INT, C_POINTER));

      int status = (int) cuInit.invokeExact(0);
      if (status != CUDA_SUCCESS) {
        arena.close();
        return Result.unavailable("cuInit failed with CUDA status " + status);
      }

      MemorySegment scratch = arena.allocate(C_LONG, 4);
      status = (int) cuDriverGetVersion.invokeExact(scratch);
      if (status != CUDA_SUCCESS) {
        arena.close();
        return Result.unavailable("cuDriverGetVersion failed with CUDA status " + status);
      }
      int driverVersion = scratch.get(C_INT, 0);

      status = (int) cuDeviceGetCount.invokeExact(scratch);
      if (status != CUDA_SUCCESS) {
        arena.close();
        return Result.unavailable("cuDeviceGetCount failed with CUDA status " + status);
      }
      int deviceCount = scratch.get(C_INT, 0);
      if (deviceCount <= ordinal) {
        arena.close();
        return Result.unavailable(
            "CUDA reports " + deviceCount + " devices; device " + ordinal + " was requested");
      }

      status = (int) cuDeviceGet.invokeExact(scratch, ordinal);
      if (status != CUDA_SUCCESS) {
        arena.close();
        return Result.unavailable("cuDeviceGet failed with CUDA status " + status);
      }
      int device = scratch.get(C_INT, 0);

      status =
          (int)
              cuDeviceGetAttribute.invokeExact(scratch, ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
      if (status != CUDA_SUCCESS) {
        arena.close();
        return Result.unavailable("cuDeviceGetAttribute(major) failed with status " + status);
      }
      int major = scratch.get(C_INT, 0);
      status =
          (int)
              cuDeviceGetAttribute.invokeExact(scratch, ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
      if (status != CUDA_SUCCESS) {
        arena.close();
        return Result.unavailable("cuDeviceGetAttribute(minor) failed with status " + status);
      }
      int minor = scratch.get(C_INT, 0);
      int computeCapability = major * 10 + minor;
      if (computeCapability < MINIMUM_COMPUTE_CAPABILITY) {
        arena.close();
        return Result.unavailable(
            "device compute capability "
                + major
                + "."
                + minor
                + " is below the required "
                + (MINIMUM_COMPUTE_CAPABILITY / 10)
                + "."
                + (MINIMUM_COMPUTE_CAPABILITY % 10));
      }

      MemorySegment nameBuffer = arena.allocate(DEVICE_NAME_BYTES);
      status = (int) cuDeviceGetName.invokeExact(nameBuffer, DEVICE_NAME_BYTES, device);
      String deviceName =
          status == CUDA_SUCCESS ? nameBuffer.getString(0, StandardCharsets.UTF_8) : "unknown";

      status = (int) cuDeviceTotalMem.invokeExact(scratch, device);
      if (status != CUDA_SUCCESS) {
        arena.close();
        return Result.unavailable("cuDeviceTotalMem failed with CUDA status " + status);
      }
      long totalMemoryBytes = scratch.get(C_LONG, 0);

      MemorySegment contextHolder = arena.allocate(C_POINTER);
      status = (int) cuDevicePrimaryCtxRetain.invokeExact(contextHolder, device);
      if (status != CUDA_SUCCESS) {
        arena.close();
        return Result.unavailable("cuDevicePrimaryCtxRetain failed with status " + status);
      }
      MemorySegment context = contextHolder.get(C_POINTER, 0);
      status = (int) cuCtxSetCurrent.invokeExact(context);
      if (status != CUDA_SUCCESS) {
        arena.close();
        return Result.unavailable("cuCtxSetCurrent failed with CUDA status " + status);
      }

      return Result.available(
          new CudaDriver(
              arena,
              cuGetErrorString,
              cuDeviceGetName,
              cuDeviceGetAttribute,
              cuDeviceTotalMem,
              downcall(
                  lookup,
                  linker,
                  "cuModuleLoadData",
                  FunctionDescriptor.of(C_INT, C_POINTER, C_POINTER)),
              downcall(
                  lookup,
                  linker,
                  "cuModuleGetFunction",
                  FunctionDescriptor.of(C_INT, C_POINTER, C_POINTER, C_POINTER)),
              downcall(
                  lookup, linker, "cuMemAlloc_v2", FunctionDescriptor.of(C_INT, C_POINTER, C_LONG)),
              downcall(lookup, linker, "cuMemFree_v2", FunctionDescriptor.of(C_INT, C_LONG)),
              downcall(
                  lookup,
                  linker,
                  "cuMemcpyHtoD_v2",
                  FunctionDescriptor.of(C_INT, C_LONG, C_POINTER, C_LONG)),
              downcall(
                  lookup,
                  linker,
                  "cuMemcpyDtoH_v2",
                  FunctionDescriptor.of(C_INT, C_POINTER, C_LONG, C_LONG)),
              downcall(
                  lookup,
                  linker,
                  "cuLaunchKernel",
                  FunctionDescriptor.of(
                      C_INT, C_POINTER, C_INT, C_INT, C_INT, C_INT, C_INT, C_INT, C_INT, C_POINTER,
                      C_POINTER, C_POINTER)),
              downcall(lookup, linker, "cuCtxSynchronize", FunctionDescriptor.of(C_INT)),
              cuDevicePrimaryCtxRelease,
              driverVersion,
              device,
              deviceName,
              computeCapability,
              totalMemoryBytes,
              context));
    } catch (RuntimeException | LinkageError failure) {
      arena.close();
      return Result.unavailable("CUDA driver binding failed: " + failure.getMessage());
    } catch (Throwable failure) {
      arena.close();
      return Result.unavailable("CUDA driver call failed: " + failure);
    }
  }

  /** The platform filename of the CUDA driver library. */
  static String driverLibraryName() {
    String os = System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT);
    if (os.contains("win")) {
      return "nvcuda.dll";
    }
    // Linux ships the driver as libcuda.so.1; libcuda.so is only present with the toolkit.
    return "libcuda.so.1";
  }

  /** The driver version reported by {@code cuDriverGetVersion}. */
  public int driverVersion() {
    return driverVersion;
  }

  /** The device's marketing name. */
  public String deviceName() {
    return deviceName;
  }

  /** The device's compute capability as {@code major * 10 + minor}. */
  public int computeCapability() {
    return computeCapability;
  }

  /** The device's total global memory in bytes, as the driver reports it. */
  public long totalMemoryBytes() {
    return totalMemoryBytes;
  }

  /** Loads a PTX module and returns its handle. */
  public MemorySegment loadModule(byte[] ptx) {
    Objects.requireNonNull(ptx, "ptx");
    try (Arena temporary = Arena.ofConfined()) {
      // cuModuleLoadData reads a NUL-terminated image.
      MemorySegment image = temporary.allocate(ptx.length + 1L);
      MemorySegment.copy(ptx, 0, image, ValueLayout.JAVA_BYTE, 0, ptx.length);
      image.set(ValueLayout.JAVA_BYTE, ptx.length, (byte) 0);
      MemorySegment holder = arena.allocate(C_POINTER);
      check((int) cuModuleLoadData.invokeExact(holder, image), "cuModuleLoadData");
      return holder.get(C_POINTER, 0);
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw new IllegalStateException("cuModuleLoadData failed", failure);
    }
  }

  /** Resolves a kernel entry point in a loaded module. */
  public MemorySegment function(MemorySegment module, String name) {
    try (Arena temporary = Arena.ofConfined()) {
      MemorySegment holder = arena.allocate(C_POINTER);
      MemorySegment symbol = temporary.allocateFrom(name);
      check((int) cuModuleGetFunction.invokeExact(holder, module, symbol), "cuModuleGetFunction");
      return holder.get(C_POINTER, 0);
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw new IllegalStateException("cuModuleGetFunction(" + name + ") failed", failure);
    }
  }

  /** Allocates {@code bytes} of device memory and returns the device pointer. */
  public long allocate(long bytes) {
    try (Arena temporary = Arena.ofConfined()) {
      MemorySegment holder = temporary.allocate(C_LONG);
      check((int) cuMemAlloc.invokeExact(holder, bytes), "cuMemAlloc");
      return holder.get(C_LONG, 0);
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw new IllegalStateException("cuMemAlloc(" + bytes + ") failed", failure);
    }
  }

  /** Frees a device allocation. */
  public void free(long devicePointer) {
    try {
      check((int) cuMemFree.invokeExact(devicePointer), "cuMemFree");
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw new IllegalStateException("cuMemFree failed", failure);
    }
  }

  /**
   * Copies {@code bytes} from {@code source} to device memory.
   *
   * <p>For a mapped GGUF tensor the segment address goes straight to the driver, so no host-side
   * duplicate of the weights is ever created.
   */
  public void copyToDevice(long devicePointer, MemorySegment source, long bytes) {
    try {
      check((int) cuMemcpyHtoD.invokeExact(devicePointer, source, bytes), "cuMemcpyHtoD");
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw new IllegalStateException("cuMemcpyHtoD failed", failure);
    }
  }

  /** Copies {@code bytes} from device memory into {@code destination}. */
  public void copyToHost(MemorySegment destination, long devicePointer, long bytes) {
    try {
      check((int) cuMemcpyDtoH.invokeExact(destination, devicePointer, bytes), "cuMemcpyDtoH");
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw new IllegalStateException("cuMemcpyDtoH failed", failure);
    }
  }

  /**
   * Launches {@code function} with {@code gridBlocks} blocks of {@code blockThreads} threads.
   *
   * <p>{@code parameters} is the {@code void**} argument array the driver expects: one pointer per
   * kernel parameter, each addressing that parameter's value.
   */
  public void launch(
      MemorySegment function, int gridBlocks, int blockThreads, MemorySegment parameters) {
    launch(function, gridBlocks, 1, blockThreads, parameters);
  }

  /**
   * Launches {@code function} over a two-dimensional grid of {@code gridBlocksX} by {@code
   * gridBlocksY} blocks.
   *
   * <p>The second dimension is the batch row. It exists because passing the batch index as a scalar
   * forced one launch per row: a five-item JevBench run on an A40 issued 54,306 launches for 890
   * projections, which is the term that made the device path slower than the CPU one.
   */
  public void launch(
      MemorySegment function,
      int gridBlocksX,
      int gridBlocksY,
      int blockThreads,
      MemorySegment parameters) {
    try {
      check(
          (int)
              cuLaunchKernel.invokeExact(
                  function,
                  gridBlocksX,
                  gridBlocksY,
                  1,
                  blockThreads,
                  1,
                  1,
                  0,
                  MemorySegment.NULL,
                  parameters,
                  MemorySegment.NULL),
          "cuLaunchKernel");
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw new IllegalStateException("cuLaunchKernel failed", failure);
    }
  }

  /** Blocks until every queued operation on the context has completed. */
  public void synchronize() {
    try {
      check((int) cuCtxSynchronize.invokeExact(), "cuCtxSynchronize");
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw new IllegalStateException("cuCtxSynchronize failed", failure);
    }
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    try {
      // Best effort: the process is tearing the context down and has nowhere to report to, so
      // the status is retained for a caller that asks rather than thrown or swallowed.
      releaseStatus = (int) cuDevicePrimaryCtxRelease.invokeExact(device);
    } catch (Throwable failure) {
      releaseStatus = -1;
    } finally {
      arena.close();
    }
  }

  /**
   * The status {@code cuDevicePrimaryCtxRelease} returned, or {@code -1} if the call itself failed;
   * {@code 0} until {@link #close()} runs.
   *
   * <p>Retained rather than logged so a bench run can report a dirty teardown alongside its numbers
   * instead of leaving it invisible.
   */
  public int releaseStatus() {
    return releaseStatus;
  }

  private void check(int status, String operation) {
    if (status == CUDA_SUCCESS) {
      return;
    }
    throw new IllegalStateException(operation + " failed: " + describe(status));
  }

  private String describe(int status) {
    try (Arena temporary = Arena.ofConfined()) {
      MemorySegment holder = temporary.allocate(C_POINTER);
      int outcome = (int) cuGetErrorString.invokeExact(status, holder);
      if (outcome != CUDA_SUCCESS) {
        return "CUDA status " + status;
      }
      MemorySegment message = holder.get(C_POINTER, 0);
      if (message.address() == 0) {
        return "CUDA status " + status;
      }
      return message.reinterpret(Long.MAX_VALUE).getString(0, StandardCharsets.UTF_8)
          + " (status "
          + status
          + ")";
    } catch (Throwable failure) {
      return "CUDA status " + status;
    }
  }

  private static MethodHandle downcall(
      SymbolLookup lookup, Linker linker, String name, FunctionDescriptor descriptor) {
    MemorySegment symbol =
        lookup
            .find(name)
            .orElseThrow(() -> new UnsatisfiedLinkError("CUDA driver does not export " + name));
    return linker.downcallHandle(symbol, descriptor);
  }

  /**
   * The outcome of trying to open a device: either a usable driver, or a reason it is not usable.
   *
   * <p>Modelled as a value rather than an exception because "no GPU here" is the ordinary case on
   * most hosts, and the fallback must be silent and free.
   */
  public record Result(Optional<CudaDriver> driver, String reason) {

    /** A usable driver. */
    static Result available(CudaDriver driver) {
      return new Result(Optional.of(driver), "");
    }

    /** No usable device, with an operator-readable explanation. */
    static Result unavailable(String reason) {
      return new Result(Optional.empty(), reason);
    }

    /** Whether a device is usable. */
    public boolean isAvailable() {
      return driver.isPresent();
    }
  }
}
