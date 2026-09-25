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
package com.integrallis.models.backend.purejava.gguf;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.channels.FileChannel;
import java.util.Locale;

/**
 * Loads GGUF weights into memory the translation buffer can actually cover.
 *
 * <p>A mapped GGUF is file backed, which on Linux means 4 KiB pages and nothing the kernel will
 * promote. Qwen3.5-4B at Q4_K_M is 2.6 GiB of weights, so that is about 650,000 page table entries
 * against an L2 TLB that holds roughly two thousand. A forward pass walks every weight exactly once
 * and never revisits one, so essentially every new block pays a page walk on top of its load.
 *
 * <p>MEASURED 2026-09-24 on a dedicated CCX33 (EPYC Milan, 4 physical cores at 2400 MHz), the same
 * bytes behind 2 MiB pages instead of 4 KiB ones:
 *
 * <ul>
 *   <li>matmul over every {@code blk.*} weight once, in layer order: 0.4033 s to <b>0.3796 s</b>,
 *       159.1 to 169.3 G-MAC/s, +5.9% over four rounds
 *   <li>whole forward pass, 18 tokens: 0.4935 s to <b>0.4700 s</b>, <b>+4.8%</b>, five of six
 *       alternating rounds won
 * </ul>
 *
 * <p>Not one instruction changed between those arms, so the result is bit-identical by
 * construction.
 *
 * <p><b>And it does not carry to a real decision, which is why this is off by default.</b> The same
 * build measured over ten alternating rounds of the video-shape workload -- fifteen decisions, a
 * fresh document each -- came out at 1.3837 s per decision on 4 KiB pages against 1.3689 s on
 * 2 MiB, which is +1.1% on five rounds won out of ten and per-round deltas that alternate sign.
 * That is a coin flip, and {@code /proc/meminfo} confirmed 2,674,688 kB of {@code AnonHugePages}
 * during those runs, so the arm genuinely ran: this is no effect rather than no data.
 *
 * <p>The untested explanation for the split is arithmetic intensity. A decision prefills a whole
 * document, so it runs at a far wider batch than the 18 tokens above, and each weight byte then
 * feeds many more multiply-accumulates -- at batch 18 Q4_K spends 144 bytes per 256 weights for
 * about 32 MAC per byte, and a long prefill is an order of magnitude better than that. Load latency
 * is what huge pages hide, and a compute-bound prefill has less of it to hide. Untested, so treat
 * it as a hypothesis and not a result.
 *
 * <p>Left in, off, because it is measured, bit-exact and cheap to switch on for a workload that is
 * prefill-narrow rather than prefill-wide. Row tiling the Q4_K kernel, by contrast, measured +9.1%
 * on an isolated matmul and +0.4% on a forward pass and was deleted.
 *
 * <p>The cost is one copy at load and the weights held twice while it runs -- once in the page
 * cache, once here. The page cache copy is evictable the moment the copy finishes.
 *
 * <p>Linux only. Transparent huge pages are a Linux facility and {@code MADV_HUGEPAGE} is a Linux
 * advice value; macOS wants a flag at map time instead and is left on the mapped path. If the
 * system is configured with {@code transparent_hugepage/enabled=never} the advice is accepted and
 * ignored, which costs the copy and buys nothing, so this does not default on.
 */
public final class GgufHugePages {

  /**
   * Selects the huge page load path: {@code true}, {@code false}, or {@code auto}.
   *
   * <p>Defaults to {@code auto}, which is <b>off</b>: measured, it buys nothing on a real decision.
   * {@code true} forces the copy on Linux x86-64, which is worth measuring for a workload that
   * prefills narrowly rather than in wide batches. It stays a property rather than becoming a
   * default because the win also depends on a kernel setting this code cannot see --
   * {@code transparent_hugepage/enabled} must be {@code always} or {@code madvise} -- and because
   * a caller who is memory constrained would rather have the mapping back.
   */
  public static final String HUGE_PAGES_PROPERTY = "models.purejava.hugePages";

  /** Linux {@code MADV_HUGEPAGE}, from {@code asm-generic/mman-common.h}. */
  private static final int MADV_HUGEPAGE = 14;

  /** {@code PROT_READ | PROT_WRITE}. */
  private static final int PROT_READ_WRITE = 0x1 | 0x2;

  /** {@code MAP_PRIVATE | MAP_ANONYMOUS} on Linux. */
  private static final int MAP_PRIVATE_ANONYMOUS = 0x02 | 0x20;

  /** The x86-64 huge page. Allocating aligned to it is what lets the kernel promote at all. */
  private static final long HUGE_PAGE_BYTES = 2L * 1024 * 1024;

  /**
   * Below this a copy cannot pay for itself: the TLB covers a small model without help.
   *
   * <p>Sized at 64 MiB, which is 32 huge pages and far more than the L2 TLB's reach in 4 KiB pages.
   */
  private static final long MINIMUM_WORTHWHILE_BYTES = 64L * 1024 * 1024;

  private static volatile boolean lastAdviceAccepted;

  private GgufHugePages() {}

  /**
   * Whether the most recent {@link #copyInto} had its huge page advice accepted by the kernel.
   *
   * <p>Exists so a benchmark can assert that the arm it believes it is running is the arm that
   * ran. {@code /proc/meminfo}'s {@code AnonHugePages} is the independent check.
   */
  public static boolean lastAdviceAccepted() {
    return lastAdviceAccepted;
  }

  /**
   * Whether {@code fileSize} bytes should be copied into huge-page-backed memory.
   *
   * @param fileSize the size of the GGUF about to be loaded
   */
  public static boolean isRequested(long fileSize) {
    String requested =
        System.getProperty(HUGE_PAGES_PROPERTY, "auto").trim().toLowerCase(Locale.ROOT);
    if (requested.equals("false")) {
      return false;
    }
    if (fileSize < MINIMUM_WORTHWHILE_BYTES) {
      // An explicit "true" still loses to arithmetic here, and silently doing the slow thing
      // because someone set a flag is worse than ignoring the flag.
      return false;
    }
    if (requested.equals("true")) {
      return isLinuxX64();
    }
    // "auto" is off. See the class javadoc: +3.7% on a bare 18-token forward pass, and nothing at
    // all on the decision path this library exists to serve.
    return false;
  }

  /**
   * Copies the whole file into an anonymous, huge-page-aligned segment owned by {@code arena}.
   *
   * <p>Falls back to the mapping on any failure. There is no configuration in which failing to
   * allocate 2.6 GiB of anonymous memory should stop a model from loading when the mapping that
   * already works is one line away.
   *
   * @param channel the open GGUF, positioned anywhere
   * @param fileSize the full size of the file
   * @param arena the arena that will own the returned segment
   * @return a segment holding the file's bytes, huge page backed when the kernel agreed
   */
  @SuppressWarnings("restricted")
  public static MemorySegment copyInto(FileChannel channel, long fileSize, Arena arena)
      throws IOException {
    // The region is mapped here rather than taken from the arena because Arena.allocate returns
    // ZERO-INITIALISED memory: it faults every page in as 4 KiB before this code can advise
    // anything, and advising afterwards only leaves the pages for khugepaged to maybe collapse
    // later. MEASURED 2026-09-24, that route produced 16 MiB of huge pages out of 2.6 GiB and no
    // speedup at all, while the identical advice on a fresh anonymous mapping produced 2.67 GiB.
    // An anonymous mapping is already zero, so nothing is lost by skipping the arena's zeroing.
    MemorySegment destination = mapAnonymous(fileSize);
    if (destination.equals(MemorySegment.NULL)) {
      return channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, arena);
    }

    // Advise before the pages are faulted in, which is the whole point of mapping them here.
    lastAdviceAccepted = advise(destination, fileSize);

    // Hand ownership to the arena so the mapping dies with it.
    destination = destination.reinterpret(fileSize, arena, segment -> unmap(segment, fileSize));

    // Copy through a mapping rather than a read loop so the kernel moves the bytes. The mapping
    // is confined to this call: once the bytes are in the arena's segment nothing refers to it.
    try (Arena mapping = Arena.ofConfined()) {
      MemorySegment source = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, mapping);
      MemorySegment.copy(source, 0, destination, 0, fileSize);
    }
    return destination;
  }

  /**
   * Maps {@code length} bytes of fresh anonymous memory, or {@link MemorySegment#NULL} on failure.
   *
   * <p>Deliberately not {@code Arena.allocate}: see the comment at the call site.
   */
  @SuppressWarnings("restricted")
  private static MemorySegment mapAnonymous(long length) {
    if (!isLinuxX64()) {
      return MemorySegment.NULL;
    }
    try {
      Linker linker = Linker.nativeLinker();
      MemorySegment symbol = linker.defaultLookup().find("mmap").orElse(MemorySegment.NULL);
      if (symbol.equals(MemorySegment.NULL)) {
        return MemorySegment.NULL;
      }
      MethodHandle mmap =
          linker.downcallHandle(
              symbol,
              FunctionDescriptor.of(
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_INT,
                  ValueLayout.JAVA_INT,
                  ValueLayout.JAVA_INT,
                  ValueLayout.JAVA_LONG));
      MemorySegment result =
          (MemorySegment)
              mmap.invokeExact(
                  MemorySegment.NULL,
                  length,
                  PROT_READ_WRITE,
                  MAP_PRIVATE_ANONYMOUS,
                  -1,
                  0L);
      // mmap reports failure as MAP_FAILED, which is -1 and not null.
      if (result.address() == -1L || result.address() == 0L) {
        return MemorySegment.NULL;
      }
      return result;
    } catch (Throwable unavailable) {
      return MemorySegment.NULL;
    }
  }

  @SuppressWarnings("restricted")
  private static void unmap(MemorySegment segment, long length) {
    try {
      Linker linker = Linker.nativeLinker();
      MemorySegment symbol = linker.defaultLookup().find("munmap").orElse(MemorySegment.NULL);
      if (symbol.equals(MemorySegment.NULL)) {
        return;
      }
      MethodHandle munmap =
          linker.downcallHandle(
              symbol,
              FunctionDescriptor.of(
                  ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
      int result = (int) munmap.invokeExact(segment, length);
      assert result == 0 : "munmap failed";
    } catch (Throwable ignored) {
      // The process is either exiting or leaking one mapping; neither is worth failing a close.
    }
  }

  private static boolean isLinuxX64() {
    String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
    String architecture = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
    return os.contains("linux") && (architecture.equals("amd64") || architecture.equals("x86_64"));
  }

  /**
   * Asks the kernel to back {@code segment} with huge pages.
   *
   * <p>Returns whether {@code madvise} was reached and accepted, rather than succeeding silently.
   * An advisory call is exactly the kind of switch that can quietly do nothing -- the first draft
   * of this method discarded {@code invokeExact}'s result in statement position, which makes it a
   * {@code void} call against an {@code int} descriptor, throws {@link
   * java.lang.invoke.WrongMethodTypeException} on every invocation, and leaves the catch below to
   * swallow it. The measurement would then have compared 4 KiB pages against 4 KiB pages.
   *
   * @return true when the kernel accepted the advice
   */
  @SuppressWarnings("restricted")
  private static boolean advise(MemorySegment segment, long length) {
    if (!isLinuxX64()) {
      return false;
    }
    try {
      Linker linker = Linker.nativeLinker();
      MemorySegment symbol = linker.defaultLookup().find("madvise").orElse(MemorySegment.NULL);
      if (symbol.equals(MemorySegment.NULL)) {
        return false;
      }
      MethodHandle madvise =
          linker.downcallHandle(
              symbol,
              FunctionDescriptor.of(
                  ValueLayout.JAVA_INT,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_INT));
      // Bind the result. EINVAL means the kernel was built without transparent huge pages, which
      // is a reason to get 4 KiB pages and not a reason to fail the load.
      int result = (int) madvise.invokeExact(segment, length, MADV_HUGEPAGE);
      return result == 0;
    } catch (Throwable unavailable) {
      // An advisory call that did not happen leaves correct, slower memory.
      return false;
    }
  }
}
