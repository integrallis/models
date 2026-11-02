import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Random;

/**
 * Splits the Q4_K decode gap into kernel/memory throughput versus per-matmul orchestration cost by
 * streaming the SAME number of weight bytes through one large native call and then through many
 * small ones. Identical bytes, identical kernel, different invocation counts: the delta is
 * orchestration.
 */
public final class Q4KThroughputProbe {
  private static final int QK_K = 256;
  private static final int BLOCK_BYTES = 144;

  public static void main(String[] args) {
    Path library = Path.of(System.getProperty("models.native.kernels.library"));
    int onlyThreads = Integer.parseInt(args[0]);
    int cols = 4096;
    long targetBytes = 2L << 30; // 2 GiB: far beyond any L3 slice, so every read misses
    int blocksPerRow = cols / QK_K;
    long rowBytes = (long) blocksPerRow * BLOCK_BYTES;
    int rows = (int) (targetBytes / rowBytes);
    long bytes = rows * rowBytes;

    try (Arena arena = Arena.ofShared()) {
      System.out.printf("weights=%.2f GiB rows=%d cols=%d%n", bytes / 1024.0 / 1024 / 1024, rows, cols);

      MemorySegment weights = arena.allocate(bytes, 64);
      Random random = new Random(20261101L);
      // Valid Q4_K blocks: d (fp16), dmin (fp16), 12 B of packed 6-bit scales, 128 B of nibbles.
      // Random bytes would yield NaN/denormal fp16 scales and measure the FPU's garbage handling
      // instead of the kernel.
      short d = Float.floatToFloat16(0.0125f);
      short dmin = Float.floatToFloat16(0.0021f);
      byte[] block = new byte[BLOCK_BYTES];
      for (long offset = 0; offset < bytes; offset += BLOCK_BYTES) {
        random.nextBytes(block);
        block[0] = (byte) (d & 0xFF);
        block[1] = (byte) ((d >>> 8) & 0xFF);
        block[2] = (byte) (dmin & 0xFF);
        block[3] = (byte) ((dmin >>> 8) & 0xFF);
        for (int i = 4; i < 16; i++) {
          block[i] = (byte) (block[i] & 0x3F); // 6-bit scale/min fields
        }
        MemorySegment.copy(block, 0, weights, java.lang.foreign.ValueLayout.JAVA_BYTE, offset, BLOCK_BYTES);
      }
      float[] input = new float[cols];
      for (int i = 0; i < cols; i++) {
        input[i] = (float) random.nextGaussian();
      }
      float[] out = new float[rows];
      int calls = 224;
      int rowsPerCall = rows / calls;
      long sliceBytes = (long) rowsPerCall * rowBytes;
      float[] sliceOut = new float[rowsPerCall];
      double checksum = 0;

      for (int threads : new int[] {onlyThreads}) {
        try (RustGgufBatchedMatrixKernel kernel =
            RustGgufBatchedMatrixKernel.open(library, threads)) {
          if (true) {
            System.out.printf("impl=%s supportsQ4_K=%s eligible(batch=1)=%s%n",
                kernel.implementation(), kernel.supports(GgufTensorType.Q4_K),
                kernel.isEligible(GgufTensorType.Q4_K, 1, rows, cols));
          }
          // Arm A: one call over all rows -- one orchestration event for the whole 2 GiB.
          double bestA = 0;
          for (int pass = 0; pass < 3; pass++) {
            long start = System.nanoTime();
            kernel.multiply(out, input, weights, GgufTensorType.Q4_K, 1, rows, cols);
            double gbs = bytes / ((System.nanoTime() - start) / 1e9) / 1e9;
            bestA = Math.max(bestA, gbs);
          }
          // Arm B: the same bytes as 224 calls -- one llama-7B token's worth of matmuls.
          double bestB = 0;
          for (int pass = 0; pass < 3; pass++) {
            long start = System.nanoTime();
            for (int call = 0; call < calls; call++) {
              MemorySegment slice = weights.asSlice((long) call * sliceBytes, sliceBytes);
              kernel.multiply(sliceOut, input, slice, GgufTensorType.Q4_K, 1, rowsPerCall, cols);
            }
            double gbs = sliceBytes * calls / ((System.nanoTime() - start) / 1e9) / 1e9;
            bestB = Math.max(bestB, gbs);
          }
          System.out.printf(
              "threads=%2d  A one-call: %6.1f GB/s   B %d-calls: %6.1f GB/s   orchestration cost: %4.1f%%%n",
              threads, bestA, calls, bestB, 100.0 * (bestA - bestB) / bestA);
          checksum += out[0] + sliceOut[0];
        }
      }
      System.out.printf("checksum=%.4f%n", checksum);
    }
  }
}
