import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;

/** Q5_1 projection: Java kernel against the Rust shim, at the shapes the artifacts actually use. */
public final class Q51Bench {
  public static void main(String[] args) throws Exception {
    Path lib = Path.of(args[0]);
    // all-MiniLM-L6-v2 Q5_K_S: attn_q/k/v/output are 384x384 Q5_1, ffn_up is 1536x384 Q5_1.
    int[][] shapes = {{384, 384}, {1536, 384}};
    int[] batches = {8, 64, 256, 512};
    try (Arena arena = Arena.ofConfined();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(lib)) {
      System.out.printf("%-14s %-6s %12s %12s %12s %12s %8s%n", "shape", "batch", "java min", "java mean", "shim min", "shim mean", "x");
      for (int[] shape : shapes) {
        int rows = shape[0];
        int cols = shape[1];
        MemorySegment weights = arena.allocate((long) rows * cols / 32L * 24L);
        fill(weights, rows, cols);
        for (int batch : batches) {
          float[] input = new float[batch * cols];
          for (int i = 0; i < input.length; i++) {
            input[i] = ((i * 17) % 29 - 14) * 0.0625f;
          }
          float[] javaOut = new float[batch * rows];
          float[] shimOut = new float[batch * rows];
          float[] row = new float[cols];
          int reps = 500;
          // Warm up both arms: this is HotSpot, so an un-warmed Java arm measures the interpreter.
          for (int i = 0; i < 500; i++) {
            TensorOps.ggufExactBatchedMatmul(
                javaOut, input, weights, GgufTensorType.Q5_1, batch, rows, cols, row);
            kernel.multiply(shimOut, input, weights, GgufTensorType.Q5_1, batch, rows, cols);
          }
          double[] java2 = time(() -> TensorOps.ggufExactBatchedMatmul(
              javaOut, input, weights, GgufTensorType.Q5_1, batch, rows, cols, row), reps);
          double[] shim2 = time(() -> kernel.multiply(
              shimOut, input, weights, GgufTensorType.Q5_1, batch, rows, cols), reps);
          double javaMs = java2[0];
          double shimMs = shim2[0];
          for (int i = 0; i < javaOut.length; i++) {
            if (javaOut[i] != shimOut[i]) {
              throw new AssertionError("arms disagree at " + i + ": " + javaOut[i] + " vs " + shimOut[i]);
            }
          }
          System.out.printf("%-14s %-6d %12.4f %12.4f %12.4f %12.4f %8.2f%n",
              rows + "x" + cols, batch, java2[0], java2[1], shim2[0], shim2[1], java2[0] / shim2[0]);
        }
      }
    }
  }

  /** Returns {min, mean} milliseconds per call over several trials. */
  private static double[] time(Runnable task, int reps) {
    int trials = 9;
    long best = Long.MAX_VALUE;
    long total = 0;
    for (int trial = 0; trial < trials; trial++) {
      long start = System.nanoTime();
      for (int i = 0; i < reps; i++) {
        task.run();
      }
      long elapsed = System.nanoTime() - start;
      best = Math.min(best, elapsed);
      total += elapsed;
    }
    return new double[] {best / 1e6 / reps, total / 1e6 / (double) reps / trials};
  }

  private static void fill(MemorySegment weights, int rows, int cols) {
    int blocksPerRow = cols / 32;
    for (int row = 0; row < rows; row++) {
      for (int block = 0; block < blocksPerRow; block++) {
        long base = (long) (row * blocksPerRow + block) * 24L;
        int index = row * blocksPerRow + block;
        weights.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), base,
            (short) (0x3000 | ((index * 37) & 0x03FF)));
        weights.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), base + 2,
            (short) (((index % 2) == 0 ? 0 : 0x8000) | 0x3000 | ((index * 53) & 0x03FF)));
        weights.set(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), base + 4,
            index * 0x9E3779B9);
        for (int n = 0; n < 16; n++) {
          weights.set(ValueLayout.JAVA_BYTE, base + 8 + n, (byte) ((index * 13 + n * 29) & 0xFF));
        }
      }
    }
  }
}
