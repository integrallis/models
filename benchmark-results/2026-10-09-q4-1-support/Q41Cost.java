import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/** What the exact Q4_1 path costs against the integer Q4_0 kernel at the same shape. */
public final class Q41Cost {
  public static void main(String[] args) {
    int[][] shapes = {{1536, 1536}, {8960, 1536}};
    int[] batches = {1, 8};
    try (Arena arena = Arena.ofConfined()) {
      System.out.printf("%-12s %-6s %12s %12s %10s%n", "shape", "batch", "Q4_0 ms", "Q4_1 ms", "ratio");
      for (int[] shape : shapes) {
        int rows = shape[0];
        int cols = shape[1];
        MemorySegment q40 = arena.allocate((long) rows * cols / 32L * 18L);
        MemorySegment q41 = arena.allocate((long) rows * cols / 32L * 20L);
        fill40(q40, rows, cols);
        fill41(q41, rows, cols);
        for (int batch : batches) {
          float[] x = new float[batch * cols];
          for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 17) % 29 - 14) * 0.0625f;
          }
          float[] out40 = new float[batch * rows];
          float[] out41 = new float[batch * rows];
          byte[] qa = new byte[batch * cols];
          float[] qs = new float[batch * (cols + 31) / 32];
          int[] zp = new int[batch * (cols + 3) / 4];
          short[] sums = new short[batch * (cols + 15) / 16];
          float[] lanes = new float[batch * rows * 8];
          Runnable a = () -> TensorOps.ggufBatchedMatmul(out40, x, q40, GgufTensorType.Q4_0,
              batch, rows, cols, qa, qs, zp, sums, lanes, GgufQ4Kernel.WIDENED);
          Runnable b = () -> TensorOps.ggufBatchedMatmul(out41, x, q41, GgufTensorType.Q4_1,
              batch, rows, cols, qa, qs, zp, sums, lanes, GgufQ4Kernel.WIDENED);
          for (int i = 0; i < 30; i++) { a.run(); b.run(); }
          double t40 = time(a), t41 = time(b);
          System.out.printf("%-12s %-6d %12.4f %12.4f %9.2fx%n",
              rows + "x" + cols, batch, t40, t41, t41 / t40);
        }
      }
    }
  }

  private static double time(Runnable task) {
    long best = Long.MAX_VALUE;
    for (int trial = 0; trial < 7; trial++) {
      long start = System.nanoTime();
      for (int i = 0; i < 20; i++) task.run();
      best = Math.min(best, System.nanoTime() - start);
    }
    return best / 1e6 / 20;
  }

  private static void fill40(MemorySegment w, int rows, int cols) {
    int blocks = rows * cols / 32;
    for (int b = 0; b < blocks; b++) {
      long base = (long) b * 18L;
      w.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), base, (short) 0x3000);
      for (int i = 0; i < 16; i++) w.set(ValueLayout.JAVA_BYTE, base + 2 + i, (byte) ((b * 13 + i) & 0xFF));
    }
  }

  private static void fill41(MemorySegment w, int rows, int cols) {
    int blocks = rows * cols / 32;
    for (int b = 0; b < blocks; b++) {
      long base = (long) b * 20L;
      w.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), base, (short) 0x3000);
      w.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), base + 2, (short) 0x3400);
      for (int i = 0; i < 16; i++) w.set(ValueLayout.JAVA_BYTE, base + 4 + i, (byte) ((b * 13 + i) & 0xFF));
    }
  }
}
