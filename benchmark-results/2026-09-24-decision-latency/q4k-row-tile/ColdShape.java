package demo;

import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The same matmuls a forward pass does, in the order it does them, each weight touched once.
 *
 * <p>PerShape times one exemplar tensor per shape, eight warmups and ten rounds deep, so by the
 * time it is timed that tensor is cache resident. A forward pass never sees a weight twice: it
 * streams every one of them once. If a kernel change helps the first and not the second, the
 * forward pass is not bound by what the change improved.
 */
public final class ColdShape {
  public static void main(String... args) throws Exception {
    int batch = Integer.getInteger("diag.tokens", 18);
    int rounds = Integer.getInteger("diag.rounds", 3);
    try (Arena arena = Arena.ofShared()) {
      var file = GgufParser.parse(java.nio.file.Path.of(args[0]), arena);
      List<String> names = new ArrayList<>();
      for (var info : file.tensorInfos()) {
        long[] shape = info.shape();
        if (shape.length == 2
            && info.name().startsWith("blk.")
            && shape[0] * shape[1] >= 1_000_000L) {
          names.add(info.name());
        }
      }
      names.sort((a, b) -> {
        int la = Integer.parseInt(a.split("\\.")[1]);
        int lb = Integer.parseInt(b.split("\\.")[1]);
        return la != lb ? Integer.compare(la, lb) : a.compareTo(b);
      });

      try (RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.openBundled()) {
        Random random = new Random(7L);
        float[] scratch = new float[batch * 12288];
        for (int index = 0; index < scratch.length; index++) {
          scratch[index] = (float) (random.nextGaussian() * 0.35);
        }
        float[] out = new float[batch * 12288];
        // Warm the code path, not the weights: one pass over the first layer only.
        for (int warm = 0; warm < 3; warm++) {
          for (String name : names.subList(0, Math.min(8, names.size()))) {
            var t = file.getTensor(name);
            kernel.multiply(out, scratch, t.dataSegment(), t.type(), batch,
                (int) t.shape()[1], (int) t.shape()[0]);
          }
        }
        double best = Double.MAX_VALUE;
        double macs = 0;
        for (int round = 0; round < rounds; round++) {
          double total = 0;
          macs = 0;
          for (String name : names) {
            var t = file.getTensor(name);
            int cols = (int) t.shape()[0];
            int rows = (int) t.shape()[1];
            MemorySegment weights = t.dataSegment();
            long start = System.nanoTime();
            kernel.multiply(out, scratch, weights, t.type(), batch, rows, cols);
            total += (System.nanoTime() - start) / 1e9;
            macs += (double) batch * rows * cols;
          }
          best = Math.min(best, total);
        }
        System.out.printf(
            "  every blk 2-D weight once, batch %d: %.4f s  (%.1f G-MAC/s over %d tensors)%n",
            batch, best, macs / best / 1e9, names.size());
      }
    }
  }
}
