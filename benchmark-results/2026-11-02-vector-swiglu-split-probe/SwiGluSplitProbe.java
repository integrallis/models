///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector
//DEPS com.integrallis:vectors-core:0.1.32

import com.integrallis.vectors.core.VectorUtil;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Serial VectorUtil.swiGlu against the same call split over a fixed pool, the split Models uses:
 * workers = min(cap, size / threshold), chunks aligned to 64 floats, the caller takes chunk 0.
 * Reports the best of 9 timed blocks per (size, workers) after warm-up.
 */
public class SwiGluSplitProbe {
  static final int CPUS = Runtime.getRuntime().availableProcessors();
  static final ExecutorService POOL = Executors.newFixedThreadPool(Math.max(1, CPUS - 1), r -> { Thread t = new Thread(r); t.setDaemon(true); return t; });

  static void split(float[] o, float[] g, float[] u, int size, int workers) throws Exception {
    if (workers < 2) { VectorUtil.swiGlu(o, 0, g, 0, u, 0, size); return; }
    int chunk = (size + workers - 1) / workers; chunk = ((chunk + 63) / 64) * 64;
    List<Future<?>> pending = new ArrayList<>();
    for (int w = 1; w < workers; w++) {
      int from = w * chunk; if (from >= size) break; int to = Math.min(size, from + chunk);
      pending.add(POOL.submit(() -> VectorUtil.swiGlu(o, from, g, from, u, from, to - from)));
    }
    VectorUtil.swiGlu(o, 0, g, 0, u, 0, Math.min(size, chunk));
    for (Future<?> f : pending) f.get();
  }

  public static void main(String[] args) throws Exception {
    System.out.println("cpus " + CPUS + ", java " + System.getProperty("java.version"));
    int[] sizes = {1024, 2048, 4096, 8192, 14336, 18944, 32768, 65536};
    int[] workerCounts = {1, 2, 4, Math.max(1, CPUS - 1), CPUS};
    for (int size : sizes) {
      float[] g = new float[size], u = new float[size], o = new float[size];
      for (int i = 0; i < size; i++) { g[i] = (i % 97) * 0.1f - 4f; u[i] = (i % 13) * 0.2f; }
      StringBuilder line = new StringBuilder(String.format("size %6d", size));
      double serial = 0;
      for (int workers : workerCounts) {
        int reps = Math.max(50, 4_000_000 / size);
        for (int w = 0; w < reps; w++) split(o, g, u, size, workers);
        long best = Long.MAX_VALUE;
        for (int r = 0; r < 9; r++) {
          long t = System.nanoTime();
          for (int k = 0; k < reps; k++) split(o, g, u, size, workers);
          best = Math.min(best, System.nanoTime() - t);
        }
        double us = best / 1e3 / reps;
        if (workers == 1) serial = us;
        line.append(String.format("  w%-2d %8.2f us (x%.2f)", workers, us, serial / us));
      }
      System.out.println(line);
    }
  }
}
