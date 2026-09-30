package demo;

import com.integrallis.models.backend.purejava.gguf.GgufHugePages;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;

/** Does the huge page load path actually run, and does the kernel actually honour it? */
public final class HugeCheck {
  public static void main(String... args) throws Exception {
    Path model = Path.of(args[0]);
    long size = Files.size(model);
    System.out.printf("  property        = %s%n", System.getProperty("models.purejava.hugePages"));
    System.out.printf("  isRequested(%d) = %s%n", size, GgufHugePages.isRequested(size));
    try (Arena arena = Arena.ofShared()) {
      long start = System.nanoTime();
      var file = GgufParser.parse(model, arena);
      double seconds = (System.nanoTime() - start) / 1e9;
      System.out.printf("  parse           = %.3f s, %d tensors%n", seconds, file.tensorInfos().size());
      System.out.printf("  adviceAccepted  = %s%n", GgufHugePages.lastAdviceAccepted());
      for (String line : Files.readAllLines(Path.of("/proc/meminfo"))) {
        if (line.startsWith("AnonHugePages") || line.startsWith("ShmemHugePages")) {
          System.out.printf("  %s%n", line.trim());
        }
      }
      for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
        if (line.startsWith("VmRSS") || line.startsWith("VmHWM")) {
          System.out.printf("  %s%n", line.trim());
        }
      }
    }
  }
}
