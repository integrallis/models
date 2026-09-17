import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsFile;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsParser;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;

/** Parses every listed GGUF / Safetensors file with the size-asserting parsers. */
public class LoaderScan {
  public static void main(String[] args) throws Exception {
    int accepted = 0, rejected = 0;
    for (String line : Files.readAllLines(Path.of(args[0]))) {
      if (line.isBlank()) continue;
      Path path = Path.of(line.trim());
      try (Arena arena = Arena.ofConfined()) {
        long start = System.nanoTime();
        String summary;
        if (line.endsWith(".gguf")) {
          GgufFile file = GgufParser.parse(path, arena);
          summary = "tensors=" + file.tensorInfos().size();
        } else {
          SafetensorsFile file = SafetensorsParser.parse(path, arena);
          summary = "tensors=" + file.tensorNames().size();
        }
        accepted++;
        System.out.printf("ACCEPT %s %s bytes=%d parse-ms=%.1f%n", path, summary, Files.size(path),
            (System.nanoTime() - start) / 1e6);
      } catch (RuntimeException failure) {
        rejected++;
        System.out.printf("REJECT %s %s: %s%n", path, failure.getClass().getSimpleName(), failure.getMessage());
      }
    }
    System.out.println("SUMMARY accepted=" + accepted + " rejected=" + rejected);
  }
}
