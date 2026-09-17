import com.integrallis.models.backend.purejava.gguf.*;
import java.lang.foreign.Arena;
import java.nio.file.*;
public class ParseAll {
  public static void main(String[] a) throws Exception {
    int ok=0, bad=0;
    for (String p : a) {
      try (Arena arena = Arena.ofConfined()) {
        GgufFile f = GgufParser.parse(Path.of(p), arena);
        long align = f.metadata().getUint32("general.alignment").map(Integer::toUnsignedLong).orElse(32L);
        ok++; System.out.println("OK   " + p + " tensors=" + f.tensorInfos().size() + " alignment=" + align);
      } catch (Exception e) { bad++; System.out.println("FAIL " + p + " : " + e.getMessage()); }
    }
    System.out.println("accepted=" + ok + " rejected=" + bad);
  }
}
