import com.integrallis.models.backend.purejava.gguf.*;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.*;
public class BuildTime {
  public static void main(String[] a) throws Exception {
    for (String p : a) try (Arena arena = Arena.ofConfined()) {
      GgufMetadata m = GgufParser.parse(Path.of(p), arena).metadata();
      long best = Long.MAX_VALUE;
      for (int i=0;i<8;i++){ long t0=System.nanoTime(); GgufTokenizer.fromMetadata(m); best=Math.min(best,System.nanoTime()-t0);} 
      System.out.println(Path.of(p).getFileName()+" best-of-8 ms="+best/1_000_000);
    }
  }
}
