import com.integrallis.models.backend.purejava.gguf.*;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.*;
import java.util.*;
public class EogAll {
  public static void main(String[] a) throws Exception {
    for (String p : a) try (Arena arena = Arena.ofConfined()) {
      GgufMetadata m = GgufParser.parse(Path.of(p), arena).metadata();
      var toks = m.getStringArray("tokenizer.ggml.tokens").orElse(List.of());
      if (toks.isEmpty()) continue;
      StringBuilder sb = new StringBuilder();
      for (int id : GgufTokenizer.fromMetadata(m).endOfGenerationTokenIds()) sb.append(id).append('=').append(toks.get(id)).append(' ');
      System.out.println(Path.of(p).getFileName()+" : "+sb);
    }
  }
}
