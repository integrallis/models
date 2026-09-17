import com.integrallis.models.backend.purejava.gguf.*;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.*;
import java.util.*;
public class EogProv {
  public static void main(String[] a) throws Exception {
    boolean prov = Boolean.getBoolean("prov");
    for (String p : a) try (Arena arena = Arena.ofConfined()) {
      GgufMetadata m = GgufParser.parse(Path.of(p), arena).metadata();
      var toks = m.getStringArray("tokenizer.ggml.tokens").orElse(List.of());
      if (toks.isEmpty()) continue;
      long t0 = System.nanoTime();
      GgufTokenizer t = GgufTokenizer.fromMetadata(m);
      long ms = (System.nanoTime()-t0)/1_000_000;
      StringBuilder sb = new StringBuilder();
      for (int id : t.endOfGenerationTokenIds()) sb.append(id).append('=').append(toks.get(id)).append(' ');
      if (!prov) { System.out.println(Path.of(p).getFileName()+" : "+sb); continue; }
      System.out.println(Path.of(p).getFileName()+" | template="+t.chatTemplateEndOfTurnResolution()+" | build-ms="+ms);
      t.endOfGenerationSources().forEach((id, src) -> System.out.println("    "+id+"="+toks.get(id)+" <- "+src));
    }
  }
}
