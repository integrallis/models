import com.integrallis.models.backend.purejava.gguf.*;
import java.lang.foreign.Arena;
import java.nio.file.*;
import java.util.*;
public class Vocab {
  public static void main(String[] a) throws Exception {
    try (Arena arena = Arena.ofConfined()) {
      GgufMetadata m = GgufParser.parse(Path.of(a[0]), arena).metadata();
      var toks = m.getStringArray("tokenizer.ggml.tokens").orElse(List.of());
      var types = m.getInt32Array("tokenizer.ggml.token_type").orElse(List.of());
      for (int i=Integer.parseInt(a[1]); i<=Integer.parseInt(a[2]); i++) System.out.print(i+"="+toks.get(i)+"/t"+types.get(i)+"  ");
      System.out.println();
    }
  }
}
