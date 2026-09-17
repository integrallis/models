import com.integrallis.models.backend.purejava.gguf.*;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.*;
import java.util.*;
public class RoundTrip {
  public static void main(String[] a) throws Exception {
    String text = "Use <s>old price</s> for strikethrough.";
    for (String p : a) try (Arena arena = Arena.ofConfined()) {
      GgufTokenizer t = GgufTokenizer.fromMetadata(GgufParser.parse(Path.of(p), arena).metadata());
      int[] ids = t.encode(text);
      System.out.println(Path.of(p).getFileName()+" ids="+Arrays.toString(ids)+"\n  decoded='"+t.decode(ids)+"' roundtrip="+t.decode(ids).equals(text));
    }
  }
}
