import com.integrallis.models.backend.purejava.gguf.*;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.*;
import java.util.*;
public class EogProbe {
  public static void main(String[] a) throws Exception {
    for (String p : a) {
      try (Arena arena = Arena.ofConfined()) {
        GgufFile f = GgufParser.parse(Path.of(p), arena);
        GgufMetadata m = f.metadata();
        var toks = m.getStringArray("tokenizer.ggml.tokens").orElse(List.of());
        if (toks.isEmpty()) { System.out.println(p+": no vocab"); continue; }
        GgufTokenizer t = GgufTokenizer.fromMetadata(m);
        StringBuilder sb = new StringBuilder();
        for (int id : t.endOfGenerationTokenIds()) sb.append(id).append('=').append(toks.get(id)).append(' ');
        var types = m.getInt32Array("tokenizer.ggml.token_type").orElse(List.of());
        System.out.println(Path.of(p).getFileName()+" arch="+m.getString("general.architecture").orElse("")+" model="+m.getString("tokenizer.ggml.model").orElse("")+" vocab="+toks.size()+" eos="+m.getUint32("tokenizer.ggml.eos_token_id").orElse(-1)+" eot="+m.getUint32("tokenizer.ggml.eot_token_id").orElse(-1));
        System.out.println("  EOG: "+sb);
        String ct = m.getString("tokenizer.chat_template").orElse(null);
        System.out.println("  template: " + (ct==null? "none" : ct.length()+" chars"));
        if (System.getProperty("id")!=null) { int id=Integer.getInteger("id"); if (id<toks.size()) System.out.println("  id "+id+"='"+toks.get(id)+"' type="+(id<types.size()?types.get(id):null)); }
        if (System.getProperty("dump")!=null && ct!=null) System.out.println(ct);
      } catch (Exception e) { System.out.println(p+": "+e); }
    }
  }
}
