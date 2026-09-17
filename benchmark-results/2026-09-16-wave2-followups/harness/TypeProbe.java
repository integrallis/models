import com.integrallis.models.backend.purejava.gguf.*;
import java.lang.foreign.Arena;
import java.nio.file.*;
import java.util.*;
public class TypeProbe {
  static final Set<String> T = Set.of("<|eot_id|>","<|im_end|>","<|end|>","<|return|>","<|call|>","<|flush|>","<|calls|>","<end_of_turn>","<|endoftext|>","</s>","<|eom_id|>","<EOT>","_<EOT>","[EOT]","[EOS]","<|end_of_text|>","<end_of_utterance>","<eos>","<turn|>","<|tool_response>","<｜end▁of▁sentence｜>");
  public static void main(String[] a) throws Exception {
    for (String p : a) {
      try (Arena arena = Arena.ofConfined()) {
        GgufMetadata m = GgufParser.parse(Path.of(p), arena).metadata();
        var toks = m.getStringArray("tokenizer.ggml.tokens").orElse(List.of());
        var types = m.getInt32Array("tokenizer.ggml.token_type").orElse(List.of());
        Set<Integer> declared = new TreeSet<>();
        for (String k : List.of("tokenizer.ggml.eos_token_id","tokenizer.ggml.eot_token_id","tokenizer.ggml.eom_token_id")) m.getUint32(k).ifPresent(declared::add);
        StringBuilder sb = new StringBuilder();
        for (int i=0;i<toks.size();i++) if (T.contains(toks.get(i))) sb.append(i).append('=').append(toks.get(i)).append("/t").append(i<types.size()?types.get(i):"-").append(declared.contains(i)?"/decl":"").append(' ');
        System.out.println(Path.of(p).getFileName()+" model="+m.getString("tokenizer.ggml.model").orElse("")+" pre="+m.getString("tokenizer.ggml.pre").orElse("")+" : "+sb);
      } catch (Exception e) { System.out.println(p+": "+e); }
    }
  }
}
