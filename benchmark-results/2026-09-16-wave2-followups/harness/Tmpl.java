import com.integrallis.models.backend.purejava.gguf.*;
import java.lang.foreign.Arena;
import java.nio.file.*;
public class Tmpl {
  public static void main(String[] a) throws Exception {
    Files.createDirectories(Path.of(a[0]));
    for (int i=1;i<a.length;i++) try (Arena arena = Arena.ofConfined()) {
      GgufMetadata m = GgufParser.parse(Path.of(a[i]), arena).metadata();
      String ct = m.getString("tokenizer.chat_template").orElse(null);
      if (ct != null) Files.writeString(Path.of(a[0], Path.of(a[i]).getFileName() + ".jinja"), ct);
    }
  }
}
