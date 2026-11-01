//JAVA 25+
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector
//RUNTIME_OPTIONS --enable-native-access=ALL-UNNAMED
//DEPS org.modeljars:modeljars:0.1.71
//DEPS org.modeljars.huggingface:walkingcat.soprano-1.1-80m-gguf.q8_0:1.1.0-q8_0.1

import com.integrallis.models.api.WavEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.modeljars.ModelJars;

/**
 * Synthesizes a sentence list through the RELEASED Soprano route and writes one wav per sentence
 * plus a manifest. This exists to validate a round-trip-WER judge pipeline against a model whose
 * waveform parity is already proven, before that pipeline is pointed at anything new.
 */
class SopranoCorpus {
  private static final String MODEL =
      "org.modeljars.huggingface:walkingcat.soprano-1.1-80m-gguf.q8_0:1.1.0-q8_0.1";

  public static void main(String... args) throws Exception {
    Logger.getLogger("org.modeljars").setLevel(Level.WARNING);
    Logger.getLogger("com.integrallis").setLevel(Level.WARNING);
    Path sentences = Path.of(args.length > 0 ? args[0] : "sentences.txt");
    Path outputDir = Path.of(args.length > 1 ? args[1] : "wav");
    Files.createDirectories(outputDir);
    List<String> lines = Files.readAllLines(sentences, StandardCharsets.UTF_8).stream()
        .map(String::strip).filter(line -> !line.isEmpty()).toList();

    List<String> manifest = new ArrayList<>();
    manifest.add("index\twav\taudioMillis\tsynthesisMillis\trealTimeFactor\ttext");
    try (var model = ModelJars.openSpeech(MODEL)) {
      model.synthesize("warm up");                     // one warm-up, not recorded
      for (int index = 0; index < lines.size(); index++) {
        String text = lines.get(index);
        long started = System.nanoTime();
        var audio = model.synthesize(text);
        double synthesisMillis = (System.nanoTime() - started) / 1_000_000.0;
        Path wav = outputDir.resolve("%02d.wav".formatted(index));
        Files.write(wav, WavEncoder.pcm16(audio));
        double audioMillis = audio.duration().toNanos() / 1_000_000.0;
        manifest.add("%d\t%s\t%.0f\t%.1f\t%.4f\t%s".formatted(
            index, wav.getFileName(), audioMillis, synthesisMillis,
            synthesisMillis / audioMillis, text));
        System.out.printf("%02d  %6.0f ms audio  RTF %.3f  %s%n",
            index, audioMillis, synthesisMillis / audioMillis, text);
      }
    }
    Files.write(outputDir.resolve("manifest.tsv"),
        String.join(System.lineSeparator(), manifest).getBytes(StandardCharsets.UTF_8));
    System.out.println("manifest -> " + outputDir.resolve("manifest.tsv").toAbsolutePath());
  }
}
