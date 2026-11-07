import com.integrallis.mfcqi.metrics.CognitiveComplexity;
import com.integrallis.mfcqi.smells.JavaSmellDetector;
import com.integrallis.mfcqi.core.JavaSourceFiles;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ParserConfiguration;
import java.nio.file.Path;

// Exposes the same MFCQI 0.7.0 analyzers' details, which metrics-only CLI JSON omits.
class ExtractHotspots {
  public static void main(String[] args) throws Exception {
    if (args[0].equals("modern")) {
      StaticJavaParser.getParserConfiguration().setLanguageLevel(ParserConfiguration.LanguageLevel.BLEEDING_EDGE);
    }
    for (int i = 1; i < args.length; i++) {
      var path = Path.of(args[i]);
      for (var file : JavaSourceFiles.findAll(path)) {
        try { StaticJavaParser.parse(file); }
        catch (RuntimeException e) {
          System.out.printf("PARSE\t%s\t%s\t%s%n", path, file, clean(e.getMessage()));
        }
      }
      for (var f : new CognitiveComplexity().extract(path).functions()) {
        System.out.printf("COGNITIVE\t%s\t%s\t%d\t%s\t%d%n", path, f.filePath(), f.lineNumber(), f.functionName(), f.complexity());
      }
      for (var s : new JavaSmellDetector().detect(path)) {
        System.out.printf("SMELL\t%s\t%s\t%s\t%s\t%s%n", path, s.location(), s.id(), s.severity(), clean(s.description()));
      }
    }
  }
  static String clean(String s) { return s.replace('\t', ' ').replace('\n', ' ').replace('\r', ' '); }
}
