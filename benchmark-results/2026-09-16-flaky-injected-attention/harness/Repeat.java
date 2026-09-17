import org.junit.platform.launcher.*;
import org.junit.platform.launcher.core.*;
import org.junit.platform.launcher.listeners.*;
import static org.junit.platform.engine.discovery.DiscoverySelectors.*;
public class Repeat {
  public static void main(String[] a) {
    int n = Integer.parseInt(a[0]);
    String sel = a.length > 1 ? a[1] : "com.integrallis.models.backend.purejava.llama.LlamaForwardPassTest$NanoModel#injectedAttentionKernelPreservesPrefillAndFallsBackForDecode";
    Launcher launcher = LauncherFactory.create();
    long fails = 0, total = 0;
    for (int i = 0; i < n; i++) {
      var req = LauncherDiscoveryRequestBuilder.request().selectors(sel.contains("#") ? selectMethod(sel) : selectClass(sel)).build();
      var l = new SummaryGeneratingListener();
      launcher.execute(req, l);
      var s = l.getSummary();
      total += s.getTestsStartedCount(); fails += s.getTotalFailureCount();
      if (s.getTotalFailureCount() > 0 && fails <= 3) s.getFailures().forEach(f -> System.out.println("FAIL iter " + " " + f.getTestIdentifier().getDisplayName() + ": " + String.valueOf(f.getException().getMessage()).lines().limit(3).toList()));
    }
    System.out.println("iterations=" + n + " testsRun=" + total + " failures=" + fails);
  }
}
