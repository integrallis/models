# GLiNER schema classification

`RustGlinerClassificationModel` loads an exact GLiNER2.5-Decide Safetensors snapshot
and classifies text against caller-supplied labels. It does not generate text.
The snapshot directory must contain the model weights, both configuration files,
and tokenizer files. ModelJars installs and verifies those files when an exact
snapshot has been admitted to its classification catalog.

```java
import com.integrallis.models.api.SchemaClassificationModel;
import com.integrallis.models.backend.nativekernel.RustGlinerClassificationModel;
import java.nio.file.Path;
import java.util.List;

var task = new SchemaClassificationModel.Task(
    "intent",
    List.of(
        new SchemaClassificationModel.Label("refund request"),
        new SchemaClassificationModel.Label("shipping question"),
        new SchemaClassificationModel.Label("other")),
    false,
    0.5);

try (var model = RustGlinerClassificationModel.load(Path.of(args[0]))) {
    var results = model.classify(new SchemaClassificationModel.Request(
        "My package never arrived. Where is it?", List.of(task)));
    for (var result : results) {
        for (var label : result.labels()) {
            if (label.selected()) System.out.println(label.label());
        }
    }
}
```

Run with Java 25 or newer, `--add-modules jdk.incubator.vector` and
`--enable-native-access=ALL-UNNAMED`. The native library comes from the Models
platform artifact. `-Dmodels.deberta.threads=8` selects the thread budget measured
in the current CPU evidence. The library owns its worker pool and releases it on
`close()`; the tokenizer and Java encoder resources have the same lifetime.

A single-label task selects one label. A multi-label task selects labels at the
specified threshold; if none meet it, the publisher-compatible decoder selects
the highest-scoring label and sets `thresholdFallback`. Scores are model outputs,
not a claim of calibrated confidence. Multiple tasks are returned in request
order. Label descriptions can be supplied through the two-argument `Label`
constructor. The encoded text plus schema must fit the model's input limit.

The runtime uses prepared native F32 matrices and attention kernels with Java
schema processing and encoder orchestration. There is one GLiNER execution path.
The shared DeBERTa encoder also supports existing F16 rerankers, whose regression
checks remain independent of GLiNER admission.

Runtime support does not grant catalog qualification. The exact snapshot, runtime,
thread budget, quality cohorts and performance repetitions are bound in the
qualification evidence. See
[the measured performance screening](https://github.com/integrallis/models/blob/c316cd80/benchmark-results/2026-11-01-gliner25-decide/gliner25-eval-20261101T220219Z/README.md)
and [upstream emotion baseline context](https://github.com/integrallis/models/blob/c316cd80/benchmark-results/2026-11-01-gliner25-decide/EMOTION-BASELINE-CONTEXT.md).
