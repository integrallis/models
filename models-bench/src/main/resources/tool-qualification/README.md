# Needle 2 tool qualification

This gate runs the pure-Java CACT backend against all 13 examples published in Needle's
playground. The tool declarations and queries in `needle2-playground-v2.json` are copied verbatim
from the exact upstream revision recorded in that file. Expected calls are local assertions layered
on top of the unchanged upstream inputs. A fourteenth case reproduces the Spring AI zipcode report
and is a mandatory exact-match regression.

V2 corrects three local assertions in the superseded V1 suite: a URI is represented as a canonical
URI, the requested email subject is retained in full, and booking dates use ISO-8601 values. This
is a versioned contract correction, not a relaxed comparison; tool selection, schema validation,
and every expected value remain exact.

The `needle2-tool-conformance-v2` policy requires:

- a parseable `<tool_call>[...]</tool_call>` response for every case;
- the exact ordered tool selection for every case;
- schema-valid arguments containing no undeclared fields for every case;
- at least 90% agreement across the upstream expected argument values;
- a correct empty-array refusal for the off-topic case.

The Spring AI zipcode regression must independently pass every check, including the exact
`"88252"` argument. Aggregate quality cannot compensate for that regression failing.

Run the gate only against the pinned artifact bytes:

```shell
./gradlew :models-bench:run --args="needle2-tool-qualification \
  --model /path/to/needle2.cact \
  --models-revision $(git rev-parse HEAD) \
  --report benchmark-results/tool-calling/needle2-cact-pure-java.json"
```

The report includes the artifact and suite digests, source revisions, generation controls, every
raw response, per-case diagnostics, backend diagnostics, and the host/JVM environment. A failed
gate exits non-zero and is not eligible for a qualified ModelJar.

## Activated-adapter long-context retention

`activated-long-context-v1.json` is the frozen semantic gate for a base-aligned activated adapter.
Each of its eight cases puts a unique archive code near the beginning of an exact 4,096-token tool
prompt. The adapter must select the declared weather tool; after the fixed tool result, the exact
base branch must recover both the early code and the result value.

The gate requires physical KV sharing and the exact prefix length for every case, all eight tool
calls correct, at least six native base answers correct, every native-correct answer retained, and
byte-identical output from the native and shared base paths. Run it with the exact base and adapter
bytes:

```shell
./gradlew :models-bench:run --args="activated-long-context \
  --model /path/to/model.gguf \
  --adapter /path/to/activated-adapter \
  --models-revision $(git rev-parse HEAD) \
  --report benchmark-results/activated-long-context.json"
```
