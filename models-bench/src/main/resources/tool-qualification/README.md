# Needle 2 tool qualification

This gate runs the pure-Java CACT backend against all 13 examples published in Needle's
playground. The tool declarations and queries in `needle2-playground-v1.json` are copied verbatim
from the exact upstream revision recorded in that file. Expected calls are local assertions layered
on top of the unchanged upstream inputs. A fourteenth case reproduces the Spring AI zipcode report
and is a mandatory exact-match regression.

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
