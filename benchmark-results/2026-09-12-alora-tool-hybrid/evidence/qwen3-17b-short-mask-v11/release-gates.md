# V11 release gates

This checklist is controlling. A green unit suite or a favorable timing result cannot substitute for
any pending row. The candidate is rejected when a fixed correctness gate fails.

| Gate | Required evidence | Status |
| --- | --- | --- |
| Frozen training | Completed adapter and training manifest match the preflight inputs and hashes | Running |
| Exposed smoke | Fixed 25 simple, 25 multiple, and 25 irrelevance cases: 100% syntax, 100% schema, at least 85% exact, no worse than base, at most 5% false calls | Pending |
| Sealed qualification | The precommitted 100/100/100 manifest passes the same gates without changing its cases, evaluator, or thresholds | Pending; unopened |
| Runtime package | Packaging validates the complete training provenance, exact Qwen3 base GGUF, tokenizer hashes, adapter bytes, license, and notice | Pending |
| Projection oracle | Every real adapter projection agrees with the independently generated NumPy oracle within the frozen tolerance | Pending |
| JVM mechanics | Real Qwen3 weights prove one immutable physical KV prefix, exact base continuation, repeated turns, and disabled-adapter no-op behavior | Pending |
| JVM tool behavior | The existing 14-case suite, zipcode regression, argument threshold, abstention, six-turn conversation, and result synthesis all pass | Pending |
| Framework adapters | Real Spring AI 1.1.4, 1.1.8, and 2.0 plus LangChain4j 1.0.0, 1.13.1, and 1.17.2 execute the tool loop and natural-language result synthesis | Pending |
| Long context | All eight 4,096-token cases share physical storage, retain every native-correct fact, preserve exact base output, and call the correct tool | Pending |
| Performance and memory | Shared and independently recomputed arms are token-exact at 256/1,024/4,096; sharing wins by at least 20% at 4K; crossover, adapter/KV bytes, JVM/native memory, and peak RSS are complete | Pending |
| Models artifact | Released artifacts resolve on a clean host and repeat the applicable JVM and framework gates | Pending |
| ModelJars component | Hidden adapter metadata binds every file plus the exact qualified base hash/size and immutable Models evidence revision | Pending |
| ModelJars composition | The facade opens the activated runtime—not two independent models—and published coordinates pass Java, Spring AI, and LangChain4j clean-host tests | Pending |
| Production | Catalog, site, CLI refresh, artifact resolution, and hybrid count are verified from public endpoints | Pending |
| Infrastructure | Exact temporary instances are deleted, local watchdogs unloaded, and provider-wide inventory is zero | Pending |

The exposed smoke continuation may start automatically after successful training. It cannot start
the sealed qualification; that transition requires inspecting and preserving the exposed report.
