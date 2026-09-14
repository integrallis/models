# V20 Java hybrid-generation result — rejected

The frozen V20 development gate finished all 75 already-exposed cases on the local Intel Mac. The
process completed and wrote `development.json`; the observer connection ended after case 56, but
the report contains all 75 case records. This is a **failed development gate**, not independent
qualification. The V11 sealed 300-case window was not opened, and no hybrid ModelJars artifact or
Models release is authorized by this run.

| Frozen gate | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Correct applicable-call decisions | 48/50 | 47/50 | fail |
| Correct irrelevant/no-call decisions | 24/25 | 24/25 | pass |
| Exact generated tool calls | 43/50 | 43/50 | pass |
| Strict tool syntax | 75/75 | 75/75 | pass |
| Declared-schema validity | 75/75 | 74/75 | fail |
| Physical KV-prefix sharing | 75/75 | 75/75 | pass |

Report SHA-256:
`a3f35f257148dc06e7a912b03449c1c6c1ed38bcf687f6900421d3d31a0e2309`.
The report pins Models scorer revision `beea98aa922fd45915883213e84a10911e2e98e4`, base GGUF
`061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`, adapter
`f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`, and exposed
records `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`.

## Failure analysis

- The policy abstained on three applicable requests: `multiple_53`, `multiple_76`, and
  `simple_264`. The V19 diagnostic margins and V20 live margins differed enough that
  `multiple_53` moved from `4.8653` to `4.7051`, below the frozen primary threshold `4.8179874`.
  Thus the prior retrospective 48/50 decision estimate was not reproducible in the complete live
  generation run. The other two abstentions remained below the frozen rescue conditions.
- `irrelevance_165` was the sole false call, exactly exhausting the allowed no-call error budget.
- `simple_83` generated JSON arrays for `coord1` and `coord2`, while the declared source schema
  gives both fields type `string`. The upstream V9 evaluator recorded `schemaValid=true` for the
  same output; the strict Java source-schema validator correctly rejects that type mismatch.
  This is a source/evaluator inconsistency, not permission to relax the schema gate after seeing
  the result. Its tool arguments also contain positive longitudes where the accepted values are
  negative, so the output is not exact under either interpretation.
- The seven non-exact positive cases include three missed call decisions and four generated calls
  with mismatched arguments or incomplete selection. The frozen exact-call floor was met, but the
  decision and schema floors were not.

The selected local `simple_121` mechanics run used the correct runtime adapter bundle and passed
exact output, schema, and physical sharing before the complete run. Two earlier launcher attempts
stopped before inference: Gradle's module working directory did not resolve a relative adapter
path, and the training checkpoint directory lacked the runtime manifest. The successful commands
used absolute paths and the existing runtime bundle, whose adapter weights matched the frozen
SHA-256. These are launcher/artifact-layout findings, not changed model inputs.

## Next experiment boundary

V20 is rejected unchanged. Any new decision policy, training run, schema handling, or generation
constraint requires a separately versioned preflight and development result. Retuning V20 after
observing these cases would invalidate its gate. A future development pass still cannot itself
qualify the model: the sealed window and all product, integration, provenance, memory, and
performance gates remain mandatory.

No VPS, cloud GPU, external inference server, or Rust inference kernel was used by V20. There is
no V20 infrastructure to decommission.
