# Models 0.3.47 release qualification

Release preparation on 2026-09-25. This patch ships the previously merged routing-policy and
blocking cancellation fixes, completes per-request requirements forwarding in framework adapters,
consumes Vectors 0.1.23, and updates Jackson to 2.21.7. The existing huge-page experiment remains
off by default. No new accelerator support or measured model-selection quality is claimed.

## Routing behavior

The four-argument constructors on `RoutedSpringAiChatModel`, `RoutedChatModel`, and
`RoutedStreamingChatModel` receive a requirements factory over the complete original request.
Both the winner and ranked fallbacks must satisfy its declared capabilities and data boundary.
Existing constructors continue to use `RoutingRequirements.none()`.

Sixteen added regression cases exercise blocking and streaming paths in both frameworks:

- Policy derived from earlier history overrides a higher-scoring hosted client; the next public
  request can route remotely. The delegate receives the identical original request object.
- A failing eligible local client falls back only to another eligible local client; a hosted
  model and a local model lacking the required capability are never invoked.
- An empty eligible set fails before any client invocation.
- Null requirements fail before any client invocation.

These are synthetic control-flow tests. They do not measure model quality, detect PII, attest a
provider's locality, or establish real-provider SLA/cost behavior. Classification, total budgets,
completion deadlines, and streaming cancellation still require further qualification.

Local Temurin 25.0.3 tests on the Intel Mac: 116 cases, 112 passed, zero failures/errors, four
missing-corpus skips. Formatting and staging of the six relevant Models modules passed. These
local checks used the already-qualified staged Vectors 0.1.23 repository while Central publication
was processing; the remote release must independently consume Central's Vectors artifacts.

## Publication checks

The release workflow resolves four independent Maven consumers from a fresh temporary cache:
`backend-java`, `models-router`, `models-spring-ai`, and `models-langchain4j`. It checks the actual
Vectors and Jackson transitive versions from POMs, without Gradle metadata or dependency management
from a consumer masking a defective publication. Results are uploaded with the release logs.

OSV queried on 2026-09-25 reports three advisories for declared Jackson Databind 2.21.4:
GHSA-5gvw-p9qm-jgwh, GHSA-5jmj-h7xm-6q6v, and GHSA-mhm7-754m-9p8w. The direct declarations move to
2.21.7. This is a dependency finding, not a claim that an application exploit was reproduced.

The local staged Maven check passed all four cases (29 expected-version checks). OSV also returned
zero matches for the four external runtime coordinates resolved by those consumers. This is a
narrow dependency scan of these consumers, not of every Models module or application SDK.

## Gates

Release is pending until normal CI/framework checks, native builds/tests on all six supported
platforms, the Apple bridge build/tests, staged Maven consumers, staged notebooks, and the signed
release dry run pass. Publication must then be checked directly against Maven Central.

Optional GPU paths are not newly qualified by hosted CPU CI. The release does not include open
CUDA/Tornado/NEON experimental pull requests, paid-provider tests, or model-quality claims.
