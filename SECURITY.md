# Security Policy

## Supported Versions

| Version | Status |
|---------|--------|
| 0.2.x | Current |
| 0.1.x | Security fixes |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in models (JVLLM), please report it responsibly.

### How to Report

**Email:** security@integrallis.com

Please include:
- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fix (optional)

### Response Timeline

| Action                     | Timeline         |
|----------------------------|------------------|
| Acknowledgment of report   | Within 48 hours  |
| Initial assessment         | Within 5 days    |
| Fix development            | Within 30 days   |
| Public disclosure           | Within 90 days   |

### Coordinated Disclosure

We follow a **90-day coordinated disclosure policy**:

1. Reporter submits vulnerability via email
2. We acknowledge receipt within 48 hours
3. We assess severity and develop a fix
4. We release a patched version
5. We publish a security advisory (GitHub Security Advisories)
6. Reporter may publish their findings after 90 days or after the fix is released, whichever comes first

### What Qualifies

- Memory safety issues (buffer overflows, use-after-free in Arena-managed segments)
- Denial of service via crafted model files or inference inputs
- Information disclosure through side channels
- Any bug that compromises data integrity of model weights or inference outputs

### What Does NOT Qualify

- Performance issues or inefficiencies (report as regular issues)
- Bugs in test code or benchmarks
- Issues requiring physical access to the machine

## Security-Relevant Design Properties

Security-relevant boundaries:

- **Published execution backends are explicit** — `backend-java` is Java bytecode.
  `backend-native` optionally loads the small Models-owned Rust kernel library through
  Java FFM; it does not load llama.cpp, Ollama, or another inference engine.
- **Apple bridge integrity is enforced before loading** — The bridge bundled in
  `backend-apple` is SHA-256 verified, extracted to a content-addressed owner-controlled cache,
  and reverified on every resolution. An explicit bridge path requires its expected SHA-256.
  The unverified override is an explicit local-development opt-in.
- **Apple native results are length-delimited and call-owned** — The FFM bridge reads only the
  byte length returned for that call, and frees the result after decoding. Native error detail is
  not stored in shared process state.
- **No downloader in the core runtime** — The published core modules perform no network I/O.
  Users supply a local model path or use the separate ModelJars facade.
- **Arena-based model mapping** — The backend owns a shared `Arena` and closes it from
  `PureJavaBackend.close()`; using the backend after close is unsupported.
- **Artifact signing is external** — JReleaser signs staged Maven artifacts with the maintainer's
  GPG key. Runtime libraries do not perform release-signing operations.
- **Publication is allowlisted** — Only modules named in `publishedModuleNames` are staged for
  Maven Central. Adding a Gradle subproject does not publish it.

## Supply Chain Security

- **SBOM generation** — CycloneDX SBOMs for every published module are generated and validated by
  `complianceCheck`
- **Dependency locking** — Gradle lockfiles pin all transitive dependency versions
- **Vulnerability scanning** — OWASP Dependency-Check is available as an explicit release audit;
  findings with CVSS >= 7.0 fail that task
- **Artifact signing** — The release workflow signs Maven Central artifacts
  with GPG through JReleaser.
- **Deterministic JAR settings** — JAR tasks disable file timestamps and use
  reproducible file order. A byte-for-byte clean-room rebuild has not yet been
  independently verified.
- **SAST** — A GitHub CodeQL workflow is configured for pull requests, pushes
  to `main`, and a weekly schedule.
- **Code quality** — MFCQI scores the combined published production sources and
  each source-bearing module on pull requests and pushes to `main`.
