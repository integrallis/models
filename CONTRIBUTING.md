# Contributing to models (JVLLM)

Thank you for your interest in contributing to models! This guide will help you get started.

## Development Setup

### Prerequisites

- **JDK 25** — Required by the current API and FFM-based GGUF loader
- **Gradle 9.4.1** — Included via the Gradle Wrapper (`./gradlew`)
- **Git** — For version control

### Clone and Build

```bash
git clone https://github.com/integrallis/models.git
cd models
./gradlew build
```

## Build Commands

All commands run from the `models/` directory:

```bash
# Full build (all library modules)
./gradlew build

# Run all tests (excludes slow/benchmark/integration)
./gradlew test

# Run a single test class
./gradlew :models-api:test --tests "com.integrallis.models.api.SomeTest"

# Unit tests only
./gradlew unitTest

# Slow tests (large models, extended inference)
./gradlew slowTest

# Code formatting (auto-fix)
./gradlew spotlessApply

# Code formatting (check only)
./gradlew spotlessCheck

# Compliance checks (SBOM, governance, reproducibility)
./gradlew complianceCheck

# Generate aggregated Javadoc
./gradlew aggregateJavadoc
```

## Test Conventions

We follow **test-driven development** with integration tests first:

1. Write an integration test exercising the public API end-to-end
2. Run it — it fails (Red)
3. Implement the minimum code to make it pass (Green)
4. Refactor, then add `@Tag("unit")` tests for edge cases
5. Repeat

### Test Tags

| Tag           | Task                | Description                              |
|---------------|---------------------|------------------------------------------|
| *(default)*   | `test`              | All tests except slow/benchmark/infra    |
| `unit`        | `unitTest`          | Fine-grained unit tests                  |
| `slow`        | `slowTest`          | Large model / extended inference tests   |
| `benchmark`   | —                   | Performance regression tests             |
| `integration` | `integrationTest`   | Integration tests                        |

Always add the appropriate `@Tag` annotation to new test classes.

## Code Style

- **Google Java Format** — Enforced via Spotless. Run `./gradlew spotlessApply` before committing.
- **No Lombok** — This is a low-level library; we use explicit code.
- **Minimal dependencies** — Library modules should only depend on `slf4j-api` at runtime.
- **Pure Java** — No JNI, no FFM-to-C++ bindings, no native backends (in the pure-java backend).

## Module Structure

```
models-api/                  — Backend SPI, ChatModel, Tokenizer, SamplingOptions, TokenStream
models-runtime/              — Generation lifecycle, templates, sampling, observability
models-backend-purejava/     — GGUF parser, scalar kernels, KV cache
models-spring-ai/            — planned adapter scaffold
models-langchain4j/          — planned adapter scaffold
models-embedding/            — planned vectors bridge scaffold
models-test/                 — planned test-support scaffold
models-bench/                — planned benchmark scaffold
```

Dependencies flow downward: `api -> runtime -> backend-purejava`.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests first** — follow the TDD workflow described above
3. **Keep changes focused** — one feature or fix per PR
4. **Run the full build** before submitting:
   ```bash
   ./gradlew spotlessCheck build complianceCheck
   ```
5. **Write a clear PR description** — explain what changed and why
6. **CI must pass** — the GitHub Actions pipeline runs build + test + SBOM generation

## Reporting Issues

- Use GitHub Issues for bugs and feature requests
- For security vulnerabilities, see [SECURITY.md](SECURITY.md)
