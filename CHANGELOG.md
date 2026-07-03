# Changelog

All notable changes to models are documented here.

## [Unreleased]

### Added

- Maven Central staging and JReleaser release automation based on the proven
  `integrallis/mfcqi-java` pipeline.

### Changed

- Limited the first release to the three implemented modules.
- Removed the accidental sibling vectors dependency so the repository builds
  and publishes independently.
- Reframed release documentation as a pre-alpha scalar GGUF runtime with
  framework modules explicitly identified as scaffolding.

### Fixed

- Reset the mutable KV cache between independent generations and serialize
  calls sharing one backend.
- Replaced collision-prone BPE merge keys based on `String.hashCode()` with
  full merge-pair keys and added byte-level/ranked-merge regressions.
- Removed unused runtime dependencies and made strict Javadocs, dependency
  locks, staged publications, SBOMs, SpotBugs, and 80% published-module
  coverage part of the release gate.
