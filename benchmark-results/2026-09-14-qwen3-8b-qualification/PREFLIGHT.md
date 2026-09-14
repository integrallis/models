# Qwen3 8B Java qualification preflight

- **Experiment:** Qwen3 8B Q4_K_M base conformance, then unchanged tool/no-tool screen.
- **Provider / plan / region:** Hetzner Cloud `cpx51` / `ash` / Ubuntu 24.04 x86-64. The lower
  priced European locations were marked unavailable by the provider API at creation time.
- **Expected price:** EUR 0.4479 per hour (provider API observed 2026-09-14).
- **Maximum duration and spend:** three hours, EUR 1.35 before traffic; delete earlier on any
  checksum, Java conformance, or tool screen failure.
- **Code:** `feat/alora-cache-sharing` at `2c702d4` before the run; final evidence records the
  exact checked-out revision.
- **Runtime rule:** Java 25 and the owned Models pure-Java path only. No external inference
  runtime is installed or used; a reference implementation may be consulted only outside the
  product execution path for declared oracle comparison.
- **Correctness gate:** pinned GGUF SHA-256, existing Qwen3 8B greedy-token fixture, then a
  separately recorded full tool/no-tool gate with the native template.
- **Performance evidence:** measured cold/warm TTFT, decode rate, peak RSS, and host fingerprint.
- **Stop condition:** copy and hash final evidence locally, then delete the server and confirm the
  provider has no server, volume, floating IP, primary IP, firewall, or snapshot owned by this run.
