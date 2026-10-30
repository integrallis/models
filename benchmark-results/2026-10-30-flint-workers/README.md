# Flint native worker diagnosis — complete

Runtime payload is unchanged from `9c453e9de85f77b398e9ad57eba13e305db96f72`.
Worker revision: `6cce978f926d3ccd03435429c846e7ed585a649f`.
Default 16 active native workers, then 8 and 4, followed by the unchanged 16-thread Ollama
control. One case, one warmup, three repetitions, 512 output tokens, context 2048; not
qualification. Existing native decode-worker settings change scheduling only.

Instance `i-0c91cfc3df883ced3`, root `vol-02e5a124a7eff9c1f`, launched 2026-10-30T19:28:51Z.
The independent shutdown bound is 90 minutes; root deletion is enabled. The gp3 volume is
120 GiB, 3000 IOPS, 500 MiB/s, verified against EC2. Instance termination and root-volume deletion were verified.
The full build for the identical runtime is retained in the attention campaign validation.

Median decode: default16 9.894, decode8 11.258, decode4 10.692, Ollama16 14.472 tokens/s.
Eight workers reach 77.79% of the control, below the unchanged 80% floor. All nine native
answers are byte-identical across profiles; each arm passes its three visible fact/citation checks.
These single-case reports intentionally lack the coverage required for qualification.
