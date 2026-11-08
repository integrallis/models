# Q4_K adjacent scale conversion experiment

The full production repeat at `3f31646c743647449310de2c5ff57fbd55d9e957`
passes the longer integration confirmation. Its Q4_K_S short-answer comparison
passes three of four faster-reference cases but fails `vqav2-423731001`: decode
ratio 1.1081989206096103 against the unchanged 1.10 gate. This is a measured
remaining performance failure, not a reason to change the threshold.

Generated AVX2 assembly converts the adjacent half-precision Q4_K `d` and `dmin`
scales with two `vcvtph2ps` instructions per block. Test loading their four bytes
together and converting both in one instruction, then multiplying both by the
same activation scale. Preserve the subsequent integer products, floating-point
accumulation order, ordered-mode behavior and public ABI. No model-specific branch,
alternate execution mode, cached weight expansion or precision change is permitted.

Compile and inspect the generated kernel and run the synthetic Rust tests first.
After the current complete repeat is archived and its worker removed, declare a
new bounded Linux worker and run baseline/candidate/baseline using the established
caption schedule (3 warmups, 7 repetitions), plus the short-answer diagnostic. The
baseline is the exact retained production runtime. Require every generated token
to match and a gain beyond the bracketing baseline variation before retaining code.
Then repeat the complete frozen corpus and both reference precision arms; a small
diagnostic gain is not catalog admission. Keep all rejected results separately.

The existing 0.5.16 release dry run may continue, but do not label these experimental
sources as released or qualified. No extra cloud spend is authorized by this file;
provisioning requires its own control record within the existing campaign cap.
