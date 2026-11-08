# Rejected attention arithmetic variants

These seven-case diagnostics follow the completed normalization-only full attempt. They do not replace the frozen full corpus, change admission gates or qualify a model. All use the same saved image files and prompts.

- Double CLIP moments with F16-rounded Q/K/V and corrected QuickGELU trade the remaining textvqa-38337 disagreement for textvqa-36979.
- F32 square-root/reciprocal rounding after double moments, with either F32 or F16-rounded attention, also trades disagreements.
- 548656f7 normalizes after the prefill value sum, preserving F32 operands; it still trades disagreements.
- 3b36ce7a keeps prefill Q in F32 and rounds decode Q plus K/V to F16, following the reference's effective operand distinction. Its standalone and post-value-normalization combination regress other retained cases.

None of these changes is in the production branch. The retained runtime remains 3f31646c: double CLIP moments, upstream QuickGELU graph, head-major F32 cache and fixed Q4 views. The full 80-case Q4_K_S candidate later identifies textvqa-36979 as a new token disagreement, so the seven-case set must not be described as complete qualification coverage.

The supplemental archive is 60,866,811 bytes and contains 929 files, including all diagnostic payloads, source and outputs. The original 2.23 GB archive retains the earlier full run. Both local copies were SHA-256 verified before primary-worker termination.
