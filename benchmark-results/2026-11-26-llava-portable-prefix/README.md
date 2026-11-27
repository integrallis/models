# A LLaVA 1.5 7B prefix holding an image, moved between independent backends (#425)

**Measured here, 2026-11-26, one run on a developer laptop** (`raw/host.txt`). Code: Models commit
`519a0f84` (the run used the identical working tree immediately before that commit). Test:
`LlavaRealModelPortablePrefixIntegrationTest`, task `:backend-java:llavaPortablePrefixIntegrationTest`.

Weights, the pinned catalog files (second-state/Llava-v1.5-7B-GGUF @ ffb5e9a1, sha256 checked on
download): `llava-v1.5-7b-Q4_K_M.gguf` (2687b20a...) and `llava-v1.5-7b-mmproj-model-f16.gguf` (50da4e5b...).

| field | value |
|---|---|
| prefix | "USER: " + one 336x336 synthetic PNG + "\nWhat is in the image? ASSISTANT:" = 595 positions, 576 image rows |
| payload | 623,902,939 bytes |
| receiver fork vs donor fork, next-token logits | identical (`containsExactly`) |
| receiver recomputing the prefix itself vs donor | identical |
| same bytes imported with a different image behind the same placeholders | refused, naming segment 1 |

Times printed by the test (donor session prefill 6,418,580 ms, import 13,512 ms, recomputation
6,402,386 ms) are single-run laptop figures on the row-by-row session image path; they are recorded
for completeness and support no performance claim. Raw: `raw/` (JUnit XML, console excerpt, host).
