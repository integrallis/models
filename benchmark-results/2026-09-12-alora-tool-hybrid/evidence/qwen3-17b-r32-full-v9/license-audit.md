# License and attribution audit

Checked 2026-09-13 before publication qualification. This audit does not authorize publication by
itself; it fixes the notices a passing adapter must carry in both its downloadable bundle and its
publication metadata.

| Role | Pinned source | Declared license | Evidence |
|---|---|---|---|
| Base model | `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0 | Pinned `LICENSE`, SHA-256 `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |
| Positive tool-call data | `edbuildingstuff/bfcl-ft-data` at `a7ceb3b1e1605f609fda6f2befec704f575290de` | CC-BY-4.0 | Pinned `README.md`, SHA-256 `9aca1ac9e77a4affbbd6f47914e45b5707730c2ccd41e0788a2167eb2160c885` |
| No-applicable-tool data | `MadeAgents/xlam-irrelevance-7.5k` at `34323bf09efc7e4a394998a0fa91ff997617c369` | CC-BY-4.0 | Pinned `README.md`, SHA-256 `5ac5b26d3f8e7563691f87c3a4d8f1c97bdda899ecd4095ac9075bce9f59db10` |

The adjacent `LICENSE` is the pinned Qwen Apache-2.0 license with only a terminating newline added
for the repository text-file convention; its packaged SHA-256 is
`c156170b718ec29139d3653d40ed1986fd92fb7e0959b5c71f3c48f62e6636f4`.

The positive corpus requires attribution to its dataset author, Salesforce/xLAM (APIGen), and
Team-ACE/ToolACE. The no-applicable-tool corpus was created by MadeAgents from Salesforce's xLAM
function-calling data by removing the ground-truth function and relabelling the request as
irrelevant. The adapter is a modified work produced as an activated LoRA and must identify that
change.

Pinned evidence URLs:

- <https://huggingface.co/Qwen/Qwen3-1.7B/raw/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/LICENSE>
- <https://huggingface.co/datasets/edbuildingstuff/bfcl-ft-data/raw/a7ceb3b1e1605f609fda6f2befec704f575290de/README.md>
- <https://huggingface.co/datasets/MadeAgents/xlam-irrelevance-7.5k/raw/34323bf09efc7e4a394998a0fa91ff997617c369/README.md>
