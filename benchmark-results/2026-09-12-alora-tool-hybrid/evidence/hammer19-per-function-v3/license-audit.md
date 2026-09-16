# License and attribution audit

Checked 2026-09-13 before behavioral qualification. This audit does not authorize publication by
itself; it records the notices that a passing adapter must carry in its artifact and POM.

| Role | Pinned source | Declared license | Evidence |
|---|---|---|---|
| Base model | `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 | Pinned `LICENSE`, SHA-256 `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |
| Tool-call training data | `edbuildingstuff/bfcl-ft-data` at `a7ceb3b1e1605f609fda6f2befec704f575290de` | CC-BY-4.0 | Pinned `README.md`, SHA-256 `9aca1ac9e77a4affbbd6f47914e45b5707730c2ccd41e0788a2167eb2160c885` |
| No-call training data | `MadeAgents/xlam-irrelevance-7.5k` at `34323bf09efc7e4a394998a0fa91ff997617c369` | CC-BY-4.0 | Pinned `README.md`, SHA-256 `5ac5b26d3f8e7563691f87c3a4d8f1c97bdda899ecd4095ac9075bce9f59db10` |

Pinned evidence URLs:

- <https://huggingface.co/Qwen/Qwen3-0.6B/raw/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE>
- <https://huggingface.co/datasets/edbuildingstuff/bfcl-ft-data/raw/a7ceb3b1e1605f609fda6f2befec704f575290de/README.md>
- <https://huggingface.co/datasets/MadeAgents/xlam-irrelevance-7.5k/raw/34323bf09efc7e4a394998a0fa91ff997617c369/README.md>

The positive corpus requires attribution to its dataset author, Salesforce/xLAM (APIGen), and
Team-ACE/ToolACE. The no-call corpus requires attribution to MadeAgents and its xLAM source. The
adapter is a modified work trained from those sources, so its notice must identify the training and
filtering changes and link CC-BY-4.0. ModelJars publication must expose the same license and
attribution information rather than relying only on this experiment note.
