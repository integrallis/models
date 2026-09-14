# License and attribution audit

Checked 2026-09-13 before publication qualification. This audit does not authorize publication by
itself; it fixes the notices a passing adapter must carry in both its downloadable bundle and its
publication metadata.

| Role | Pinned source | Declared license | Evidence |
|---|---|---|---|
| Base model | `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0 | Pinned `LICENSE`, SHA-256 `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |
| Positive tool-call data | `edbuildingstuff/bfcl-ft-data` at `a7ceb3b1e1605f609fda6f2befec704f575290de` | CC-BY-4.0 | Pinned `README.md`, SHA-256 `9aca1ac9e77a4affbbd6f47914e45b5707730c2ccd41e0788a2167eb2160c885` |
| Hard no-call data | `ShishirPatil/gorilla` at `c15b2a151662cac9839c96d7dfb1493b5329c975` | Apache-2.0 | Pinned repository `LICENSE`, SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

The positive corpus requires attribution to its dataset author, Salesforce/xLAM (APIGen), and
Team-ACE/ToolACE. The hard no-call rows come from `BFCL_v3_live_irrelevance.json`, SHA-256
`0d259e1c3ab6ba06c2a51911e2f49ecaa2eac3723a4a14f183a7687955e61338`. Six rows shared with the
static qualification inputs were excluded before deterministic selection. The adapter is a
modified work produced by activated-LoRA continuation and must identify that change.

Pinned evidence URLs:

- <https://huggingface.co/Qwen/Qwen3-1.7B/raw/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/LICENSE>
- <https://huggingface.co/datasets/edbuildingstuff/bfcl-ft-data/raw/a7ceb3b1e1605f609fda6f2befec704f575290de/README.md>
- <https://raw.githubusercontent.com/ShishirPatil/gorilla/c15b2a151662cac9839c96d7dfb1493b5329c975/LICENSE>
