# Qwen3 1.7B applicability-contract V12 preflight

Status: **policy-only V12 rejected by the frozen exposed screen; live development and the sealed
qualification window were not opened**.

## Single variable under test

V12 reuses the unchanged V9 rank-32 adapter and adds the bytes in
`tool-applicability-policy.txt` as the first caller system instruction. No weight, data,
masking, loss, sampling, decoder, or qualification-window change is part of this screen.

The policy targets the recurring capability-boundary errors seen across V4, V5, V8, V9, and
V11: calling a related-but-inapplicable tool or inventing a required argument. It is opt-in and
must not alter the global Qwen template.

## Frozen evaluation order

1. Prove Python evaluator tests and Java/Python prompt-token parity.
2. Run the fixed exposed window: 25 simple, 25 multiple, and 25 irrelevance cases.
3. Run 25 cases of each kind from the disjoint BFCL live-development source.
4. Inspect reports only. Do not open the sealed 300-case qualification window unless every
   exposed gate passes.

## Frozen gates

| Stage | Requirement |
|---|---|
| Prompt parity | One system turn; policy occurs once before `# Tools`; activated invocation remains `[151644, 77091, 198]`; Python and Java prompt bytes and token IDs are identical |
| Exposed | Syntax 75/75; schema 75/75; tool exact at least 43/50 and no worse than the base with the same policy; false calls at most 1/25 |
| Live development | No syntax or schema regression; tool exact and false-call rate both move in the intended direction relative to V9 without the policy |
| Sealed qualification | Existing frozen 100+100+100 V9 manifest; syntax 300/300; schema 300/300; exact at least 170/200 and no worse than same-policy base; false calls at most 5/100 |

## Frozen identities

| Input | SHA-256 |
|---|---|
| Applicability policy file | `1064ab56de3e4abf274641cc158e5a0bd5cd905184e8b8e10b3763cec1f96c4b` |
| V9 adapter weights | `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21` |
| V9 training manifest | `d0620df4860c879f6d3f6e5573168bb08afc0b48d954040825e1317972f7de47` |
| Frozen qualification window | `5fee70a0d18801ac9541b2cfa3b504c0fd06e98f5bf1adeb196adfc89dffa1d2` |
| Static evaluator | `dbe5374139cb2fb4ccaed9e972568de0fb9fc6d8bfa523526823f7e1df70b5e3` |
| Static evaluator tests | `9dd75208d0b6fe796d68105236bc7f2877e479b44ffad32d171d68606294d9e3` |
| Live-development evaluator | `7741af5281296ba75c75ff232b49dca456140e4dc7e75df151c705f5a2246951` |
| Live-development tests | `f04de923b2e4e2e3962176902f62fe04f49ef0f87ba7d21a3e90540362448713` |
| Policy prompt-oracle generator | `30274051366f0eec4fba5b47043efae09818a64ec7de75f14e1fc303bc5ba641` |
| Policy prompt-oracle tests | `52cfba6a6cf1e87d43853cae5ddf3db65644b7674d443f450117354bd4d12cf0` |
| Evaluator normalization helper | `05d4f219b45c951911b07713499371e2cfb8d4bb2b8135a16ea1fb21fc3e2968` |
| Accepted oracle properties | `134e25e9d9df4bb9dfb738a43432826aacc292635fba7821e19d544423179aa4` |
| Accepted oracle prompt | `478140782be508a7a44e3324e3fd742cdc9e2fd8b893480c25d1dc6640b1bd8e` |
| Accepted oracle token IDs | `59ee373ac86901dda5c19037bebfbadc61e9923f8546188bc68a82a1be08654b` |
| Java chat-template implementation | `799cb61d77539555cfc20de5b515da377f802aa6d7b0738b463a86c0bcbebdeb` |
| Java published-template tests | `15c6526f4317ec6ed035b6714cc4ed95b6f305e85cdf8fbabb1738bc8c77fca8` |
| Java policy-parity gate | `6f353cbd7db0c9b7c243fb6e3ca15e176516eeeb2b0d1deb3494c33e3927aa3e` |

The complete Python suite passed 90/90 before these hashes were frozen.

## Prompt-parity result

The first pre-score comparison rejected two real prompt-contract defects without producing any
model score. Java initially preserved raw schema whitespace where Qwen's published Jinja template
uses `tojson`; a test-first correction now produces the published deterministic spacing without
moving developer-supplied schema bytes into trusted control segments. A first policy draft also
spelled the model's control delimiters in caller text. Models correctly kept those bytes untrusted,
while the Hugging Face tokenizer elevated them to special tokens. The redundant delimiters were
removed rather than weakening Models' injection boundary; the published tool preamble already
defines the empty-list protocol.

The regenerated `policy-oracle-v2` contains 1,072 prompt bytes and 228 little-endian token IDs.
`ActivatedLoraModelIntegrationTest.matchesThePinnedPolicyPromptAndTokensExactly` passed against the
real Qwen3 1.7B Q8_0 GGUF and the V9 adapter: Java and Hugging Face prompt text and every token ID
are identical, and the activation sequence remains `[151644, 77091, 198]`. The earlier rejected
oracle is retained separately as failure evidence and is not an input to evaluation.

## Stop conditions

- Any prompt or token mismatch stops the experiment.
- Any exposed syntax, schema, exactness, or false-call failure rejects policy-only V12 without
  opening the sealed set.
- Training is a separate follow-up only if both policy-only screens improve the intended metric
  without damaging the others.

## Outcome

The exposed 25 simple + 25 multiple + 25 irrelevance screen stopped V12. The adapter produced
75/75 syntactically valid and 75/75 schema-valid results, but only 42/50 exact tool calls versus
the 43/50 floor and the same-policy base's 45/50. It made 5/25 false tool calls versus the 1/25
ceiling. The policy therefore did not solve the recurring applicability error and also reduced
tool-call exactness. No live-development or sealed case was evaluated.
