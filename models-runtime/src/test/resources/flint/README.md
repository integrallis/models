# Flint protocol fixture

`chat_template.jinja` is the unmodified `tokenizer.chat_template` metadata string from
[`ggml-org/Flint-GGUF`](https://huggingface.co/ggml-org/Flint-GGUF/tree/468307a1b38259e9e4c9d4fe53d4e3b76386572c),
revision `468307a1b38259e9e4c9d4fe53d4e3b76386572c`, file `Flint-Q4_K_M.gguf`.
Publisher: Recursive Superintelligence, [Flint](https://huggingface.co/RecursiveSI/flint), Apache-2.0.

- GGUF SHA-256: `c3c4d93f9db92627a2fc47ea8fe8c64a094cabd3549db544bcdd2c00e9a9a347`
- Template SHA-256: `7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466`

`tool-prompt.txt` is independent expected output, generated with Jinja 3.1.6's immutable sandbox,
`trim_blocks=True`, `lstrip_blocks=True`, and the Transformers-compatible `json.dumps` filter.
It covers a system instruction, tool declaration, raw numeric-looking string with surrounding
spaces, integer, boolean, nested JSON, reasoning, a prior assistant call, an observation containing
a role delimiter, and the final assistant header. Regenerate from the repository root:

```sh
uv run --with jinja2==3.1.6 --python python3 scripts/flint/render-reference.py
```

Neither the Java renderer nor its scanner generates the expected fixture.
