"""Qwen3-8B last-token pooling in-process, standing in for CLM's vLLM embeddings endpoint.

CLM's `Embedder` is transport: it POSTs to an OpenAI-compatible `/v1/embeddings` served by vLLM
`--runner pooling` and then L2-normalises whatever comes back (`clm/embedder.py`, `_fetch` applies
`l2()` to every row). So the contract a substitute has to honour is exactly:

    embed(texts) -> ([n, 4096] L2-normalised float32, tokens_spent)

which is last-token pooling over the final hidden state, normalised. That is what this does, with
`transformers` instead of a server.

Why not vLLM: vLLM 0.30 ships a CUDA 13 torch and this host's driver is 12.8, so the engine will
not initialise. Going direct removes the version fight and also removes vLLM's pooling
configuration from the trust surface -- there is no question about which pooling ran, because the
pooling is three lines below.

**This is a deviation from their reference stack and is recorded as one.** A frozen model in one
dtype is the same function either way, but kernel and attention-implementation differences move
low bits, and the heads are a cosine away from the answer. Treat scores from this arm as CLM's
architecture measured faithfully, not as a byte-reproduction of their server.
"""
from __future__ import annotations

import numpy as np
import torch


def l2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """The same normalisation `clm.embedder.l2` applies, including its epsilon."""
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


class LocalQwenEmbedder:
    """Duck-types `clm.embedder.Embedder`: `embed`, `healthy`, and a `cache` the Engine may touch."""

    def __init__(self, model_id: str = "Qwen/Qwen3-8B", device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, batch: int = 16,
                 max_tokens: int = 2048):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.device, self.batch, self.max_tokens = device, batch, max_tokens
        self.cache: dict[str, np.ndarray] = {}
        self.tokens_spent = 0

    @torch.no_grad()
    def _forward(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                 max_length=self.max_tokens)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        hidden = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"]
        self.tokens_spent += int(mask.sum().item())
        # Last non-pad position per row. Padding is on the right by default for this tokenizer, so
        # the final real token is at sum(mask) - 1 and NOT at -1; taking -1 would pool a pad token
        # for every row shorter than the longest in the batch.
        last = mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=self.device), last]
        return pooled.float().cpu().numpy()

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        spent_before = self.tokens_spent
        for start in range(0, len(missing), self.batch):
            chunk = missing[start:start + self.batch]
            for text, vector in zip(chunk, self._forward(chunk)):
                self.cache[text] = l2(vector.astype(np.float32))
        return np.stack([self.cache[t] for t in texts]), self.tokens_spent - spent_before

    def healthy(self) -> bool:
        return True
