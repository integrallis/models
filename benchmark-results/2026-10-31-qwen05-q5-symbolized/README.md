# Symbolized CPU profile

This is diagnostic sampling of unchanged native source with symbols; it is not
qualification evidence. The native override is explicitly identified by SHA.

Correction to the initial code-inspection hypothesis in the frozen protocol:
`nativeGroupedAttention` is enabled only by `usesGraniteScaling()`. Qwen uses Java
attention, so the Java fused-grouped-attention flag does affect its selected path.
The backend diagnostic reports native capability, not per-architecture activation.
CPU sample share also weights active worker count and is not wall-clock latency.
