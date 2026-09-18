# Java GPU acceleration

`backend-tornado` is an optional in-process device backend for Models. Its kernels are Java source;
TornadoVM compiles eligible Q4_0, Q4_K and Q6_K projections for a GPU at runtime. It does not invoke
an external model server or embed another inference engine.

The Maven artifact does not bundle or transitively install TornadoVM's device runtime. Applications
provide a compatible TornadoVM distribution and launch configuration explicitly; without it, the
default loader uses the Java Vector API.

Adding this module also registers its accelerator through Java `ServiceLoader`.
`PureJavaBackend.loadAutomatic(...)`—and ModelJars' Java backend path—select it only when the exact
artifact and device pass the capacity gate.

The qualified production scope is deliberately narrow:

- NVIDIA GPUs reached through TornadoVM's PTX backend;
- Q4_0 (Q8_0 activations) and Q4_K / Q6_K (Q8_K activations) GGUF projection work for both prefill
  and single-token decode, with the two activation families never mixed inside one grouped dispatch;
- a single weight tensor under 2 GiB, the limit of TornadoVM's 32-bit array index;
- attention and unsupported tensor formats remain on the Java Vector API; and
- eager readiness compiles reusable plans before the first visible request.

The capacity selector passed exact output-parity and full-model gates on NVIDIA A16 and A40 profiles
with Q4_0. The K-quant kernels are covered off-device against the vectors-core CPU kernels — exactly
on exactly-representable super-blocks, and within two float roundings per super-block otherwise — and
have not yet been run on a GPU. AMD, Intel, and Metal devices remain on the CPU fallback until they
pass equivalent real hardware gates.

Run the hardware gate with the `accelerator-profile` command of `models-bench`, launched through the
TornadoVM launcher (see the models-bench README). It reports the selected device, readiness time,
prefill and decode throughput, and how many projections reached the device per GGUF weight format.

The single-token decode step of causal grouped-query attention can additionally run on the device
behind `TornadoBackendOptions.withAcceleratedAttention()`. Prefill chunks stay on the Vector API,
because a layer holds one KV mirror and a prefill chunk is a different execution shape. It is off by
default and outside the qualified scope: its kernel and routing are covered off-device, but no real-hardware parity or performance
gate has been run for it, and the only real-model device-attention measurement this codebase has is
a negative one on prefill (A16, 2026-08-29). The device keeps a KV mirror per sequence per layer;
the host cache and its physical prefix sharing are unchanged, but two branches of one shared prefix
each rebuild their own mirror. `TornadoBackendRuntime.attentionRouting()` reports what actually ran,
so a kernel that never accepted a step cannot be mistaken for one that ran and did not help.

See the published [Java GPU acceleration guide](https://integrallis.github.io/models/docs/models/current/gpu-acceleration.html)
for dependencies, launcher requirements, status reporting, and measured evidence.
