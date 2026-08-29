# Java GPU acceleration

`backend-tornado` is an optional in-process device backend for Models. Its kernels are Java source;
TornadoVM compiles eligible Q4_0 projections for a GPU at runtime. It does not invoke an external
model server or embed another inference engine.

The Maven artifact does not bundle or transitively install TornadoVM's device runtime. Applications
provide a compatible TornadoVM distribution and launch configuration explicitly; without it, the
default loader uses the Java Vector API.

Adding this module also registers its accelerator through Java `ServiceLoader`.
`PureJavaBackend.loadAutomatic(...)`—and ModelJars' Java backend path—select it only when the exact
artifact and device pass the capacity gate.

The qualified production scope is deliberately narrow:

- NVIDIA GPUs reached through TornadoVM's PTX backend;
- Q4_0 GGUF projection work for both prefill and single-token decode;
- attention and unsupported tensor formats remain on the Java Vector API; and
- eager readiness compiles reusable plans before the first visible request.

The capacity selector passed exact output-parity and full-model gates on NVIDIA A16 and A40
profiles. AMD, Intel, and Metal devices remain on the CPU fallback until they pass equivalent real
hardware gates.

See the published [Java GPU acceleration guide](https://integrallis.github.io/models/docs/models/current/gpu-acceleration.html)
for dependencies, launcher requirements, status reporting, and measured evidence.
