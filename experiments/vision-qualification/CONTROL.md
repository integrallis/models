# PR 380 integration verification

- Experiment: real converted LLaVA 1.5 projector against independent Transformers CLIP + PyTorch MLP, then two-file image-to-text smoke.
- Branch: complete/pr380-vision-qualification; payload commit/hash recorded before execution.
- Provider: EC2 us-east-1a, m6a.4xlarge (16 vCPU, 64 GiB), Ubuntu 24.04, 40 GiB gp3.
- Expected duration: 30–60 minutes; hard lifetime two hours.
- Maximum spend: $3 (existing project compute authorization; terminate early when done).
- Creation deadline: 2026-11-07T17:45:44.731073+00:00.
- Deletion deadline: 2026-11-07T19:45:44.731073+00:00.
- Correctness gate: normalized pixels max absolute error <= 0.02; projected embeddings cosine >= 0.999 and normalized RMSE <= 0.01; finite, identical shapes; independent text-vs-injected-logit synthetic regression.
- Performance gate: measurement completeness only for integration readiness: same host, image, artifact, threads, warmups, repetitions; report actual distributions. No catalog speed claim or admission from this smoke.
- Stop condition: timeout, spend cap, or completed evidence; terminate owned VM only.
- Secrets: ephemeral SSH public key only, no service/API credentials.
- Evidence destination: experiments/vision-qualification/results/; full raw vectors retained outside git in audit-artifacts/pr380-vision/.
- Quota checked before provisioning: standard 192 vCPU; no pending/running/stopping/shutting-down instances.
- Artifact: second-state/Llava-v1.5-7B-GGUF revision ffb5e9a1a3172f82448fbdbe578a051c8dc2ff77, llava-v1.5-7b-mmproj-model-f16.gguf and llava-v1.5-7b-Q4_K_M.gguf.
