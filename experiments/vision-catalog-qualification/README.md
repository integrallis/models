# LLaVA catalog attempt

This pre-merge experiment attempts to qualify LLaVA 1.5 7B Q4_K_S and Q4_K_M with
the same F16 projector. It does not qualify Qwen3-VL or claim specialized OCR support.
The immutable artifacts, dataset shards, selection seed and thresholds are in
`manifest.json`. The runtime is the exact recorded PR380 commit, not a released library.

## Hypothesis and frozen decision rule

The shipping two-file image backend preserves the independent reference's task
accuracy and completes a matched image-question workload within 1.10 times its
median latency. Both arms run sequentially on the same Linux worker with 8 threads,
1024 context positions, greedy decoding, 32 output tokens and F32 KV cache.
The reference is Transformers CLIP plus PyTorch MLP and an independently built
llama.cpp decoder using the identical GGUF weights. The existing integration
experiment additionally compares the optimized F16 KV reference.

There are 32 deterministic validation cases from each of VQAv2 and TextVQA.
Selection sorts unique question IDs within pinned shards by SHA-256 of the seed
concatenated with the decimal question ID, then takes cases round-robin across
shards. Labels never influence selection. Images are decoded and losslessly saved
as RGB PNG; both arms consume those same files. TextVQA OCR tokens are not supplied.
This small subset is an admission probe, not a reproduction of publisher benchmarks.

Both arms use the LLaVA 1.5 Vicuna system prefix, insert the image, then append
`\n{question}\nAnswer the question using a single word or phrase. ASSISTANT:`.
The pinned upstream LLaVA `TextVQAAccuracyEvaluator` supplies the VQA soft score
and normalization for both ten-answer datasets. The score must be at least 0.50
on VQAv2 and 0.20 on TextVQA, with at most 0.005 absolute loss against the reference.
At least 0.99 of complete generated token sequences must match. These thresholds
are declared before observation, not inferred from publisher numbers.

The first two selected cases from each dataset receive one warmup and three
measured runs. Prefill (including image processing), decode and total latency
are compared per case; each median candidate/reference ratio must be <= 1.10.
Zero-token decodes are excluded only from decode timing and remain in total timing.
The first eight cases per dataset also run with a mean-colored blank image in both
arms. The candidate's VQAv2 score must improve by at least 0.05 over that control.
All cap hits, case outputs, warmup outputs, load times and resource usage are retained.

Failure, missing evidence, mismatched inputs or missing reference means no admission.
Do not tune selection, scoring or thresholds after observation. Runtime defects may
be fixed, but every subsequent attempt gets a new source hash and output directory.

## Catalog integration

A passing backend result is necessary but insufficient: the ModelJars branch must
verify both files through its installer and run an image question through its public
API. No record may call this branch build a released backend. Public Maven catalog
publication requires the dependency release sequence after PR review; pre-merge
qualification and the exact candidate marker can be exercised from an isolated payload.

## Stop conditions

Stop on corrupt downloads, nonfinite outputs, unsupported architecture, a hard
worker deadline or the recorded spend cap. Preserve failures and terminate only
this experiment's worker. See the per-run control record for ownership and deadlines.
