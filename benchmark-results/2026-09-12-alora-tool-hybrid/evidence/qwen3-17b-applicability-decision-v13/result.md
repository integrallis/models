# V13 result: rejected before treatment

V13 did not produce a candidate. The frozen screen required the treatment to
classify at least nine more of the 81 no-call validation rows correctly than the
control, while losing no more than seven of the 390 call rows and improving
balanced accuracy.

The exact 32-step control classified 77/81 no-call rows and 362/390 call rows
correctly. Only four no-call errors remained, so the required gain of nine was
mathematically impossible. The treatment process was terminated before its
first optimizer step, and neither the exposed generation screen nor the sealed
300-case qualification set was opened.

The first control attempt exposed a separate harness defect: Hugging Face
Trainer leaves Accelerate's BF16 autocast wrapper installed on the trained
model. Comparing those wrapped logits with a plain reloaded artifact produced
different hashes even though all 588 serialized adapter tensors were identical.
The harness now removes that wrapper before every artifact and inference gate.
The corrected control proved exact base behavior with the adapter disabled,
exact adapter state after save/reload, and bit-identical sampled logits after
save/reload.

Evidence:

- `gpu-screen/host-preflight.json`: exact environment and input preflight
- `gpu-screen/control-screen.log`: first attempt, stopped by the pre-checkpoint
  hook-order defect before training
- `gpu-screen/control-screen-r2.log`: completed training, then exposed the
  autocast-wrapper comparison defect
- `gpu-screen/control-screen-r3.log`: corrected 32-step control
- `gpu-screen/control-training-manifest.json`: corrected control metrics and
  hashes
- `gpu-screen/control-reload-diagnostic.json`: trained/reloaded state and logit
  identity
- `gpu-screen/treatment-screen.log`: treatment stopped before optimization

The failed and diagnostic artifacts remain evidence only. None is eligible for
runtime packaging or publication.
