# Why these three cells are not evidence

`cap 2048` against `--context 2048`. Generation ran past the KV window on 7 of 9, 8 of 9 and 2 of 9
cases with `IllegalArgumentException: position out of range: 2048`.

They are kept because the failure is the finding that fixed the protocol -- the fleet worker pins
`--context 2048` (`scripts/fleet/qual-worker-two-arm.sh:427`), so no shard can run a cap near 2048
-- and because a reader checking `derive.py`'s exclusion logic needs the inputs that trigger it.

They are not reported as results. The cases that failed are the long-generating ones, so the
surviving attempts are the short ones and every rate computed over them reads better than the
truth.
