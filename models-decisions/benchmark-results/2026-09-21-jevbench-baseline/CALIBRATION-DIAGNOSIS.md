# Temperature scaling cannot close the calibration gap, and the reason matters

Offline analysis of the 72-item run, 2026-09-21. No inference, no compute: re-tempering from
recorded probabilities is exact, since `p = softmax(z)` at T=1 means `log p = z - c` and
`softmax(log p / T)` reproduces `softmax(z / T)`.

## The sweep

| T | ECE | Brier | accuracy |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.2714 | 0.5410 | 0.6667 |
| **1.0 (as run)** | **0.2062** | 0.5034 | 0.6667 |
| 2.0 | 0.1452 | 0.4802 | 0.6667 |
| **2.5 (best)** | **0.1087** | 0.4792 | 0.6667 |
| 3.0 | 0.1299 | 0.4807 | 0.6667 |
| 10.0 | 0.2418 | 0.5586 | 0.6667 |

**Best achievable ECE is 0.1087**, against a gate of 0.05 and Jev's measured 0.0333. That best case
is fitted *and* evaluated on the same 72 items, so it is an optimistic bound; a held-out temperature
would do worse. Accuracy is invariant across every temperature, confirming the transform never moves
a winner.

## Why: the reliability curve is not monotonic

At T=1.0, the 0.7-0.8 bin is **under**-confident (+0.150) while both neighbours are heavily
**over**-confident (0.6-0.7 at -0.292, 0.8-0.9 at -0.404). Temperature scaling is a single-parameter
monotonic transform. It cannot repair a curve that alternates direction, and at the best T=2.5 the
alternation persists: 0.8-0.9 sits at -0.359 between bins at -0.079 and -0.068.

## The deeper finding: confidence barely ranks correctness

| | top-half accuracy | bottom-half | separation |
| --- | ---: | ---: | ---: |
| T = 1.0 | 0.778 | 0.556 | +0.222 |
| T = 2.5 | 0.722 | 0.611 | +0.111 |

Against an overall accuracy of 0.667, a separation of 0.11-0.22 means confidence is close to
uninformative about correctness.

**The best temperature halves the discrimination.** T=2.5 lowers ECE mostly by flattening everything
toward the base rate, buying calibration by destroying signal. Part of that ECE gain is the model
becoming less informative rather than more honest, which is why ECE must never be read without a
discrimination figure beside it.

## Consequence for the plan

Calibration was ranked first as the cheapest fix with the biggest win. **That was backwards.** A
post-hoc calibrator adjusts how confidence maps to probability; it cannot manufacture a confidence
signal that tracks correctness when the underlying scores do not rank. The ordering is now:

1. **A trained scoring head — a prerequisite, not an optimisation.** It must produce scores that
   discriminate before calibration means anything.
2. **Calibration** — cheap and effective *once* a ranking signal exists; the fitter is already built.
3. **A smaller base** — speed and cost axes, unchanged.

This also explains routing at 0.3333 and adequacy at 0.4167: on those families the base cannot
discriminate at all, so its confidence is noise being asked to behave like a probability.

Cost of this diagnosis: ten minutes, no compute. Cost of discovering it by building the calibration
path first: weeks.
