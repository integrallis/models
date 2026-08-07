#!/usr/bin/env python3
"""Pick the classifier's unclassified threshold from measurements.

The threshold cannot separate correct classifications from incorrect ones: the
nearest *wrong* training neighbour routinely sits closer than the median correct
one, so any cut that removed mistakes would remove far more right answers. What
it can do is reject queries that resemble nothing in the corpus, so that an
out-of-domain request routes on cost and latency instead of being forced into
whichever task happens to be least distant.

So this reports two distributions against the same index:

  in-domain   held-out prompts, which should mostly clear the threshold
  out-of-domain  text no task covers, which should mostly fall below it

and prints, for a range of candidate thresholds, how much of each it accepts.

Requires the same llama.cpp binary and models as bakeoff.py.

Run:  python3 calibrate_threshold.py --model embeddinggemma-300m
"""

import argparse
import json
import pathlib

from bakeoff import CANDIDATES, embed, load_corpus, unit

HERE = pathlib.Path(__file__).parent

# Text a task classifier should decline rather than force into a task. Deliberately
# varied: gibberish alone would make any threshold look good.
OUT_OF_DOMAIN = [
    "asdkjfh qwoieur zxcvbnm poiuyt",
    "-----------------",
    "1 2 3 4 5 6 7 8 9 10 11 12 13",
    "Ph'nglui mglw'nafh Cthulhu R'lyeh wgah'nagl fhtagn",
    "the the the the the the the the",
    "こんにちは、元気ですか",
    "My knee has been hurting since Tuesday and it clicks when I bend it.",
    "I would like to file a complaint about my neighbour's fence.",
    "Book me a haircut for Thursday afternoon at the place on Third Street.",
    "The mitochondrion is the powerhouse of the cell.",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "What is your refund policy for orders placed over thirty days ago?",
    "Turn the living room lights off and set the thermostat to 68.",
    "Remind me to take the bins out on Sunday evening.",
    "https://example.com/some/path?query=1",
    "{\"status\": 200, \"body\": null}",
]


def nearest(query, train_vectors, train_tasks):
    best, best_task = -2.0, None
    for index, candidate in enumerate(train_vectors):
        score = sum(a * b for a, b in zip(query, candidate))
        if score > best:
            best, best_task = score, train_tasks[index]
    return best, best_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="embeddinggemma-300m")
    parser.add_argument("--out", default=str(HERE / "threshold-calibration.json"))
    args = parser.parse_args()

    path, pooling = CANDIDATES[args.model]
    train, evaluation = load_corpus()
    train_tasks = [t for t, _ in train]

    prompts = [p for _, p in train] + [p for _, p in evaluation] + OUT_OF_DOMAIN
    vectors = [unit(v) for v in embed(path, pooling, prompts)]
    train_vectors = vectors[: len(train)]
    eval_vectors = vectors[len(train) : len(train) + len(evaluation)]
    alien_vectors = vectors[len(train) + len(evaluation) :]

    in_domain = []
    correct_flags = []
    for index, (want, _prompt) in enumerate(evaluation):
        similarity, predicted = nearest(eval_vectors[index], train_vectors, train_tasks)
        in_domain.append(similarity)
        correct_flags.append(predicted == want)

    out_domain = []
    for index, text in enumerate(OUT_OF_DOMAIN):
        similarity, predicted = nearest(alien_vectors[index], train_vectors, train_tasks)
        out_domain.append(similarity)
        print("  out-of-domain sim=%.3f -> %-14s %s" % (similarity, predicted, text[:60]))

    print("\nmodel: %s" % args.model)
    print("in-domain cosine    min=%.3f p05=%.3f p50=%.3f"
          % (min(in_domain), sorted(in_domain)[int(len(in_domain) * 0.05)],
             sorted(in_domain)[len(in_domain) // 2]))
    print("out-of-domain cosine max=%.3f p50=%.3f"
          % (max(out_domain), sorted(out_domain)[len(out_domain) // 2]))

    print("\n threshold  keeps-in-domain  rejects-out-of-domain  accuracy-of-kept")
    rows = []
    for step in range(0, 71, 5):
        threshold = step / 100.0
        kept = [i for i, s in enumerate(in_domain) if s >= threshold]
        rejected = sum(1 for s in out_domain if s < threshold)
        accuracy = (
            sum(1 for i in kept if correct_flags[i]) / len(kept) if kept else float("nan"))
        rows.append({"threshold": threshold,
                     "kept_in_domain": len(kept) / len(in_domain),
                     "rejected_out_of_domain": rejected / len(out_domain),
                     "accuracy_of_kept": accuracy})
        print("     %.2f        %6.1f%%              %6.1f%%            %6.1f%%"
              % (threshold, 100 * len(kept) / len(in_domain),
                 100 * rejected / len(out_domain), 100 * accuracy))

    pathlib.Path(args.out).write_text(json.dumps(
        {"model": args.model,
         "in_domain": {"n": len(in_domain), "min": min(in_domain), "p50":
                       sorted(in_domain)[len(in_domain) // 2]},
         "out_of_domain": {"n": len(out_domain), "max": max(out_domain)},
         "sweep": rows}, indent=2) + "\n")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
