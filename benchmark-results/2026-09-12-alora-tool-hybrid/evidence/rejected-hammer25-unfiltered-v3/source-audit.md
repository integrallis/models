# Rejected unfiltered Hammer preparation audit

Status: **rejected before training**.

The Hammer irrelevance corpus has a sounder construction than the random no-call rows: it starts
from an xLAM positive and removes the known-correct function. The raw corpus still cannot be used
blindly because some source prompts have multiple requested calls or more than one tool capable of
the same operation. Removing one labelled function does not make every remaining function
irrelevant.

Examples found before training:

- Hammer row 2,847 removes the selected imperial-BMI function but leaves another `bmi` function
  that explicitly accepts imperial units. Its positive source also requests a horoscope.
- Row 4,651 removes `euclidean_distance` but leaves `calculate_distance`, whose description and
  schema can satisfy the same request.
- Row 3,223 removes the string-ID order lookup while retaining an integer-only order lookup. This is
  a useful near miss, but its called-to-remaining tool cosine is close to the ambiguous cases and is
  conservatively excluded rather than guessed into training.
- Row 2,581 is a valid state-scope near miss—Kentucky requested, Georgia tool offered—but it is also
  conservatively excluded by the same fixed policy.

The replacement policy requires a traceable, valid, single-call positive example and rejects a
masked row when any remaining function has binary lexical-schema cosine of 0.45 or greater to the
called function. That threshold excludes every known duplicate-capability case above. It leaves
3,180 candidates after schema and contamination checks, 70 fewer than the planned 25% population;
the next preparation therefore uses a declared 24% no-call fraction instead of weakening the
ambiguity filter.
