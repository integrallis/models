# Rejected 24% Hammer preparation audit

Status: **rejected before training**.

The called-to-remaining function filter removed duplicate-capability tools close to the function
selected by the positive source. Inspection of the retained rows found a second ambiguity class:
the remaining tool clearly matched the user request even though its wording differed enough from
the originally selected function to evade that comparison.

Examples include an email-validation request retaining `email_verifier` and `validate_email`, an
Instagram profile request retaining `get_user_info`, and a two-part sorting request retaining
`sort_numbers`. These rows were not sent to training.

The preparation pipeline now also compares each query with all remaining tool names, descriptions,
and schemas and rejects a row at binary lexical cosine 0.15 or greater. Because correctness takes
priority over preserving the earlier class percentage, the replacement split declares a 22%
no-call population that fits the smaller high-confidence source.
