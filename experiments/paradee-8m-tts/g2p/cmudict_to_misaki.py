#!/usr/bin/env python3
"""Convert CMUdict (BSD-2-Clause, Carnegie Mellon University) to misaki's American IPA spelling,
keep the words misaki's own dictionaries lack, and measure the conversion where both agree.

The map is calibrated on the overlap: every CMUdict word that misaki's gold dictionary also has is
converted and compared, and the report records exact agreement and phoneme error rate, so the
quality of the tier is a number and not a belief. Stress marks go immediately before the vowel, as
misaki writes them; the American flap is written T where a /t/ sits between a vowel and an
unstressed vowel, as misaki does.

  python3 cmudict_to_misaki.py cmudict.dict MISAKI_DATA_DIR out/cmudict-us.json > out/cmudict-report.json
"""
import json, re, sys

VOWELS = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "W", "AY": "I", "EH": "ɛ", "ER": "ɜɹ",
          "EY": "A", "IH": "ɪ", "IY": "i", "OW": "O", "OY": "Y", "UH": "ʊ", "UW": "u"}
CONSONANTS = {"B": "b", "CH": "ʧ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "ʤ", "K": "k",
              "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t",
              "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}
STRESS = {"1": "ˈ", "2": "ˌ", "0": ""}


OBSTRUENTS = {"B", "CH", "D", "DH", "F", "G", "JH", "K", "P", "S", "SH", "T", "TH", "V", "Z", "ZH"}
SIBILANTS = {"S", "Z", "SH", "ZH", "CH", "JH"}


def split(p):
    return (p[:-1], p[-1]) if p[-1].isdigit() else (p, None)


def convert(arpabet):
    phones = arpabet.split()
    out = []
    for i, p in enumerate(phones):
        base, digit = split(p)
        prev = split(phones[i - 1])[0] if i > 0 else None
        nxt = split(phones[i + 1])[0] if i + 1 < len(phones) else None
        nxt_digit = split(phones[i + 1])[1] if i + 1 < len(phones) else None
        if base in VOWELS:
            ipa = VOWELS[base]
            if base == "AH" and digit == "0":
                # misaki's syllabic consonants: ᵊl after any consonant, ᵊn after an obstruent.
                if nxt == "L" and prev in CONSONANTS and (i + 2 >= len(phones) or split(phones[i + 2])[0] in CONSONANTS):
                    ipa = "ᵊ"
                elif nxt == "N" and prev in OBSTRUENTS and i + 2 < len(phones) and split(phones[i + 2])[0] in CONSONANTS:
                    ipa = "ᵊ"
                else:
                    ipa = "ə"
            elif base == "ER" and digit == "0":
                ipa = "əɹ"
            elif base == "IH" and digit == "0" and i + 2 == len(phones) and nxt in ("Z", "D") and prev in (SIBILANTS | {"T", "D"}):
                ipa = "ᵻ"  # the -es / -ed endings
            out.append(STRESS.get(digit, "") + ipa)
        elif base in CONSONANTS:
            ipa = CONSONANTS[base]
            if base == "T" and prev in VOWELS and nxt in VOWELS and nxt_digit == "0":
                ipa = "ɾ"  # the American flap, written ɾ in misaki's dictionaries (T at output)
            out.append(ipa)
        else:
            return None
    return "".join(out)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main(dict_path, misaki_dir, out_path):
    gold = json.load(open(f"{misaki_dir}/us_gold.json", encoding="utf-8"))
    silver = json.load(open(f"{misaki_dir}/us_silver.json", encoding="utf-8"))
    known = {w.lower() for w in gold} | {w.lower() for w in silver}
    gold_default = {w.lower(): (v if isinstance(v, str) else v.get("DEFAULT")) for w, v in gold.items()}
    entries, overlap_exact, overlap_edits, overlap_total, overlap_n, dropped = {}, 0, 0, 0, 0, 0
    sample_disagreements = []
    for line in open(dict_path, encoding="latin-1"):
        line = line.strip()
        if not line or line.startswith(";;;"):
            continue
        word, _, pron = line.partition(" ")
        pron = pron.split("#")[0].strip()
        if re.search(r"\(\d+\)$", word):
            continue  # alternate pronunciations: keep the first only
        w = word.lower()
        if not re.fullmatch(r"[a-z'.\-]+", w):
            continue
        ipa = convert(pron)
        if ipa is None:
            dropped += 1
            continue
        ref = gold_default.get(w)
        if ref:
            overlap_n += 1
            overlap_exact += ipa == ref
            d = levenshtein(ipa, ref); overlap_edits += d; overlap_total += len(ref)
            if ipa != ref and len(sample_disagreements) < 40:
                sample_disagreements.append({"word": w, "cmudict": ipa, "misaki": ref})
        if w not in known:
            entries[w] = ipa
    json.dump(entries, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    report = {"source": "cmudict.dict (cmusphinx/cmudict master), BSD-2-Clause, Carnegie Mellon University",
              "entriesWritten": len(entries), "words_misaki_already_has": overlap_n, "dropped_unmappable": dropped,
              "agreementWithMisakiGoldOnOverlap": {"exact": overlap_exact / overlap_n, "phonemeErrorRate": overlap_edits / overlap_total, "words": overlap_n},
              "sampleDisagreements": sample_disagreements}
    json.dump(report, sys.stdout, ensure_ascii=False, indent=1); print()
    print(f"wrote {len(entries)} entries; overlap {overlap_n}: exact {overlap_exact/overlap_n:.3f}, PER {overlap_edits/overlap_total:.3f}; dropped {dropped}", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:4])
