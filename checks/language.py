"""Surface language statistics of the training answers and of every variant's greedy answers.

    python checks/language.py [--verify runs/verify.json]

What it does: for data/train.jsonl (the `output` field), data/heldout.jsonl (the reference
answers the variants are scored against) and the greedy generations of each variant from the
verify run named in runs/verify.json (logs/verify_generations_<run_id>_gpu*.jsonl, only the
lines generated with the run's batch size, so the batch-1 comparison lines are not counted
twice), it counts:
- loop_share: share of texts in which some word 3-gram occurs more than 2 times;
- repeat_pair_share: share of texts with "X және X" or "X, X" (the same word twice);
- kyrgyz_per_100_words: matches of \\w+(дун|дүн|тун|түн|нун|нүн)\\b, \\bбул\\b, \\bтил\\b,
  маалымат\\w* per 100 words (each pattern separately and summed);
- no_space_after_period_share: share of texts with "." directly followed by an upper-case letter.
Numbers only, no thresholds and no interpretation.
Reads: data/train.jsonl, data/heldout.jsonl, runs/verify.json, logs/verify_generations_*.jsonl.
Writes: runs/checks/language.json (+ language_<run_id>.json).
Task part: Part 3 (silent failures) and the verdict inputs.
"""

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from common import (DATA, LOGS, RUNS, log, new_run_id, read_json, read_jsonl, rel,  # noqa: E402
                    run_record, write_json)

OUT_DIR = RUNS / "checks"
WORD = re.compile(r"\w+")
REPEAT_PAIR = re.compile(r"\b(\w+)(?:\s+және\s+|,\s*)\1\b", re.IGNORECASE)
KYRGYZ = {
    "genitive_dun_tun_nun": re.compile(r"\w+(?:дун|дүн|тун|түн|нун|нүн)\b", re.IGNORECASE),
    "bul": re.compile(r"\bбул\b", re.IGNORECASE),
    "til": re.compile(r"\bтил\b", re.IGNORECASE),
    "maalymat": re.compile(r"маалымат\w*", re.IGNORECASE),
}
NO_SPACE_AFTER_PERIOD = re.compile(r"\.[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ]")


def has_loop3(text):
    """Some word 3-gram occurs more than 2 times."""
    words = text.split()
    grams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    return any(count > 2 for count in grams.values())


def stats(texts):
    n = len(texts)
    words = sum(len(WORD.findall(t)) for t in texts)
    kyrgyz = {k: sum(len(p.findall(t)) for t in texts) for k, p in KYRGYZ.items()}
    per100 = {k: (100 * v / words if words else None) for k, v in kyrgyz.items()}
    total = sum(kyrgyz.values())
    return {
        "texts": n,
        "words": words,
        "loop_share": sum(has_loop3(t) for t in texts) / n if n else None,
        "repeat_pair_share": sum(bool(REPEAT_PAIR.search(t)) for t in texts) / n if n else None,
        "repeat_pair_matches": sum(len(REPEAT_PAIR.findall(t)) for t in texts),
        "kyrgyz_matches": kyrgyz,
        "kyrgyz_per_100_words": per100,
        "kyrgyz_per_100_words_total": 100 * total / words if words else None,
        "no_space_after_period_share":
            sum(bool(NO_SPACE_AFTER_PERIOD.search(t)) for t in texts) / n if n else None,
        "no_space_after_period_matches": sum(len(NO_SPACE_AFTER_PERIOD.findall(t)) for t in texts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", default=str(RUNS / "verify.json"))
    args = parser.parse_args()

    run_id = new_run_id()
    result = {"check": "language", "run_id": run_id,
              "definitions": {
                  "loop_share": "share of texts where a word 3-gram (whitespace split) occurs > 2 times",
                  "repeat_pair_share": f"share of texts matching {REPEAT_PAIR.pattern}",
                  "kyrgyz_per_100_words": {k: p.pattern for k, p in KYRGYZ.items()},
                  "words": f"matches of {WORD.pattern}",
                  "no_space_after_period_share": f"share of texts matching {NO_SPACE_AFTER_PERIOD.pattern}"},
              "sources": {}, "groups": {}}

    for name in ("train", "heldout"):
        path = DATA / f"{name}.jsonl"
        result["groups"][f"{name}_reference"] = stats([r["output"] for r in read_jsonl(path)])
        result["sources"][f"{name}_reference"] = rel(path)

    verify = read_json(args.verify)
    assert verify, f"{args.verify} missing: run verify first"
    batch = (verify.get("generation") or {}).get("batch")
    files = sorted(LOGS.glob(f"verify_generations_{verify['run_id']}_gpu*.jsonl"))
    assert files, f"no logs/verify_generations_{verify['run_id']}_gpu*.jsonl"
    result.update({"verify_run_id": verify["run_id"], "verify_limit": verify.get("limit"),
                   "generation_batch_counted": batch, "generation_files": [rel(f) for f in files]})
    per_variant = {}
    for path in files:
        for line in read_jsonl(path):
            if line.get("run_id") == verify["run_id"] and line.get("batch") == batch:
                per_variant.setdefault(line["variant"], {})[line["row"]] = line["generation"]
    for variant, rows in sorted(per_variant.items()):
        result["groups"][f"generations_{variant}"] = stats([rows[r] for r in sorted(rows)])
        result["sources"][f"generations_{variant}"] = [rel(f) for f in files]

    result.update(run_record())
    write_json(OUT_DIR / f"language_{run_id}.json", result)
    shutil.copyfile(OUT_DIR / f"language_{run_id}.json", OUT_DIR / "language.json")

    print(f"\n{'group':28s} {'texts':>5s} {'loop3':>7s} {'X,X':>7s} {'kyrg/100w':>10s} {'.Б':>7s}")
    for name, g in result["groups"].items():
        print(f"{name:28s} {g['texts']:5d} {g['loop_share']:7.3f} {g['repeat_pair_share']:7.3f} "
              f"{g['kyrgyz_per_100_words_total']:10.4f} {g['no_space_after_period_share']:7.3f}")
    log(f"[language] -> {rel(OUT_DIR / 'language.json')}")


if __name__ == "__main__":
    main()
