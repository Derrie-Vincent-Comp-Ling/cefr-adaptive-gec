"""Normalise W&I+LOCNESS and JFLEG into unified JSONL.

Schema (one JSON per line):
  id, dataset, split, cefr, source, target, references, edits

Outputs:
  data/processed/wi_locness.train.jsonl   (from ABC.train; cefr in {A,B,C})
  data/processed/wi_locness.dev.jsonl     (from ABCN.dev;  cefr in {A,B,C,N})
  data/processed/jfleg.dev.jsonl
  data/processed/jfleg.test.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, SEED, ensure_dirs


def parse_m2(path: Path):
    """Yield (source, [edit_lines]) for each sentence block in an M2 file."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        src, edits = None, []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                if src is not None:
                    yield src, edits
                src, edits = line[2:], []
            elif line.startswith("A "):
                edits.append(line[2:])
            elif line == "" and src is not None:
                yield src, edits
                src, edits = None, []
        if src is not None:
            yield src, edits


def apply_m2_edits(source: str, edit_lines: list[str]) -> tuple[str, list[dict]]:
    """Apply annotator-0 edits to reconstruct the gold target, return (target, edits_struct)."""
    tokens = source.split(" ")
    parsed = []
    for e in edit_lines:
        # format: "start end|||type|||correction|||REQUIRED|||-NONE-|||annotator_id"
        try:
            span, etype, corr, _req, _com, ann = e.split("|||")
        except ValueError:
            continue
        if int(ann) != 0:
            continue
        if etype == "noop":
            continue
        start, end = (int(x) for x in span.split())
        parsed.append({"start": start, "end": end, "type": etype, "correction": corr})

    # Apply from right to left so indices stay valid
    parsed_sorted = sorted(parsed, key=lambda x: (x["start"], x["end"]), reverse=True)
    out = list(tokens)
    for e in parsed_sorted:
        repl = e["correction"].split(" ") if e["correction"] else []
        out[e["start"]:e["end"]] = repl
    target = " ".join(t for t in out if t != "")
    return target, parsed


def infer_cefr(source_tokens_or_meta: str, fallback: str) -> str:
    return fallback  # ABC.train/ABCN.dev don't carry CEFR in the M2 line; we use per-file metadata


def normalise_wi(m2_path: Path, dataset_split: str, per_file_cefr: str | None):
    """per_file_cefr='auto' means read the CEFR from the ABCN.dev / ABC.train files via side-files.

    ABC.train and ABCN.dev don't expose CEFR per sentence in the M2 alone, so we derive
    per-example CEFR by parsing the individual level files and matching sources.
    """
    return list(parse_m2(m2_path))


def build_wi_jsonl(
    combined_m2: Path,
    level_m2s: dict[str, Path],
    out_path: Path,
    split_name: str,
) -> int:
    """Build a JSONL for a combined W&I file, tagging each example with CEFR by
    matching its source text against the per-level files."""
    src_to_cefr: dict[str, str] = {}
    for level, path in level_m2s.items():
        for src, _edits in parse_m2(path):
            src_to_cefr.setdefault(src, level)

    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, (src, edit_lines) in enumerate(parse_m2(combined_m2)):
            target, edits = apply_m2_edits(src, edit_lines)
            cefr = src_to_cefr.get(src, "UNK")
            rec = {
                "id": f"wi_{split_name}_{i:06d}",
                "dataset": "wi_locness",
                "split": split_name,
                "cefr": cefr,
                "source": src,
                "target": target,
                "references": [target],
                "edits": edits,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_jfleg_jsonl(jf_root: Path, split: str, out_path: Path) -> int:
    src = (jf_root / split / f"{split}.src").read_text(encoding="utf-8").splitlines()
    refs = [
        (jf_root / split / f"{split}.ref{i}").read_text(encoding="utf-8").splitlines()
        for i in range(4)
    ]
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, s in enumerate(src):
            references = [r[i] for r in refs if i < len(r)]
            target = references[0] if references else None
            rec = {
                "id": f"jfleg_{split}_{i:06d}",
                "dataset": "jfleg",
                "split": split,
                "cefr": None,
                "source": s,
                "target": target,
                "references": references,
                "edits": [],
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ensure_dirs()
    wi_m2_root = DATA_RAW_DIR / "wi_locness" / "wi+locness" / "m2"
    jf_root = DATA_RAW_DIR / "jfleg"

    # Per-level files for CEFR tagging
    train_levels = {
        "A": wi_m2_root / "A.train.gold.bea19.m2",
        "B": wi_m2_root / "B.train.gold.bea19.m2",
        "C": wi_m2_root / "C.train.gold.bea19.m2",
    }
    dev_levels = {
        "A": wi_m2_root / "A.dev.gold.bea19.m2",
        "B": wi_m2_root / "B.dev.gold.bea19.m2",
        "C": wi_m2_root / "C.dev.gold.bea19.m2",
        "N": wi_m2_root / "N.dev.gold.bea19.m2",
    }

    wi_train_out = DATA_PROCESSED_DIR / "wi_locness.train.jsonl"
    wi_dev_out = DATA_PROCESSED_DIR / "wi_locness.dev.jsonl"
    jf_dev_out = DATA_PROCESSED_DIR / "jfleg.dev.jsonl"
    jf_test_out = DATA_PROCESSED_DIR / "jfleg.test.jsonl"

    n_wi_train = build_wi_jsonl(
        wi_m2_root / "ABC.train.gold.bea19.m2", train_levels, wi_train_out, "train"
    )
    n_wi_dev = build_wi_jsonl(
        wi_m2_root / "ABCN.dev.gold.bea19.m2", dev_levels, wi_dev_out, "dev"
    )
    n_jf_dev = build_jfleg_jsonl(jf_root, "dev", jf_dev_out)
    n_jf_test = build_jfleg_jsonl(jf_root, "test", jf_test_out)

    print(f"seed={SEED}")
    print(f"wi_locness.train.jsonl  : {n_wi_train} examples -> {wi_train_out}")
    print(f"wi_locness.dev.jsonl    : {n_wi_dev} examples -> {wi_dev_out}")
    print(f"jfleg.dev.jsonl         : {n_jf_dev} examples -> {jf_dev_out}")
    print(f"jfleg.test.jsonl        : {n_jf_test} examples -> {jf_test_out}")

    # Show one sample from each for sanity
    for p in (wi_train_out, wi_dev_out, jf_dev_out, jf_test_out):
        with open(p, encoding="utf-8") as f:
            first = f.readline().strip()
        print(f"\n--- first line of {p.name} ---")
        print(first[:600])


if __name__ == "__main__":
    main()