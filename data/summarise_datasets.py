"""Summarise raw W&I+LOCNESS and JFLEG datasets."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW_DIR, LOGS_DIR, SEED, ensure_dirs


def parse_m2(path):
    entries = []
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as f:
        src, edits = None, []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                if src is not None:
                    entries.append({"source": src, "edits": edits})
                src, edits = line[2:], []
            elif line.startswith("A "):
                edits.append(line[2:])
            elif line == "" and src is not None:
                entries.append({"source": src, "edits": edits})
                src, edits = None, []
        if src is not None:
            entries.append({"source": src, "edits": edits})
    return entries


def read_lines(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def main():
    ensure_dirs()
    summary = {"seed": SEED, "datasets": {}}

    wi_root = DATA_RAW_DIR / "wi_locness" / "wi+locness" / "m2"
    wi_files = {
        "A.train": "A.train.gold.bea19.m2",
        "B.train": "B.train.gold.bea19.m2",
        "C.train": "C.train.gold.bea19.m2",
        "ABC.train": "ABC.train.gold.bea19.m2",
        "A.dev": "A.dev.gold.bea19.m2",
        "B.dev": "B.dev.gold.bea19.m2",
        "C.dev": "C.dev.gold.bea19.m2",
        "N.dev": "N.dev.gold.bea19.m2",
        "ABCN.dev": "ABCN.dev.gold.bea19.m2",
    }
    wi_splits, wi_samples = {}, {}
    for split, fname in wi_files.items():
        entries = parse_m2(wi_root / fname)
        wi_splits[split] = len(entries)
        wi_samples[split] = entries[:2]
    summary["datasets"]["wi_locness"] = {
        "source": "Cambridge BEA-2019 v2.1 (M2)",
        "path": str(wi_root),
        "splits": wi_splits,
    }

    jf_root = DATA_RAW_DIR / "jfleg"
    jf_splits, jf_samples = {}, {}
    for split in ("dev", "test"):
        src = read_lines(jf_root / split / f"{split}.src")
        refs = [read_lines(jf_root / split / f"{split}.ref{i}") for i in range(4)]
        jf_splits[split] = len(src)
        jf_samples[split] = [
            {"source": src[i], "references": [r[i] for r in refs if i < len(r)]}
            for i in range(min(2, len(src)))
        ]
    summary["datasets"]["jfleg"] = {
        "source": "keisks/jfleg (GitHub)",
        "path": str(jf_root),
        "splits": jf_splits,
    }

    out_path = LOGS_DIR / "dataset_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== W&I+LOCNESS split sizes ===")
    for k, v in wi_splits.items():
        print(f"  {k}: {v}")
    print("\n=== JFLEG split sizes ===")
    for k, v in jf_splits.items():
        print(f"  {k}: {v}")
    print(f"\nSummary written to {out_path}")

    print("\n=== W&I+LOCNESS samples (A.dev) ===")
    print(json.dumps(wi_samples.get("A.dev", [])[:2], indent=2)[:1500])
    print("\n=== JFLEG samples (dev) ===")
    print(json.dumps(jf_samples.get("dev", [])[:2], indent=2)[:1500])


if __name__ == "__main__":
    main()