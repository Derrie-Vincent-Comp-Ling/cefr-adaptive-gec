"""Extract source-target training pairs from W&I+LOCNESS M2 annotation files.

Parses M2 files, applies annotator-0 edits to reconstruct gold-corrected
targets, and writes source-target-cefr_band triples to JSONL.

Outputs:
    data/processed/train_pairs.jsonl  (from A/B/C.train M2)
    data/processed/dev_pairs.jsonl    (from A/B/C/N.dev M2)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, LOGS_DIR, SEED, ensure_dirs

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "extract_pairs.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

M2_ROOT = DATA_RAW_DIR / "wi_locness" / "wi+locness" / "m2"


# ---------------------------------------------------------------------------
# M2 parsing
# ---------------------------------------------------------------------------
def parse_m2(path: Path):
    """Yield (source_str, [edit_lines]) per sentence block in an M2 file."""
    if not path.exists():
        log.warning("File not found: %s", path)
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


def apply_edits(source: str, edit_lines: list[str]) -> str:
    """Apply annotator-0 edits to source tokens and return the corrected target."""
    tokens = source.split(" ")
    parsed = []
    for e in edit_lines:
        try:
            span, etype, corr, _req, _com, ann = e.split("|||")
        except ValueError:
            continue
        if int(ann) != 0:
            continue
        if etype == "noop":
            continue
        start, end = (int(x) for x in span.split())
        parsed.append((start, end, corr))

    # Apply right-to-left so earlier indices stay valid
    for start, end, corr in sorted(parsed, key=lambda x: (x[0], x[1]), reverse=True):
        repl = corr.split(" ") if corr else []
        tokens[start:end] = repl

    return " ".join(t for t in tokens if t != "")


# ---------------------------------------------------------------------------
# Pair extraction
# ---------------------------------------------------------------------------
def extract_pairs(m2_files: dict[str, str], split_label: str) -> list[dict]:
    """Extract source-target pairs from a set of M2 files.

    Parameters
    ----------
    m2_files : {cefr_band: filename} mapping
    split_label : 'train' or 'dev'

    Returns
    -------
    list of dicts with keys: source, target, cefr_band
    """
    pairs: list[dict] = []
    for cefr_band, fname in m2_files.items():
        path = M2_ROOT / fname
        count = 0
        for src, edit_lines in parse_m2(path):
            target = apply_edits(src, edit_lines)
            pairs.append({
                "source": src,
                "target": target,
                "cefr_band": cefr_band,
            })
            count += 1
        log.info("  %s (%s): %d pairs", fname, cefr_band, count)
    return pairs


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()
    log.info("seed=%d", SEED)

    # Verify M2 files exist
    if not M2_ROOT.exists():
        log.error(
            "M2 directory not found at %s. "
            "Please download and extract the BEA-2019 data first:\n"
            "  curl -L -o data/raw/wi+locness_v2.1.bea19.tar.gz "
            "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz\n"
            "  tar -xzf data/raw/wi+locness_v2.1.bea19.tar.gz -C data/raw/wi_locness/",
            M2_ROOT,
        )
        sys.exit(1)

    # ---- Train pairs (A/B/C) ----
    log.info("Extracting train pairs...")
    train_m2 = {
        "A": "A.train.gold.bea19.m2",
        "B": "B.train.gold.bea19.m2",
        "C": "C.train.gold.bea19.m2",
    }
    train_pairs = extract_pairs(train_m2, "train")
    write_jsonl(train_pairs, DATA_PROCESSED_DIR / "train_pairs.jsonl")

    # ---- Dev pairs (A/B/C/N) ----
    log.info("Extracting dev pairs...")
    dev_m2 = {
        "A": "A.dev.gold.bea19.m2",
        "B": "B.dev.gold.bea19.m2",
        "C": "C.dev.gold.bea19.m2",
        "N": "N.dev.gold.bea19.m2",
    }
    dev_pairs = extract_pairs(dev_m2, "dev")
    write_jsonl(dev_pairs, DATA_PROCESSED_DIR / "dev_pairs.jsonl")

    # ---- Summary ----
    print(f"\n=== Pair counts ===")
    print(f"  train_pairs.jsonl: {len(train_pairs)}")
    print(f"  dev_pairs.jsonl:   {len(dev_pairs)}")

    # CEFR breakdown
    from collections import Counter
    train_dist = Counter(r["cefr_band"] for r in train_pairs)
    dev_dist = Counter(r["cefr_band"] for r in dev_pairs)
    print(f"\n=== Train CEFR breakdown ===")
    for band in sorted(train_dist):
        print(f"  {band}: {train_dist[band]}")
    print(f"\n=== Dev CEFR breakdown ===")
    for band in sorted(dev_dist):
        print(f"  {band}: {dev_dist[band]}")

    # Show examples
    print(f"\n=== Sample train pairs ===")
    for p in train_pairs[:3]:
        print(json.dumps(p, indent=2, ensure_ascii=False))

    print(f"\n=== Sample dev pairs ===")
    for p in dev_pairs[:3]:
        print(json.dumps(p, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
