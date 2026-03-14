"""List every unique SNOMED Dx code in Chapman .hea files (no wfdb import).
Usage: python scripts/list_dx_codes.py [--sample N]  (default: all files; --sample 5000 for quick run)"""
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WFDB_DIR = PROJECT_ROOT / "data" / "raw" / "chapman" / "WFDBRecords"


def extract_codes(text: str) -> list[str]:
    for line in text.splitlines():
        if "Dx:" in line:
            dx = line.split("Dx:")[-1].strip()
            return [p.strip() for p in dx.split(",") if p.strip()]
    return []


def main():
    if not WFDB_DIR.exists():
        print(f"Not found: {WFDB_DIR}")
        return
    hea_paths = sorted(WFDB_DIR.rglob("*.hea"))
    if "--sample" in sys.argv:
        idx = sys.argv.index("--sample")
        n = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 5000
        hea_paths = hea_paths[:n]
        print(f"(Sampling first {len(hea_paths)} .hea files)")
    all_codes: set[str] = set()
    code_counter: Counter = Counter()
    n_with_dx = n_without_dx = 0
    for p in hea_paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            n_without_dx += 1
            continue
        codes = extract_codes(text)
        if codes:
            n_with_dx += 1
            for c in codes:
                all_codes.add(c)
                code_counter[c] += 1
        else:
            n_without_dx += 1
    total = n_with_dx + n_without_dx
    print("Record counts:")
    print(f"  With Dx line:   {n_with_dx} ({100*n_with_dx/total:.1f}%)")
    print(f"  Without Dx:     {n_without_dx} ({100*n_without_dx/total:.1f}%)")
    print()
    print("All unique SNOMED codes (sorted by count):")
    print("  Code        Count")
    for c in sorted(all_codes, key=lambda x: -code_counter[x]):
        print(f"  {c}  {code_counter[c]}")

    # Map to our 4 meta-classes (from src/data/io.py CODE_GROUPS)
    CODE_GROUPS = [
        ["164889003", "164890007"],   # AF (AFIB, AFL)
        ["426761007", "713422000", "233896004", "233897008", "195101003", "427172004"],  # SVT
        ["426177001"],                # Sinus Brady
        ["426783006", "427393009"],   # Sinus Rhythm (+ Sinus Irregularity)
    ]
    CLASS_NAMES = ["AF", "SVT", "Sinus Brady", "Sinus Rhythm"]
    print()
    print("Codes in our 4 meta-classes (present in dataset):")
    for name, group in zip(CLASS_NAMES, CODE_GROUPS):
        present = [(c, code_counter[c]) for c in group if c in all_codes]
        missing = [c for c in group if c not in all_codes]
        print(f"  {name}: {present}")
        if missing:
            print(f"    (not seen in this run): {missing}")


if __name__ == "__main__":
    main()
