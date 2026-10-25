"""Generate synthetic arithmetic training data for the tiny character model.

Every example uses the exact format that ``ask.py --qa`` sends to the model::

    Q: 12 + 7
    A: 19

(a blank line separates examples).

Usage
-----
    python generate_data.py                        # all ops, 0..99, shuffled
    python generate_data.py --max-number 50 --operators + -
    python generate_data.py --append               # add only NEW problems
    python generate_data.py --output my_math.txt   # custom file

By default the file lands in ``data_to_train_on/math_data.txt`` next to this
script, which is the folder ``train.py`` auto-detects. Any other .txt files in
that folder are trained on too — move math_data.txt elsewhere (or pass
``--data`` to train.py) if you want a math-only run.

Design notes
------------
* Idempotent by default: running twice with the same arguments rewrites the
  same file instead of stacking duplicates.
* ``--append`` parses the existing file and adds only problems it does not
  already contain; it never re-splits or reorders what is there.
* Examples are SHUFFLED by default (seeded). train.py holds out the *tail* of
  the corpus for validation; with sorted data that tail would be one narrow
  slice of problems (a = 95..99). Shuffling makes validation a random sample
  of problems the model never trained on — the only honest way to measure
  whether it learned arithmetic or just memorised the table.
* Integer division uses ``//`` (floor), exactly like Python.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

SUPPORTED_OPERATORS = ["+", "-", "*", "//"]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "data_to_train_on" / "math_data.txt"
MAX_ALLOWED = 999            # 1000x1000x4 ≈ 4M examples ≈ 50 MB — plenty
WARN_ABOVE = 300             # beyond this the file gets big; just tell the user

_Q_LINE = re.compile(r"^Q:\s*(-?\d+)\s*(//|[+\-*])\s*(-?\d+)\s*$", re.MULTILINE)


def calculate(a: int, operator: str, b: int) -> int:
    """Evaluate ``a <operator> b`` with integer semantics."""
    if operator == "+":
        return a + b
    if operator == "-":
        return a - b
    if operator == "*":
        return a * b
    if operator == "//":
        if b == 0:
            raise ZeroDivisionError("integer division by zero")
        return a // b
    raise ValueError(f"Unsupported operator: {operator}")


def format_example(a: int, operator: str, b: int) -> str:
    """One example block (no trailing blank line)."""
    return f"Q: {a} {operator} {b}\nA: {calculate(a, operator, b)}"


def generate_problems(max_number: int, operators: list[str]) -> list[tuple[int, str, int]]:
    """Every (a, op, b) with 0 <= a, b <= max_number; division by zero skipped."""
    problems = []
    for a in range(max_number + 1):
        for b in range(max_number + 1):
            for op in operators:
                if op == "//" and b == 0:
                    continue
                problems.append((a, op, b))
    return problems


def existing_problems(text: str) -> set[tuple[int, str, int]]:
    """Problems already present in an existing data file."""
    return {(int(a), op, int(b)) for a, op, b in _Q_LINE.findall(text)}


def build_text(problems: list[tuple[int, str, int]]) -> str:
    return "\n\n".join(format_example(*p) for p in problems) + "\n"


def atomic_write(path: Path, text: str, *, append: bool) -> None:
    """Write via a temp file + rename so a crash can't truncate the data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        old = path.read_text(encoding="utf-8")
        text = old.rstrip("\n") + "\n\n" + text if old.strip() else text
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-number", type=int, default=99,
                    help="largest operand, inclusive (default 99 → 0..99).")
    ap.add_argument("--operators", nargs="+", choices=SUPPORTED_OPERATORS,
                    default=SUPPORTED_OPERATORS, metavar="OP",
                    help=f"operators to include (default: all). "
                         f"Choices: {' '.join(SUPPORTED_OPERATORS)}")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help=f"output file (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--append", action="store_true",
                    help="add only problems not already in the file "
                         "(default: overwrite, so re-running never duplicates)")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                    help="keep sorted order (NOT recommended: validation then "
                         "covers only the last few operands)")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed")
    args = ap.parse_args()

    if not 1 <= args.max_number <= MAX_ALLOWED:
        ap.error(f"--max-number must be between 1 and {MAX_ALLOWED}")
    if args.max_number > WARN_ABOVE:
        print(f"note: --max-number {args.max_number} makes a large file "
              f"(~{(args.max_number + 1) ** 2 * len(args.operators) * 14 / 1e6:.0f} MB).")

    operators = list(dict.fromkeys(args.operators))      # de-dupe, keep order
    problems = generate_problems(args.max_number, operators)

    out = Path(args.output).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out

    skipped = 0
    if args.append and out.exists():
        have = existing_problems(out.read_text(encoding="utf-8", errors="replace"))
        fresh = [p for p in problems if p not in have]
        skipped = len(problems) - len(fresh)
        problems = fresh
        print(f"Appending to {out} ({len(have):,} problems already present).")

    if args.shuffle:
        random.Random(args.seed).shuffle(problems)

    if not problems:
        print(f"Nothing to add — all {skipped:,} problems are already in {out}.")
        return 0

    atomic_write(out, build_text(problems), append=args.append)

    print(f"Wrote {len(problems):,} examples to: {out}"
          + (f"  (skipped {skipped:,} duplicates)" if skipped else ""))
    print(f"Operators: {' '.join(operators)}   operands: 0..{args.max_number}"
          f"   order: {'shuffled (seed %d)' % args.seed if args.shuffle else 'sorted'}")
    print("Sample:")
    print("\n\n".join(format_example(*p) for p in problems[:3]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
