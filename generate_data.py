"""Generate synthetic arithmetic examples for the tiny model."""

import argparse
from pathlib import Path
import math

SUPPORTED_OPERATORS = ["+", "-", "*", "//"]


def calculate(a: int, operator: str, b: int) -> int | float:
    # Return type is adjusted to handle division results if they were floats, 
    # but since the original used integer division (//), we stick to integers for consistency.
    if operator == "+":
        return a + b
    if operator == "-":
        return a - b
    if operator == "*":
        return a * b
    if operator == "//":
        if b == 0:
            # This should not happen if the loops are correctly set up, but kept for safety.
            raise ValueError("Integer division by zero is undefined.")
        return a // b
    raise ValueError(f"Unsupported operator: {operator}")


def generate_examples(max_number: int) -> list[str]:
    """Generates all possible arithmetic examples for numbers up to max_number."""
    lines: list[str] = []

    # Iterate over all pairs of operands (a, b) from 0 up to max_number
    for a in range(max_number + 1):
        for b in range(max_number + 1):
            for operator in SUPPORTED_OPERATORS:
                try:
                    # Skip division by zero cases
                    if operator == "//" and b == 0:
                        continue

                    answer = calculate(a, operator, b)
                    # Every example uses the same prompt/answer format.
                    lines.append(f"Q: {a} {operator} {b}\nA: {answer}\n")
                except ValueError as e:
                    print(f"Skipping example due to error: {e}") # Should be rare given checks

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Removed --count, as full coverage is now deterministic based on max_number
    parser.add_argument("--max-number", type=int, default=99, help="largest random operand (inclusive). Full coverage up to 100 will use 99.")
    parser.add_argument(
        "--operators",
        nargs="+",
        choices=SUPPORTED_OPERATORS,
        default=SUPPORTED_OPERATORS, # Defaulting to all supported operators for full coverage
        help="operators to include.",
    )
    parser.add_argument("--output", default="math_data.txt", help="output text file")
    args = parser.parse_args()

    if args.max_number < 1:
        parser.error("--max-number must be at least 1")
    # We ensure the max number is within a reasonable range if we are generating all combinations
    elif args.max_number > 100:
        print("Warning: Limiting --max-number to 99 for comprehensive coverage.")
        args.max_number = 99

    # Generate new examples
    new_lines = generate_examples(args.max_number)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_lines: list[str] = []
    
    # Append existing content if the file exists
    if output_path.exists():
        print(f"Appending to existing data in: {output_path}")
        try:
            existing_content = output_path.read_text(encoding="utf-8")
            # Split by lines and filter out potential empty lines from previous runs
            all_lines = [line for line in existing_content.splitlines() if line.strip()]
        except Exception as e:
            print(f"Error reading existing file content: {e}. Starting with new data.")


    # Combine new and old data, then write back (new examples first to meet ordering requirement)
    if all_lines or new_lines:
        final_content = "\n\n".join(new_lines + all_lines) + "\n"
        output_path.write_text(final_content, encoding="utf-8")

    print(f"Successfully generated and appended {len(new_lines):,} examples to: {output_path.resolve()}")
    print("Example records:")
    if new_lines:
        print("\n".join(new_lines[:3]))
    else:
        print("(No new examples were generated)")
    print(f"Operators used: {', '.join(args.operators)}")
    print(f"Operand range covered: 0 to {args.max_number}")


if __name__ == "__main__":
    main()