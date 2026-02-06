import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from classifier_pipeline.phase3_sampling import (
    read_rows,
    select_focus_samples,
    select_random_samples,
    write_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create focused Phase 3 sample")
    parser.add_argument(
        "--input-path",
        default="reports/phase2/phase2_extraction.csv",
        help="Path to Phase 2 CSV",
    )
    parser.add_argument(
        "--output-path",
        default="reports/phase3/phase3_focus_sample.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=20,
        help="Total sample size",
    )
    parser.add_argument(
        "--flower-min",
        type=int,
        default=5,
        help="Minimum number of flower examples",
    )
    parser.add_argument(
        "--flower-max",
        type=int,
        default=10,
        help="Maximum number of flower examples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed",
    )
    parser.add_argument(
        "--mode",
        choices=["focus", "random"],
        default="focus",
        help="Sampling mode",
    )

    args = parser.parse_args()

    rows = read_rows(Path(args.input_path))
    if args.mode == "random":
        sample = select_random_samples(rows, total=args.total, seed=args.seed)
    else:
        sample = select_focus_samples(
            rows,
            total=args.total,
            flower_min=args.flower_min,
            flower_max=args.flower_max,
            seed=args.seed,
        )
    write_rows(Path(args.output_path), sample)
    print(f"rows_written={len(sample)}")


if __name__ == "__main__":
    main()
