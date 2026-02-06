import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from classifier_pipeline.phase2_extraction import (
    DEFAULT_CLASSIFIERS,
    DEFAULT_COLLECTIONS,
    DEFAULT_EXCLUDE_LANGS,
    DEFAULT_INCLUDE_LANGS,
    write_phase2_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Mandarin classifier extraction")
    parser.add_argument(
        "--output-path",
        default="reports/phase2/phase2_extraction.csv",
        help="Path for the output CSV",
    )
    parser.add_argument(
        "--rejected-output-path",
        default="reports/phase2/rejected_samples.csv",
        help="Path for rejected sample CSV",
    )
    parser.add_argument(
        "--rejected-sample-size",
        type=int,
        default=50,
        help="Number of rejected samples to store",
    )
    parser.add_argument(
        "--rejected-seed",
        type=int,
        default=13,
        help="Random seed for rejected sample selection",
    )
    parser.add_argument(
        "--classifiers",
        nargs="*",
        default=DEFAULT_CLASSIFIERS,
        help="Classifier tokens to include",
    )
    parser.add_argument(
        "--include-langs",
        nargs="*",
        default=list(DEFAULT_INCLUDE_LANGS),
        help="Language codes to include (defaults to zho)",
    )
    parser.add_argument(
        "--exclude-langs",
        nargs="*",
        default=list(DEFAULT_EXCLUDE_LANGS),
        help="Language codes to exclude (defaults to yue nan)",
    )
    parser.add_argument(
        "--include-collections",
        nargs="*",
        default=list(DEFAULT_COLLECTIONS),
        help="Collection names to include (defaults to Chinese)",
    )
    parser.add_argument(
        "--db-name",
        default=None,
        help="Childes-db version (defaults to current)",
    )

    args = parser.parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rejected_output_path = Path(args.rejected_output_path)
    rejected_output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = write_phase2_csv(
        output_path=str(output_path),
        classifiers=args.classifiers,
        include_langs=args.include_langs,
        exclude_langs=args.exclude_langs,
        include_collections=args.include_collections,
        rejected_output_path=str(rejected_output_path) if args.rejected_sample_size > 0 else None,
        rejected_sample_size=args.rejected_sample_size,
        rejected_seed=args.rejected_seed,
        db_name=args.db_name,
    )

    print(f"rows_written={rows_written}")


if __name__ == "__main__":
    main()
