import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from classifier_pipeline.phase1_inventory import (
    DEFAULT_LANGUAGE_FILTER,
    run_phase1_inventory,
    run_phase1_inventory_db,
)


DEFAULT_CLASSIFIERS = ["个", "条", "张", "本", "只"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 CHILDES/TalkBank inventory")
    parser.add_argument(
        "--output-dir",
        default="reports/phase1",
        help="Directory for output reports",
    )
    parser.add_argument(
        "--source",
        choices=["childes-db", "talkbank"],
        default="childes-db",
        help="Data source to use",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=None,
        help="Optional list of sections to include (TalkBank only)",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Language codes to include (childes-db only)",
    )
    parser.add_argument(
        "--db-name",
        default=None,
        help="Childes-db version (defaults to current)",
    )
    parser.add_argument(
        "--classifiers",
        nargs="*",
        default=DEFAULT_CLASSIFIERS,
        help="Classifier tokens to count",
    )

    args = parser.parse_args()

    if args.source == "talkbank":
        sections = set(args.sections) if args.sections else None
        rows = run_phase1_inventory(
            output_dir=args.output_dir,
            classifiers=args.classifiers,
            sections=sections,
        )
    else:
        languages = args.languages or list(DEFAULT_LANGUAGE_FILTER)
        rows = run_phase1_inventory_db(
            output_dir=args.output_dir,
            classifiers=args.classifiers,
            languages=languages,
            db_name=args.db_name,
        )

    print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, ensure_ascii=False))


if __name__ == "__main__":
    main()
