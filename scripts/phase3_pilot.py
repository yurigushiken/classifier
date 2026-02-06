import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from classifier_pipeline.phase3_pilot import run_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3B pilot run (first 20 rows)")
    parser.add_argument(
        "--input-path",
        default="reports/phase2/phase2_extraction.csv",
        help="Path to Phase 2 CSV",
    )
    parser.add_argument(
        "--output-path",
        default="reports/phase3/phase3_pilot_results.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of rows to process",
    )
    parser.add_argument(
        "--env-path",
        default=".env",
        help="Path to .env file",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Override provider (openrouter or lmstudio)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override base URL",
    )

    args = parser.parse_args()

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        provider = os.environ.get("LLM_PROVIDER", "openrouter").lower()
        if provider == "lmstudio":
            os.environ["LM_STUDIO_MODEL"] = args.model
        else:
            os.environ["OPENROUTER_MODEL"] = args.model
    if args.base_url:
        provider = os.environ.get("LLM_PROVIDER", "openrouter").lower()
        if provider == "lmstudio":
            os.environ["LM_STUDIO_BASE_URL"] = args.base_url
        else:
            os.environ["OPENROUTER_BASE_URL"] = args.base_url

    rows_written = run_pilot(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        limit=args.limit,
        env_path=Path(args.env_path),
    )

    print(f"rows_written={rows_written}")


if __name__ == "__main__":
    main()
