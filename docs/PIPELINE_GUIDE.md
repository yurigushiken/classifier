# Pipeline Guide

## Environment Setup
- Python 3.12+
- Install dependencies: `python -m pip install -r requirements.txt` (or use pyproject.toml)
- Ensure `.env` contains API keys (do not commit):
  - OPEN_ROUTER_API_KEY=<key>
  - Optional: OPENROUTER_SITE_URL, OPENROUTER_APP_NAME

## Phase 1: Data Inventory (childes-db)
Command:
- `python scripts\phase1_inventory.py --source childes-db --output-dir reports\phase1`

Outputs:
- reports/phase1/phase1_corpus_stats.csv
- reports/phase1/phase1_corpus_stats.json
- reports/phase1/phase1_summary.md

## Phase 2: Deterministic Extraction
Command:
- `python scripts\phase2_extraction.py`

Outputs:
- reports/phase2/phase2_extraction.csv
- reports/phase2/rejected_samples.csv

Key filters:
- Monolingual Mandarin: collection = Chinese
- Classifier list: full list from reference headers
- Determiner/Number filter based on POS tags

## Phase 3: LLM Annotation

OpenRouter is the supported provider. Three models are allowed:
- `deepseek/deepseek-v3.2-speciale` (recommended for production)
- `openai/gpt-5.2-codex`
- `moonshotai/kimi-k2.5`

All models use `reasoning.effort = "medium"` and `temperature = 0.3`.

### Focused sample workflow:
1) Create a focused sample (flower + risk nouns):
   `python scripts\phase3_focus_sample.py --output-path reports\phase3\phase3_focus_sample.csv --total 20 --flower-min 5 --flower-max 10`
2) Create a random sample:
   `python scripts\phase3_focus_sample.py --mode random --output-path reports\phase3\phase3_random_sample.csv --total 20 --seed 42`
3) Run LLM inference:
   `python scripts\phase3_pilot.py --provider openrouter --model deepseek/deepseek-v3.2-speciale --input-path <input.csv> --output-path <output.csv> --limit 20`

### Full production run (pending):
`python scripts\phase3_pilot.py --provider openrouter --model deepseek/deepseek-v3.2-speciale --input-path reports\phase2\phase2_extraction.csv --output-path reports\phase3\phase3_full_results.csv`

## Concurrency Controls
- Default MAX_CONCURRENT=10 (asyncio.Semaphore).
- Override: set MAX_CONCURRENT in .env or shell.

## Prompt Control
- Prompt lives in src/classifier_pipeline/prompts.py
- System instruction is static for caching benefits.
- JSON schema fields: identified_noun, conventional_classifier, conventional_classifier_zh, classifier_type, overuse_of_ge, rationale, flag_for_review, flag_reason
- Deterministic context feature passed to prompt: specific_semantic_class
- Computed columns (pre-inference): age_years, age_available
