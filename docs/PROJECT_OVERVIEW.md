# Project Overview

## Purpose
This project builds a research-grade pipeline to analyze Mandarin classifier usage in child language data.
The goal is a reproducible, annotated dataset that supports linguistic analysis of classifier choice,
overuse of general classifiers, and developmental patterns.

## High-level Goals
- Identify where children produce classifier constructions.
- Track which classifiers are used and whether they are conventional.
- Detect overuse of the general classifier (ge).
- Support age-based and corpus-based analyses.
- Produce a single long-format dataset aligned to the reference schema.

## Data Sources
- CHILDES / TalkBank data accessed via childes-db for programmatic access and reproducibility.
- Phase 1 uses childes-db metadata and token/utterance counts.
- Phase 2 extracts candidate classifier contexts deterministically.
- Phase 3 uses LLM inference for semantic judgments.

## Deliverables
- Phase 1 corpus inventory reports (CSV/JSON/summary).
- Phase 2 extraction table (CSV) with classifier contexts.
- Phase 3 annotated tables (CSV) produced via LLM inference.
- Focus samples for targeted validation.

## Current Status (2026-02-05)
- Phase 1 complete: inventory and counts generated via childes-db (2021.1).
- Phase 2 complete: monolingual Mandarin extraction with Speaker_Code/Speaker_Role columns (70,655 rows).
- Phase 3 prompt finalized through 4 iterations (v1-v4) on focused + random validation samples.
- Three models benchmarked (DeepSeek v3.2-speciale, GPT-5.2-Codex, Kimi k2.5) — all 20/20 on both test sets.
- DeepSeek v3.2-speciale selected for production run.
- Pipeline supports: reasoning.effort="medium", temperature=0.3, dual classifier columns (pinyin + Chinese), age_years/age_available computed columns.
- Data profiling and pre-run analysis in progress.

## Key Outputs
- reports/phase1/phase1_corpus_stats.csv
- reports/phase1/phase1_corpus_stats.json
- reports/phase1/phase1_summary.md
- reports/phase2/phase2_extraction.csv
- reports/phase2/rejected_samples.csv
- reports/phase3/phase3_focus_sample_v3.csv
- reports/phase3/phase3_focus_results_deepseek_v3.csv
- reports/phase3/phase3_focus_results_codex_v3.csv
- reports/phase3/phase3_focus_results_openrouter_v4.csv
- reports/phase3/phase3_random_sample.csv
- reports/phase3/phase3_random_results_deepseek.csv
- reports/phase3/phase3_random_results_codex.csv
- reports/phase3/phase3_random_results_kimi.csv

## Notes
- Direct Moonshot API usage is disabled; OpenRouter is the supported path.
- The system prompt must remain identical across requests for caching.
- 12.4% of rows (8,793) have missing age metadata from source corpora (flagged, not dropped).
- See MANUSCRIPT_NOTES.md for detailed methodology documentation.
