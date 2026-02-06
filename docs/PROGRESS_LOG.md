# Progress Log

## 2026-02-05 (continued)
- Data profiling of 70k Phase 2 rows initiated (classifier/speaker/age distributions, OMITTED prevalence).
- CHILDES DB schema reviewed: utterance_id, utterance_order, token_order available and easy to add; target_child_id and MLU not directly available.
- Manuscript documentation created (MANUSCRIPT_NOTES.md) with full methods, decision log, validation history.
- Three models benchmarked on focused (20 flower) + random (20) samples:
  - DeepSeek v3.2-speciale: 40/40 clean, selected for production.
  - GPT-5.2-Codex: 40/40 clean, minor adjective-in-noun issue.
  - Kimi k2.5: 40/40 clean with reasoning.effort="medium".
- Prompt v4 finalized: 5 numbered rules, 4 few-shot examples, dual classifier columns, demonstrative-copula rule, compound noun guidance, one-sentence rationale cap.
- Temperature lowered from 1.0 to 0.3 after consistency analysis.
- Random 20-row sampling mode added to phase3_focus_sample.py.

## 2026-02-06
- Phase 1 inventory finalized using childes-db (version 2021.1).
- Phase 2 extraction completed for monolingual Mandarin; long-format CSV locked (70,655 rows).
- Speaker_Code and Speaker_Role columns added to Phase 2 extraction.
- Phase 3 prompt v1: initial few-shot + expanded JSON schema.
- OpenRouter integration complete (moonshotai/kimi-k2.5).
- Focused validation sample built and run (v1): 2/20 list outputs, mixed noun format.
- Prompt v2: Focus Constraint + noun standardization. 20/20 clean.
- Prompt v3: demonstrative-copula rule, dual classifier columns, compound nouns, reformatted examples. 17/20 clean (3 malformed on multi-noun).
- Concurrency safety added via asyncio.Semaphore (MAX_CONCURRENT=10).
- age_years, age_available computed columns added.
- parse-retry handling added for malformed JSON responses.
- Prompt v5 finalized: six analysis rules, five few-shot examples, and narrow human-review flags (`flag_for_review`, `flag_reason`).
- Deterministic `specific_semantic_class` taxonomy added in Phase 2 and included in Phase 3 prompt context.
- PI random review sample (seed=99, n=20) generated and annotated for website review (`reports/phase3/phase3_pi_review_results.csv`).

## 2026-02-05
- Phase 1 scripts and reports created.
- Phase 2 deterministic extraction pipeline created.
- Initial Phase 3 pilot scaffolding built.
