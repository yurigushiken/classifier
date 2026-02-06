# Logging and QA

## Logging Philosophy
- Keep all run artifacts in reports/ and logs/.
- Prefer deterministic scripts and fixed random seeds for sampling.
- Avoid printing secrets in logs.

## Logging Locations
- reports/: data outputs (CSV/JSON/MD summaries)
- logs/: optional runtime logs (stdout/stderr captures, timing)

## Recommended Practice
- For each run, redirect console output to a timestamped log:
  - Example: `python scripts\phase3_pilot.py ... > logs\phase3_pilot_YYYYMMDD_HHMMSS.txt`
- Store any manual notes or QC decisions in logs/.

## QA Checks
- Spot-check focus sample outputs for:
  - identified_noun presence or correct OMITTED.
  - conventional_classifier plausibility.
  - overuse_of_ge matches strict rule.
- Track anomalies (list outputs, multiple noun pairs) for prompt refinement.

## Known Risks
- Multi-instance utterances can still return malformed JSON in rare cases; parse-retry is enabled, but failures should be logged and manually reviewed.
- Borderline classifier conventions (colloquial vs. prescriptive) can vary across speakers/corpora and should be checked via `flag_for_review`.
- Ellipsis contexts can blur classifier use.

## Future Enhancements
- Add structured logging (JSONL) to phase3 pipeline for easy auditing.
- Add automatic detection for list outputs and split into multiple rows.
