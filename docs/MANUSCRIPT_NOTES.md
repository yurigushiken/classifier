# Manuscript Notes

Working document for writing up the classifier acquisition study. Organized by standard empirical manuscript sections. Updated as the pipeline matures.

Last updated: 2026-02-06

---

## 1. Research Context

### Background
Mandarin Chinese uses numeral classifiers (measure words) between numerals/demonstratives and nouns. Children acquiring Mandarin must learn which classifier conventionally pairs with which noun. A well-documented developmental pattern is the **overgeneralization of the general classifier 个 (ge)** — children use 个 in contexts where a specific classifier (e.g., 本 ben for books, 条 tiao for fish) is conventionally required by adult grammar.

### Research Questions (working)
- RQ1: What is the distribution and frequency of classifier use in Mandarin-speaking children's spontaneous speech?
- RQ2: To what extent do children overgeneralize the general classifier 个 (ge) to nouns that conventionally require specific classifiers?
- RQ3: How does classifier use (and overuse of 个) change as a function of age?
- RQ4: Do patterns differ by speaker role (child vs. adult input)?
- RQ5: What is the relationship between numerosity contexts (counting, demonstratives, quantification) and classifier choice?

*Note: RQs are provisional. Final framing depends on data profiling results.*

---

## 2. Method

### 2.1 Data Source
- **Corpus:** CHILDES / TalkBank (MacWhinney, 2000)
- **Access method:** childes-db (Sanchez et al., 2019) — programmatic SQL access for reproducibility
- **Database version:** 2021.1
- **Language filter:** Mandarin Chinese (zho, cmn); excludes Cantonese (yue) and Min Nan (nan)
- **Collection filter:** `collection_name = 'Chinese'` (monolingual Mandarin corpora only)
- **Total corpora inventoried:** 39 (Phase 1); monolingual subset used for extraction

### 2.2 Phase 1: Corpus Inventory
- Enumerated all Chinese-language corpora in CHILDES via childes-db
- Collected per-corpus metadata: transcript count, utterance count, speaker codes, age ranges, classifier token frequencies (all speakers and CHI-only)
- Output: `reports/phase1/phase1_corpus_stats.csv`, summary statistics

### 2.3 Phase 2: Deterministic Extraction of Classifier Contexts
- **Target classifiers:** 34 classifiers drawn from standard Mandarin reference grammars:
  个, 条, 只, 本, 张, 位, 头, 件, 年, 次, 天, 元, 下, 块, 岁, 名, 句, 家, 片, 份, 分, 部, 场, 把, 根, 颗, 辆, 种, 笔, 群, 对, 架, 组, 碗
- **Extraction logic:** For each classifier token in the corpus, the immediately preceding token is checked. If the preceding token's POS tag indicates a numeral (`num*`), determiner (`det`), or demonstrative pronoun (`pro:dem`), the row is included. Otherwise it is rejected.
- **SQL join:** `token c JOIN token p ON p.utterance_id = c.utterance_id AND p.token_order = c.token_order - 1`
- **Output columns:** File Name, Collection_Type, Speaker_Code, Speaker_Role, Age (days), Utterance, POS tags, Determiner/Numbers, Classifier
- **Rejected rows:** Sampled (reservoir sampling, n=50, seed=13) for QA inspection
- **Total extracted rows:** 70,655
- **Missing age:** 8,793 rows (12.4%) from corpora lacking `target_child_age` metadata (primarily ZhouAssessment, LiZhou)

### 2.4 Phase 3: LLM-Assisted Annotation
- **Task:** For each classifier context, an LLM identifies the modified noun, determines the conventional (adult-standard) classifier, classifies the token as General or Specific, and judges whether 个 was overused.
- **Provider:** OpenRouter API
- **Models evaluated:** moonshotai/kimi-k2.5, deepseek/deepseek-v3.2-speciale, openai/gpt-5.2-codex
- **Reasoning:** All models set to `reasoning.effort = "medium"` via OpenRouter
- **Temperature:** 0.3 (chosen after observing inconsistency at 1.0)
- **Response format:** JSON with enforced schema (`response_format: {type: "json_object"}`)
- **Concurrency:** asyncio.Semaphore, MAX_CONCURRENT=10

#### Prompt Design (iterative)
The system prompt was refined through 5 iterations on focused and random 20-row validation samples enriched with high-risk cases (flower nouns, compound nouns, demonstrative-copula constructions, and multi-classifier utterances).

Key prompt features (final version):
1. **Six numbered analysis rules** covering noun identification, convention lookup, classifier type, overuse judgment, demonstrative-copula disambiguation, and review-flag assignment.
2. **Compound noun instruction** to treat multi-morpheme nouns as single units.
3. **Focus constraint** to analyze only the target classifier instance following the specified Preceding Word (single-object JSON output).
4. **Output format rules**: identified_noun in Simplified Chinese, conventional_classifier in pinyin, and conventional_classifier_zh in Simplified Chinese.
5. **Five few-shot examples** matching real input format (Chinese + POS tags), including one flagged borderline case.
6. **One-sentence rationale cap** to control output length.
7. **Review flags** (`flag_for_review`, `flag_reason`) for four narrow human-review triggers.

#### Output Schema (per row)
| Field | Description |
|-------|-------------|
| identified_noun | The noun modified by the classifier, in Simplified Chinese. 'OMITTED' if implicit. |
| conventional_classifier | Standard adult classifier for this noun, in pinyin |
| conventional_classifier_zh | Standard adult classifier, in Simplified Chinese |
| classifier_type | "General" if token is 个, "Specific" otherwise |
| overuse_of_ge | True if 个 was used where a specific classifier is conventional; False otherwise |
| flag_for_review | Boolean flag for prioritized human review |
| flag_reason | Trigger code: colloquial_tolerance, disputed_convention, implicit_noun_inference, or multi_instance_disambiguation |
| rationale | One-sentence explanation of the judgment |

#### Computed columns (pre-inference)
| Field | Description |
|-------|-------------|
| specific_semantic_class | Deterministic classifier-semantic class from token lookup (e.g., general, animacy, shape, temporal_event) |
| age_years | Age in decimal years (age_days / 365.25, rounded to 1 decimal) |
| age_available | Boolean flag: True if age metadata exists in source |

### 2.5 Validation

#### Prompt iteration history
| Version | Sample | Changes | Key result |
|---------|--------|---------|------------|
| v1 | Focus 20 (flower) | Initial few-shot + JSON schema | 2/20 list outputs, mixed pinyin/Chinese in nouns, colon artifact |
| v2 | Focus 20 (flower) | Added Focus Constraint, noun standardization rule | 0 list outputs, all nouns in Chinese. 20/20 clean. |
| v3 | Focus 20 (flower) | Added dual classifier columns, demonstrative-copula rule, compound noun rule, speaker-neutral wording, reformatted examples, rationale cap. Temperature lowered to 0.3. | 17/20 clean; 3 malformed rows on multi-noun utterances |
| v4 | Focus 20 + Random 20 | Added reasoning.effort="medium". Benchmarked 3 models. | All models: 20/20 clean on both samples |
| v5 | PI Random 20 (seed=99) | Added review flags and one additional few-shot example for borderline colloquial cases. | 20/20 clean; 2/20 flagged for human review |

#### Model benchmark (v4, reasoning=medium)
| Metric | DeepSeek v3.2 | Codex 5.2 | Kimi k2.5 |
|--------|---------------|-----------|-----------|
| Focus sample: clean rows | 20/20 | 20/20 | 20/20 |
| Random sample: clean rows | 20/20 | 20/20 | 20/20 |
| Focus: overuse=True | 12 | 12 | 12 |
| Random: overuse=True | 4 | 4 | 5 |
| Format consistency | Excellent | Good (adj in nouns) | Excellent |
| Rationale language | English | Mixed Eng/Chinese | English |

All three models showed equivalent accuracy. DeepSeek selected for production based on consistency of noun identification (bare head nouns, no adjective contamination) and uniform English rationales.

#### Known limitations of LLM annotation
- **No ground-truth lookup table:** The model relies on its own knowledge of conventional classifiers, which may vary for borderline nouns (e.g., 杯子, 肉圆, 怪物).
- **Non-个 misuse undetected:** The overuse_of_ge field only captures overuse of the general classifier. A child using the wrong specific classifier (e.g., 只 for 笔 instead of 支) is marked overuse_of_ge=False.
- **Multi-instance ambiguity:** When an utterance contains multiple instances of the same Preceding Word + Classifier pair (e.g., two occurrences of 一个), the model may select either instance. Positional disambiguation is not provided in the current input.

---

## 3. Data Profiling (Phase 2: 70,655 rows)

### 3.1 Classifier Distribution
The general classifier 个 dominates at 80.98% (57,219 rows). The top 4 classifiers (个, 只, 本, 张) cover 90.3%. 34 unique classifier tokens are present. This extreme skew is consistent with the literature on child language and adult Mandarin alike.

| Classifier | Count | % |
|-----------|-------|---|
| 个 | 57,219 | 80.98% |
| 只 | 3,854 | 5.45% |
| 本 | 1,383 | 1.96% |
| 张 | 1,379 | 1.95% |
| 次 | 983 | 1.39% |
| 天 | 905 | 1.28% |
| 块 | 760 | 1.08% |
| 种 | 741 | 1.05% |
| 条 | 470 | 0.67% |
| All others (25) | 4,961 | 7.02% |

### 3.2 Speaker Distribution
Target children produce 37.1% of classifier contexts; mothers produce 34.9%. Adult speech (all non-child roles) totals ~62.9%.

| Speaker Role | Count | % |
|-------------|-------|---|
| Target_Child | 26,226 | 37.12% |
| Mother | 24,631 | 34.86% |
| Investigator | 7,009 | 9.92% |
| Teacher | 4,145 | 5.87% |
| Adult | 3,757 | 5.32% |
| Child (non-target) | 3,315 | 4.69% |
| Father | 672 | 0.95% |
| Other | 900 | 1.27% |

### 3.3 Age Distribution
Age ranges from 1.2 to 10.5 years (mean 4.2, median 4.0). Peak data density at 3-4 years. 8,793 rows (12.4%) missing age.

| Age bracket | Count | % of age-available |
|------------|-------|-------------------|
| 1-2 yrs | 2,901 | 4.69% |
| 2-3 yrs | 9,866 | 15.95% |
| 3-4 yrs | 14,294 | 23.11% |
| 4-5 yrs | 12,165 | 19.66% |
| 5-6 yrs | 12,031 | 19.45% |
| 6-7 yrs | 8,595 | 13.89% |
| 7+ yrs | 2,010 | 3.25% |

### 3.4 Utterance Complexity
11.4% of rows (8,045) come from multi-classifier utterances (3,735 unique utterances generating 2+ rows each). 88.6% come from single-classifier utterances.

### 3.5 Estimated OMITTED Prevalence
~23.4% of rows (~16,520) have no noun following the classifier. These represent demonstrative/anaphoric uses (这个, 那个), bare counting (两个), and utterance-final ellipsis. These rows will return OMITTED from the LLM and contribute no classifier-noun pairing data.

### 3.6 Noun Diversity
~3,409 distinct tokens follow classifiers. After excluding non-nominal tokens (copulas, adverbs, adjectives, pronouns), estimated ~2,500-3,000 true unique nouns. Top following tokens: 是 (9.17%, copula — triggers demonstrative-copula rule), 小 (4.57%, adjective), 人 (3.26%), 故事, 桌子, 皮皮鼠, 书, 大象, 东西, 球.

### 3.7 Corpus Coverage
18 corpora. The Zhou family (Zhou1/2/3, ZhouAssessment, ZhouNarratives, ZhouDinner) contributes ~40% of all data. Top 5 corpora cover 63%.

### 3.8 Determiner/Number Distribution
Three values (这 45.98%, 一 30.22%, 那 8.35%) cover 84.6% of determiners. The interrogative 几 appears 2.12%. Ordinals (第一, 第二, etc.) are uncommon (~1.8% combined).

### 3.9 Duplicate Analysis
28.1% of rows (19,880) are duplicates sharing the same (Utterance, Classifier, Determiner) triple. The most common: "这 个" appears 1,555 times across speakers and sessions. "这 个 呢" appears 688 times. These are not errors — they reflect formulaic speech. Deduplication or speaker-level aggregation essential for treating rows as independent observations.

---

## 4. Results (placeholder)

*To be populated after full annotation run.*

---

## 5. Decision Log

Transparent record of methodological decisions and their rationale.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-05 | Use childes-db (SQL) instead of direct CHAT file parsing | Reproducibility; programmatic access; standard in the field |
| 2026-02-05 | Filter to collection='Chinese' only | Ensures monolingual Mandarin; excludes bilingual and dialectal corpora |
| 2026-02-05 | Use POS-based filter for preceding token (num, det, pro:dem) | Deterministic identification of classifier contexts without semantic analysis |
| 2026-02-06 | Switch from Moonshot direct API to OpenRouter | Moonshot API disabled; OpenRouter provides unified access |
| 2026-02-06 | Add Focus Constraint to prompt | Multi-noun utterances produced list outputs, breaking CSV schema |
| 2026-02-06 | Standardize identified_noun to Simplified Chinese | Model mixed pinyin, English, and Chinese inconsistently |
| 2026-02-05 | Add demonstrative-copula rule | Inconsistent OMITTED judgments on 这个是X patterns |
| 2026-02-05 | Add dual conventional_classifier columns (pinyin + Chinese) | Model mixed formats in single column; dual columns eliminate normalization burden |
| 2026-02-05 | Lower temperature from 1.0 to 0.3 | Same noun received opposite overuse judgments across similar utterances at temp=1.0 |
| 2026-02-05 | Set reasoning.effort="medium" for all models | 3/20 malformed rows at temp=0.3 without reasoning; 0/20 with reasoning |
| 2026-02-05 | Include all speaker roles (not just CHI) | Adult speech is the acquisition input model; enables child-vs-adult comparison |
| 2026-02-05 | Keep rows with missing age (flag, don't drop) | 12.4% of data; age needed only for developmental analyses, not annotation |
| 2026-02-05 | Select DeepSeek v3.2-speciale for production | Equivalent accuracy to alternatives; cleanest noun identification and rationale format |

---

## 6. References (working)

- MacWhinney, B. (2000). The CHILDES Project: Tools for Analyzing Talk. Lawrence Erlbaum Associates.
- Sanchez, A., Meylan, S. C., Braginsky, M., MacDonald, K. E., Yurovsky, D., & Frank, M. C. (2019). childes-db: A flexible and reproducible interface to the Child Language Data Exchange System. Behavior Research Methods, 51(4), 1928-1941.
- Erbaugh, M. S. (1986). Taking stock: The development of Chinese noun classifiers historically and in young children. In C. Craig (Ed.), Noun Classes and Categorization (pp. 399-436). John Benjamins.
