# Dataset Generation Playbook: Shakespeare -> Modern Counterparts

This document is a production-grade guide for building a clean, trustworthy, and reproducible dataset where each Shakespeare line has a modern-English counterpart.

Current source file observed: `dataset/Shakespeare_data_clean_v2_train.csv`
Current columns:

- `Dataline`
- `Play`
- `ActSceneLine`
- `PlayerLine`
- `split`

Goal: add high-quality modern counterparts for `PlayerLine` while preserving traceability, consistency, and legal safety.

---

## 1) What "spotless" means for this dataset

A spotless dataset is not just "translated text." It should satisfy:

1. **Semantic fidelity**: modern line preserves meaning, speaker intent, and emotional force.
2. **Context awareness**: references to nearby lines are handled correctly (pronouns, irony, callbacks).
3. **Consistency**: similar constructions are modernized in similar ways across plays.
4. **Reproducibility**: every row can be traced to method, source, annotator, and revision history.
5. **Auditability**: quality checks, disagreements, and corrections are documented.
6. **Licensing clarity**: no copyrighted derivative text is copied without permission.

---

## 2) Recommended schema extension (do this first)

Do not write modern text directly into a single ad-hoc column. Add structured metadata columns.

Minimum recommended new columns:

- `modern_line_v1`: first modern rewrite.
- `modern_line_final`: approved modern rewrite.
- `method`: one of `human`, `llm_seed_human_edit`, `rule_based`, `retrieved_paraphrase`, `hybrid`.
- `annotator_id`: human contributor ID.
- `reviewer_id`: quality reviewer ID.
- `status`: `todo`, `draft`, `reviewed`, `approved`, `rejected`.
- `quality_score`: numeric (e.g., 1-5).
- `confidence`: `low`, `medium`, `high`.
- `context_window`: references to surrounding lines used during rewrite.
- `reference_source`: human source used (edition, glossary, notes, URL/book ID).
- `notes`: rationale for hard choices.
- `last_updated_utc`: ISO timestamp.

Optional but valuable:

- `ambiguity_flag`: boolean for uncertain interpretation.
- `named_entity_normalized`: if person/place names were expanded.
- `literalness`: scale (literal vs adaptive paraphrase).
- `style_register`: `neutral`, `formal`, `conversational`.

---

## 3) Main avenues to create modern counterparts

Use one or more of these, but always keep human QA in the loop.

### A) Pure human annotation (gold-standard)

How it works:

1. Annotator reads line + immediate context (at least +/-2 lines).
2. Writes modern counterpart.
3. Reviewer checks fidelity/style.
4. Disagreements resolved by adjudicator.

Pros:

- Highest control and interpretive quality.
- Best for benchmark/evaluation subsets.

Cons:

- Slow and expensive.

Best use:

- Create a high-quality seed set (e.g., 5k-20k rows).

### B) LLM-assisted draft + strict human post-edit

How it works:

1. LLM proposes first draft modern line.
2. Human editor corrects meaning drift, register, and context errors.
3. Reviewer approves.

Pros:

- Much faster than pure human.
- Good for scaling.

Cons:

- Hallucinations, over-interpretation, modernization artifacts.
- Must enforce policy to avoid blind acceptance.

Best use:

- Middle-scale production where reviewers are available.

### C) Retrieval from trusted public-domain paraphrase/commentary sources

How it works:

1. Retrieve matching scene/line explanations from vetted sources.
2. Convert explanation to one-line modern equivalent.
3. Human validates and normalizes style.

Pros:

- Better factual grounding than raw generation.

Cons:

- Alignment can be noisy by line index.

Best use:

- Difficult lines with archaic vocabulary.

### D) Rule-based lexical modernization + human cleanup

How it works:

1. Apply dictionary mappings (`thou -> you`, `hath -> has`, etc.).
2. Simple syntactic reorder rules.
3. Human cleanup for fluency.

Pros:

- Deterministic and auditable.
- Good baseline for comparisons.

Cons:

- Low naturalness; fails on idioms/metaphor.

Best use:

- Baseline dataset version and error analysis.

### E) Multi-pass hybrid pipeline (recommended for production)

1. Rule-based pre-normalization.
2. Retrieval hints from glossary/commentary.
3. LLM candidate generation (2-3 candidates).
4. Human selects/edits best candidate.
5. Independent reviewer approval.

Pros:

- Best quality-speed tradeoff.
- Strong traceability with per-stage artifacts.

---

## 4) Human-first guidance sources (to avoid overdependence on LLMs)

Prioritize these classes of resources:

1. **Scholarly editions** with line notes and glosses.
2. **Public-domain Shakespeare lexicons/glossaries** (archaic terms, idioms).
3. **Academic lecture notes / university open course materials**.
4. **Critical essays** for ambiguous or figurative lines.
5. **Parallel-text study resources** where licensing permits reuse.

What to record from each source:

- Bibliographic reference.
- URL and access date (if web).
- Whether text is quoted, paraphrased, or only consulted.
- License/usage permission notes.

Important: avoid copying modern translations that are copyrighted unless license explicitly allows dataset use.

---

## 5) Annotation policy (define before writing rows)

Create a short policy doc and train annotators on it.

Core policy rules:

1. Preserve meaning, speaker stance, and intensity.
2. Keep one source line -> one modern line unless line is semantically incomplete.
3. If line is a stage marker/speaker label (e.g., `BISHOP`), either:
   - keep as metadata and set modern line empty with status `non_dialogue`, or
   - normalize to a canonical token and exclude from training.
4. Do not inject new plot information.
5. Avoid over-interpretive expansions unless necessary for comprehension.
6. Keep punctuation natural but not noisy.
7. For profanity/slurs/violent language, preserve intent and content warning policy.

Ambiguous lines protocol:

- Add `ambiguity_flag = true`.
- Provide 2 candidate rewrites in `notes`.
- Escalate to adjudication.

---

## 6) Quality control framework

### A) Multi-tier review

1. **Self-check** by annotator.
2. **Peer review** by second annotator.
3. **Adjudication** for conflicts.

### B) Row-level checklist

For each row, verify:

1. Meaning preserved.
2. No hallucinated entities/events.
3. Grammar and fluency acceptable.
4. Register consistent with project target.
5. No accidental copying from copyrighted source.
6. Metadata complete.

### C) Sampling audits

- Random 5-10% weekly audit.
- Stratified audit by play, annotator, and ambiguity flag.
- Track failure categories (literal error, polarity flip, pronoun resolution, sarcasm loss).

### D) Metrics to track

- Approval rate.
- Rework rate.
- Inter-annotator agreement (binary acceptable/not acceptable, plus quality score delta).
- Average turnaround per 1,000 lines.

---

## 7) Data cleaning and edge-case handling

Your current file includes non-dialogue artifacts (example: speaker labels like `BISHOP`). Handle systematically.

Recommended row typing:

- `dialogue`
- `speaker_label`
- `stage_direction`
- `noise`

Add a `row_type` column and processing rules:

1. `dialogue`: must have modern counterpart.
2. `speaker_label`: keep for structure, usually exclude from seq2seq training pairs.
3. `stage_direction`: optionally modernize for separate task; usually exclude.
4. `noise`: quarantine and review.

Also normalize:

- apostrophes (`ne'er`, `bless'd`) to canonical forms only in helper columns, not source.
- extra whitespace and malformed punctuation.
- encoding anomalies.

---

## 8) Practical workflows you can run now

### Workflow 1: Gold subset first (strongly recommended)

1. Sample 2,000-5,000 representative dialogue rows from train split.
2. Perform pure human annotation + dual review.
3. Freeze as `gold_v1`.
4. Use this set to evaluate any automated pipeline.

### Workflow 2: Scaled hybrid production

1. Generate draft modern lines (rule/LLM/retrieval hybrid).
2. Human post-edit all rows.
3. Fast-track high-confidence rows; deep review low-confidence rows.
4. Weekly calibration sessions to align style decisions.

### Workflow 3: High-risk line lane

Route lines to specialist reviewers when they contain:

- dense metaphors
- archaic legal/religious references
- pronoun ambiguity
- irony/sarcasm

---

## 9) Suggested tooling stack

Use simple, reproducible tools:

1. Spreadsheet UI or annotation platform for human edits.
2. Python scripts for validation and schema checks.
3. Versioned data snapshots (Git LFS/DVC optional).
4. Lightweight dashboard for progress and QA metrics.

Validation script checks should include:

- missing required fields
- duplicate `Dataline`
- empty `modern_line_final` where `row_type=dialogue`
- invalid enum values (`status`, `method`, etc.)
- outlier line lengths

---

## 10) Legal and ethical guardrails

1. Verify source and reference licenses before reuse.
2. Do not paste protected modern translations into dataset.
3. Keep provenance metadata for every external reference.
4. Document handling policy for sensitive language.

---

## 11) Release process and dataset versioning

Recommended lifecycle:

1. `v0_raw`: original extracted lines.
2. `v1_schema`: columns expanded, row types assigned.
3. `v2_draft`: first-pass modern counterparts.
4. `v3_reviewed`: peer-reviewed rows.
5. `v4_gold`: adjudicated benchmark subset.
6. `v5_trainable`: final filtered training pairs.

Each release should include:

- changelog
- row counts by type/status
- QA metrics snapshot
- known limitations

---

## 12) Starter acceptance criteria (Definition of Done)

A row is done only if:

1. `row_type=dialogue` and `modern_line_final` is non-empty.
2. `status=approved`.
3. `method`, `annotator_id`, `reviewer_id`, and `last_updated_utc` are populated.
4. quality score >= target threshold (for example, >=4/5).
5. no blocking QA flags.

Project-level done condition (example):

- >=98% dialogue rows approved
- <1% critical semantic errors in audit sample
- complete provenance for externally aided rows

---

## 13) Recommended next execution order

1. Extend schema with metadata columns.
2. Add `row_type` classification pass.
3. Build and freeze gold subset.
4. Run scaled hybrid generation + human edit.
5. Enforce validation + weekly audits.
6. Release versioned dataset snapshots.

This sequence will keep quality high while still letting you scale beyond what pure manual translation can achieve.
