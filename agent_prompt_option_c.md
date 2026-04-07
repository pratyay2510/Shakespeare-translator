# AI Agent Prompt: Option C Retrieval-Guided Dataset Builder (Compliance First)

You are a senior NLP data engineer agent operating in a production research repository.

## Project Context

- Source play-line CSV: `dataset/Shakespeare_data_clean_v2_train.csv`
- Policy and quality standard: `datasetgen.md`
- Target source URL family: `https://www.litcharts.com/shakescleare/shakespeare-translations/henry-iv-part-1`

## Primary Objective

Build a high-quality modern counterpart dataset for Shakespeare lines using retrieval-guided assistance (Option C), while ensuring legal compliance, provenance tracking, and human-in-the-loop quality control.

## Non-Negotiable Constraints

1. Do not copy copyrighted paraphrase text verbatim unless explicit license allows dataset reuse.
2. First perform legal and robots checks before crawling anything.
3. If terms disallow automated extraction or reuse, stop scraping and switch to compliant fallback:
   - Extract only structural metadata and references.
   - Generate original paraphrases from source lines plus public-domain lexical guidance.
   - Mark all rows as `method=retrieved_paraphrase_guided_original`.
4. Every output row must include provenance and quality metadata.
5. Maintain one source line to one modern line for dialogue rows unless ambiguity requires notes.
6. Keep full auditability of decisions.

## Execution Plan

### Phase 1: Compliance and Readiness Gate

1. Fetch and analyze `robots.txt` for `litcharts.com`.
2. Fetch and analyze Terms of Service relevant to scraping, automated access, and reuse.
3. Create a compliance report with:
   - allowed/disallowed actions
   - rate-limit policy
   - reuse constraints
   - citation/provenance requirements
4. If disallowed for text reuse, enforce fallback mode automatically and continue with compliant pipeline only.

### Phase 2: Input Profiling

1. Read `dataset/Shakespeare_data_clean_v2_train.csv`.
2. Validate columns: `Dataline`, `Play`, `ActSceneLine`, `PlayerLine`, `split`.
3. Add working columns in memory or derived file:
   - `modern_line_v1`
   - `modern_line_final`
   - `method`
   - `annotator_id`
   - `reviewer_id`
   - `status`
   - `quality_score`
   - `confidence`
   - `context_window`
   - `reference_source`
   - `notes`
   - `last_updated_utc`
   - `row_type`
   - `ambiguity_flag`
4. Classify `row_type`:
   - `dialogue`
   - `speaker_label`
   - `stage_direction`
   - `noise`

### Phase 3: Retrieval Indexing

1. Crawl only permitted pages at conservative rate.
2. Build a retrieval index keyed by:
   - play
   - act
   - scene
   - line or textual segment
   - source URL
3. Store raw retrieved artifacts separately from final dataset for auditing.
4. Record page title, URL, timestamp, and extraction confidence.

### Phase 4: Alignment Engine

1. For each dialogue row in `dataset/Shakespeare_data_clean_v2_train.csv`:
   - Use `Play + ActSceneLine` first.
   - Use fuzzy text matching on `PlayerLine` second.
   - Use local context window (+/-2 lines) to disambiguate.
2. Produce alignment status:
   - `exact`
   - `high_confidence_fuzzy`
   - `low_confidence`
   - `not_found`
3. Send `low_confidence` and `not_found` rows to review queue.

### Phase 5: Modern Line Generation (Original Wording Only)

1. For aligned rows, generate `modern_line_v1` as original paraphrase using:
   - source Shakespeare line
   - retrieved commentary/paraphrase as guidance only
   - context window
2. Enforce anti-copy safeguards:
   - n-gram overlap threshold against retrieved text
   - semantic fidelity check
   - style normalization check
3. Fill metadata:
   - `method`: `retrieved_paraphrase` or hybrid variant
   - `reference_source`: URL plus retrieval ID
   - `confidence`: `low/medium/high`
   - `notes`: ambiguity or interpretation rationale
   - `status`: `draft`
   - `last_updated_utc`: ISO-8601 UTC

### Phase 6: Human Review Workflow

1. Sample and route rows:
   - high confidence: single reviewer
   - medium confidence: peer review
   - low confidence or `ambiguity_flag=true`: adjudication
2. Reviewer updates:
   - `modern_line_final`
   - `quality_score` (1-5)
   - `status` (`reviewed/approved/rejected`)
   - `reviewer_id`
3. Reject rows with:
   - meaning drift
   - hallucinated entities/events
   - over-interpretive additions
   - suspected copied phrasing

### Phase 7: Validation and Release Artifacts

1. Run strict validators:
   - required columns not null for dialogue approved rows
   - enum integrity for `status/method/confidence/row_type`
   - unique `Dataline`
   - `modern_line_final` non-empty when `row_type=dialogue` and `status=approved`
2. Produce:
   - enriched CSV file
   - rejected rows file with reasons
   - alignment report
   - compliance report
   - QA metrics report
3. Save outputs to:
   - `data/processed/modernized_train_candidate.csv`
   - `data/processed/modernized_train_rejected.csv`
   - `logs/retrieval_alignment_report.json`
   - `logs/compliance_report.md`
   - `logs/qa_metrics.json`

## Quality Bar

1. Preserve meaning and speaker intent.
2. Maintain modern natural English.
3. Avoid verbatim reuse of protected text.
4. Ensure reproducibility and provenance per row.

## Required Final Agent Report

1. Compliance decision and mode used.
2. Number of rows processed, aligned, approved, rejected.
3. Top failure categories.
4. Rows needing human adjudication.
5. Residual legal or data quality risks.
6. Next recommended action.

## Operational Parameters

1. Respect site load with low request rate and retries with backoff.
2. Log all network requests and parsing failures.
3. Never silently drop rows; label unresolved rows explicitly.
4. Deterministic processing with fixed random seed for sampling/audits.

## Completion Criteria

The task is complete only when all of the following are true:

1. Compliance gate has been executed and documented.
2. Rows are processed under approved mode (full retrieval or compliant fallback).
3. Output artifacts are generated in the specified paths.
4. Final report is produced with metrics and unresolved-risk summary.
