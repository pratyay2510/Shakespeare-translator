# Repository Intelligence Brief: Shakespeare Translator (bart-trans)

Last updated: 2026-04-06 (local context)

## 1) Mission and Practical Goal

This repository is a research/engineering workspace for building a Shakespeare modernization and style-transfer system.

Two complementary tracks exist:

1. Data construction track (currently the most developed)
- Build clean, auditable Shakespeare line datasets.
- Produce modernized counterparts under legal/compliance constraints.
- Track quality, confidence, rejection reasons, and adjudication queues.

2. Model runtime/inference track
- Run BART-based sequence-to-sequence generation.
- Keep model assets local for repeatable offline/low-friction use.
- Support GPU/CPU fallback execution.

The immediate intent of the repo appears to be:
- Prepare high-volume, compliance-safe training candidates.
- Measure data quality and failure modes.
- Iterate toward better modernization quality before/while scaling model training.

## 2) Current Repository Scope (Tracked Files)

Tracked inventory is centered around:
- Documentation and prompts: README.md, datasetgen.md, agent_prompt_option_c.md
- Core scripts: src/*.py
- Data assets: dataset/*.csv, data/processed/*.csv
- Logs/reports: logs/*.md, logs/*.json
- Local pretrained model metadata: models/pretrained/bart-base/*.json
- Minimal dependency file: requirements.txt
- Git helper script: push.sh

Notable local-but-ignored runtime environment folders exist:
- shtrans/
- .venv/

These are excluded by .gitignore and represent local execution environments.

## 3) What the Project Is Trying To Do (One-Sentence Summary)

Create a legally compliant, quality-audited pipeline that modernizes Shakespearean lines into contemporary English and prepares data artifacts suitable for downstream seq2seq model development and inference.

## 4) End-to-End Data Story

### 4.1 Raw-to-clean lineage

Primary source table lineage:
1. dataset/Shakespeare_data.csv (largest base source)
2. dataset/Shakespeare_data_clean_minimal.csv (minimal filtered subset)
3. dataset/Shakespeare_data_clean_v2.csv (training-ready cleaned dataset)
4. deterministic splits:
- dataset/Shakespeare_data_clean_v2_train.csv
- dataset/Shakespeare_data_clean_v2_val.csv
- dataset/Shakespeare_data_clean_v2_test.csv

### 4.2 Processed outputs

Two major processed flows are present:

A) Train-set compliance fallback modernization
- data/processed/modernized_train_candidate.csv
- data/processed/modernized_train_rejected.csv
- logs/retrieval_alignment_report.json
- logs/compliance_report.md
- logs/qa_metrics.json
- logs/adjudication_sheet.md
- data/processed/adjudication_priority.csv

B) Test-set modernization snapshot
- data/processed/Shakespeare_data_clean_v2_test_processed.csv

### 4.3 Current dataset sizes (line counts including header)

- dataset/Shakespeare_data.csv: 111,397
- dataset/Shakespeare_data_clean_minimal.csv: 105,154
- dataset/Shakespeare_data_clean_v2.csv: 104,049
- dataset/Shakespeare_data_clean_v2_train.csv: 80,063
- dataset/Shakespeare_data_clean_v2_val.csv: 6,100
- dataset/Shakespeare_data_clean_v2_test.csv: 17,877
- data/processed/modernized_train_candidate.csv: 80,063
- data/processed/modernized_train_rejected.csv: 1,008
- data/processed/Shakespeare_data_clean_v2_test_processed.csv: 17,877
- data/processed/adjudication_priority.csv: 9

## 5) File-by-File Intent and Behavior

### 5.1 Top-level docs

README.md
- Describes BART fine-tuning objective (modern English -> Shakespearean style transfer).
- Documents memory-aware training assumptions for constrained GPUs.
- Includes DeepSpeed + fp16 + gradient checkpointing guidance.
- Provides a suggested training command and project scaffold.

Important note:
- README describes future/target training architecture; some referenced training assets (for example train.py, deepspeed config file under configs) are not currently tracked in this repo snapshot.


### 5.2 Data-policy and agent docs

datasetgen.md
- Defines "spotless" dataset criteria: fidelity, consistency, reproducibility, legal safety.
- Recommends explicit metadata schema for modernization lifecycle.
- Specifies QC frameworks, adjudication, and edge-case handling.

agent_prompt_option_c.md
- Prescribes an Option C retrieval-guided pipeline with strict compliance gating.
- Enforces legal checks first, and fallback to original paraphrase generation when external reuse/scrape is disallowed.
- Defines expected artifacts and reporting obligations.


### 5.3 Core Python scripts

src/build_option_c_fallback.py
Purpose:
- Implements compliance-first fallback modernization pipeline for train split.

Key behaviors:
- Loads dataset/Shakespeare_data_clean_v2_train.csv.
- Classifies rows into row_type: dialogue, speaker_label, stage_direction, noise.
- Applies rule-based archaic-to-modern lexical substitutions for dialogue.
- Generates rich metadata columns:
  modern_line_v1, modern_line_final, method, annotator_id, reviewer_id, status,
  quality_score, confidence, context_window, reference_source, notes,
  last_updated_utc, row_type, ambiguity_flag, alignment_status.
- Rejects low-confidence/ambiguous rows to adjudication path.
- Writes candidate/rejected CSV plus alignment/compliance/QA logs.
- Validates enums, required fields, and approved-dialogue constraints.

Compliance mode hardcoded:
- method = retrieved_paraphrase_guided_original
- retrieval_attempted = false


src/process_test_modernization.py
Purpose:
- Rule-based modernization for test split with confidence flagging.

Key behaviors:
- Reads dataset/Shakespeare_data_clean_v2_test.csv.
- If missing columns, creates ModernizedLine and ConfidenceFlag.
- Applies archaic replacements and fallback sentence humanization.
- Handles special structural rows (stage directions, all-caps speaker labels).
- Writes both:
  - dataset/Shakespeare_data_clean_v2_test.csv (updated in-place)
  - data/processed/Shakespeare_data_clean_v2_test_processed.csv


src/modernize_with_flan.py
Purpose:
- LLM modernization path using FLAN-T5 (default google/flan-t5-base).

Key behaviors:
- Loads test CSV by default.
- Drops all-caps speaker rows.
- Builds prompt with optional few-shot examples from existing ModernizedLine values.
- Generates with beam search + sampling.
- Retries near-copy outputs with stronger rewrite instruction.
- Cleans formatting and computes confidence via similarity + unresolved archaic markers.
- Writes modernized output CSV with confidence labels.


src/model_runtime.py
Purpose:
- Prepare and load BART model/tokenizer from either local path or Hugging Face ID.

Key behaviors:
- Downloads model once to local directory if absent.
- Loads with local_files_only for reproducible local inference.
- Returns model/tokenizer/device and resolved path.


src/inference.py
Purpose:
- Interactive one-line inference entrypoint for BART generation.

Key behaviors:
- Uses model_runtime helper to load model.
- Prompts user for one sentence.
- Runs generate() with configurable max_new_tokens and beam size.


### 5.4 Notebooks

dataset/shakespeare_analysis.ipynb
- Executed EDA and cleaning notebook.
- Implements:
  - schema/null/text-length analysis,
  - minimal cleaning,
  - v2 training-ready cleaning,
  - deterministic split-by-play via md5 hashing (80/10/10 buckets),
  - output writes for clean v2 and train/val/test splits.
- Later section loads processed candidate modernization output for inspection.

src/initital.ipynb
- Small experimentation notebook for BART loading/generation.
- Contains CUDA-enforced quick generation probe.
- Currently mostly scratch/initialization content.

## 6) Output Quality State (From Logged Metrics)

From logs/qa_metrics.json:
- processed_rows: 80,062
- dialogue_rows: 79,780
- approved_rows: 78,773
- rejected_rows: 1,007
- reviewed_rows: 282
- confidence distribution:
  - medium: 67,764
  - high: 11,291
  - low: 1,007
- compliance_mode: compliant_fallback
- validation_passed: true

From logs/retrieval_alignment_report.json:
- mode: compliant_fallback
- retrieval_attempted: false
- all rows marked not_applicable_fallback in alignment_status
- rows_needing_human_adjudication: 1,007

From logs/adjudication_sheet.md and adjudication_priority.csv:
Top reject reason categories:
1. length_outlier (384)
2. unresolved_lexicon:marry (259)
3. unresolved_lexicon:prithee (137)
4. unresolved_lexicon:wherein (91)
5. unresolved_lexicon:betwixt (49)
6. unresolved_lexicon:therein (46)
7. unresolved_lexicon:forsooth (40)

Interpretation:
- Pipeline is high-throughput and structurally stable.
- Biggest remaining quality gap is unresolved archaic lexicon and very short/fragment lines.
- Human adjudication queue is explicitly prepared.

## 7) Compliance and Legal Posture (Critical for Any Agent)

The repository enforces compliance-first behavior for retrieval-guided modernization:
- External site scraping/reuse concerns were assessed and logged.
- Active decision is fallback-only generation without external paraphrase copying.
- No verbatim external modern paraphrases should be ingested.

Agent rule of thumb:
- Treat the current pipeline as legally constrained to local/source-text transformations and original wording generation unless policy proof changes.

## 8) Model Assets and Runtime State

Tracked local model metadata exists for models/pretrained/bart-base:
- config.json
- generation_config.json
- tokenizer.json
- tokenizer_config.json

The large weight file model.safetensors is intentionally ignored by git (see .gitignore) but may exist locally for runtime.

## 9) Dependencies and Environment

requirements.txt currently lists:
- pandas
- torch
- transformers
- tqdm
- sentencepiece
- protobuf

Note:
- README references additional ecosystem packages (datasets, accelerate, evaluate, deepspeed, bitsandbytes, etc.), but they are not pinned in requirements.txt.

## 10) Git Automation Scripts

push.sh
- Stages all local changes.
- Commits if needed (auto message if none provided).
- fetch + pull --rebase on current branch.
- Pushes to origin.

Operational caveat:
- Because it runs git add -A, it commits everything in working tree, not a selective subset.

## 11) Primary Workflows for Future Agents

### Workflow A: Re-run compliance fallback modernization

1. Ensure dataset/Shakespeare_data_clean_v2_train.csv exists and schema is intact.
2. Run src/build_option_c_fallback.py.
3. Inspect outputs in data/processed and logs.
4. Review rejection categories and adjudication queue.

### Workflow B: Produce/refresh test modernization outputs

1. Run src/process_test_modernization.py for deterministic rule-based refresh.
2. Optionally run src/modernize_with_flan.py for LLM-based modernization experiments.
3. Compare confidence distributions and qualitative outputs.

### Workflow C: Interactive generation smoke test

1. Run src/inference.py.
2. Provide prompt.
3. Confirm generation path, device, and output quality.

## 12) Gaps, Risks, and Technical Debt

1. Training pipeline code referenced by README is partially aspirational in current tracked files (for example missing explicit train.py and config artifacts in configs).
2. Confidence labeling is heuristic; no robust semantic equivalence metric is wired in.
3. Some modernization outputs preserve archaic wording for flagged categories, creating adjudication load.
4. requirements.txt under-specifies packages implied by README and notebook workflows.
5. push.sh commits all files, which can accidentally include unrelated local artifacts if gitignore misses something.

## 13) Ground-Truth Schema Notes

### dataset/Shakespeare_data_clean_v2_train.csv
Columns:
- Dataline
- Play
- ActSceneLine
- PlayerLine
- split

### data/processed/modernized_train_candidate.csv
Columns:
- Dataline
- Play
- ActSceneLine
- PlayerLine
- split
- modern_line_v1
- modern_line_final
- method
- annotator_id
- reviewer_id
- status
- quality_score
- confidence
- context_window
- reference_source
- notes
- last_updated_utc
- row_type
- ambiguity_flag
- alignment_status

### data/processed/modernized_train_rejected.csv
Candidate columns plus:
- reject_reason

### data/processed/Shakespeare_data_clean_v2_test_processed.csv
Columns:
- Dataline
- Play
- ActSceneLine
- PlayerLine
- split
- ModernizedLine
- ConfidenceFlag

## 14) Operational Conventions for Any Agent Using This Repo

1. Preserve compliance fallback posture unless legal constraints are explicitly re-evaluated and documented.
2. Do not overwrite source datasets without writing corresponding processed outputs and logs.
3. Keep row-level provenance fields populated for any generated modernization.
4. Keep deterministic behavior where possible (fixed seeds, stable splits).
5. Treat low-confidence rows as review-required, not silently accepted.

## 15) Quick Command Reference

Run fallback pipeline:
- python src/build_option_c_fallback.py

Run test modernization (rule-based):
- python src/process_test_modernization.py

Run FLAN modernization:
- python src/modernize_with_flan.py

Run interactive inference:
- python src/inference.py

Push changes:
- ./push.sh "your commit message"

## 16) Bottom Line

This repository is not just a model playground. It is primarily a compliance-aware dataset production and quality-control system for Shakespeare modernization, with BART runtime components available for generation experiments. Any autonomous agent should prioritize legal constraints, provenance-rich outputs, and adjudication-aware quality loops over raw modernization throughput.
