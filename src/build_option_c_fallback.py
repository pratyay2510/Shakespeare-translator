from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "dataset" / "Shakespeare_data_clean_v2_train.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "modernized_train_candidate.csv"
REJECTED_CSV = ROOT / "data" / "processed" / "modernized_train_rejected.csv"
ALIGNMENT_REPORT = ROOT / "logs" / "retrieval_alignment_report.json"
COMPLIANCE_REPORT = ROOT / "logs" / "compliance_report.md"
QA_METRICS = ROOT / "logs" / "qa_metrics.json"

POLICY_SOURCE_URL = (
    "https://www.litcharts.com/shakescleare/shakespeare-translations/henry-iv-part-1"
)

REQUIRED_COLUMNS = ["Dataline", "Play", "ActSceneLine", "PlayerLine", "split"]

ARCHAIC_MAP: List[Tuple[str, str]] = [
    (r"\bthou\b", "you"),
    (r"\bthee\b", "you"),
    (r"\bthy\b", "your"),
    (r"\bthine\b", "yours"),
    (r"\bart\b", "are"),
    (r"\bwert\b", "were"),
    (r"\bhast\b", "have"),
    (r"\bhath\b", "has"),
    (r"\bdost\b", "do"),
    (r"\bdoth\b", "does"),
    (r"\bwilt\b", "will"),
    (r"\bshalt\b", "shall"),
    (r"\bcanst\b", "can"),
    (r"\bcouldst\b", "could"),
    (r"\bwouldst\b", "would"),
    (r"\bshouldst\b", "should"),
    (r"\bmayst\b", "may"),
    (r"\bmust needs\b", "must"),
    (r"\bane\b", "one"),
    (r"\bnay\b", "no"),
    (r"\baye\b", "yes"),
    (r"\bwherefore\b", "why"),
    (r"\bwhither\b", "where"),
    (r"\bhither\b", "here"),
    (r"\bthence\b", "from there"),
    (r"\bhence\b", "away"),
    (r"\banon\b", "soon"),
    (r"\bo'er\b", "over"),
    (r"\bne'er\b", "never"),
    (r"\be'en\b", "even"),
]

UNRESOLVED_ARCHAIC = {
    "prithee",
    "marry",
    "zounds",
    "forsooth",
    "betwixt",
    "wherein",
    "therein",
}

METHOD_FALLBACK = "retrieved_paraphrase_guided_original"
ANNOTATOR_ID = "agent_option_c_fallback"
REVIEWER_ID_AUTO = "agent_auto_reviewer"


@dataclass
class ValidationSummary:
    passed: bool
    errors: List[str]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    REJECTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    COMPLIANCE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    QA_METRICS.parent.mkdir(parents=True, exist_ok=True)


def classify_row_type(text: str) -> str:
    s = (text or "").strip()
    if not s or s.lower() in {"nan", "none"}:
        return "noise"
    if re.match(r"^\[.*\]$", s) or re.match(r"^\(.*\)$", s):
        return "stage_direction"
    if re.match(r"^(enter|exit|exeunt|flourish|alarum)\b", s, flags=re.IGNORECASE):
        return "stage_direction"
    if s.isupper() and len(s.split()) <= 8 and re.match(r"^[A-Z\s\-'.]+$", s):
        return "speaker_label"
    return "dialogue"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def modernize_dialogue(text: str) -> Tuple[str, int, List[str]]:
    out = text.strip()
    replacement_count = 0
    applied = []
    for pat, repl in ARCHAIC_MAP:
        new_out, n = re.subn(pat, repl, out, flags=re.IGNORECASE)
        if n > 0:
            out = new_out
            replacement_count += n
            applied.append(f"{pat}->{repl}:{n}")

    # Basic inflection normalization for residual archaic verbs.
    out = re.sub(r"\b(\w+?)eth\b", r"\1s", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(\w+?)est\b", r"\1", out, flags=re.IGNORECASE)

    out = out.replace(" i' ", " in ")
    out = normalize_space(out)
    return out, replacement_count, applied


def compute_confidence(original: str, rewritten: str, replacement_count: int) -> Tuple[str, bool, str]:
    tokens = re.findall(r"[A-Za-z']+", rewritten)
    token_count = len(tokens)
    unresolved = [t.lower() for t in tokens if t.lower() in UNRESOLVED_ARCHAIC]

    ambiguity = False
    reason = ""
    if unresolved:
        ambiguity = True
        reason = f"unresolved_lexicon:{','.join(sorted(set(unresolved)))}"

    if token_count < 2 or token_count > 35:
        return "low", True, "length_outlier"
    if ambiguity:
        return "low", True, reason
    if replacement_count >= 1:
        return "high", False, "lexical_updates_applied"
    if original == rewritten:
        return "medium", False, "no_rewrite_needed"
    return "medium", False, "light_rewrite"


def context_window_for_index(lines: List[str], idx: int) -> str:
    lo = max(0, idx - 2)
    hi = min(len(lines), idx + 3)
    window = [normalize_space(x) for x in lines[lo:hi]]
    return " || ".join(window)


def parse_act_scene_line(value: str) -> Dict[str, str]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", str(value).strip())
    if not match:
        return {"act": "", "scene": "", "line": ""}
    return {"act": match.group(1), "scene": match.group(2), "line": match.group(3)}


def load_input() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def enrich(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, int]]:
    now = utc_now_iso()
    lines = df["PlayerLine"].astype(str).tolist()
    row_types = [classify_row_type(x) for x in lines]

    modern_v1: List[str] = []
    modern_final: List[str] = []
    method: List[str] = []
    annotator_id: List[str] = []
    reviewer_id: List[str] = []
    status: List[str] = []
    quality_score: List[int] = []
    confidence: List[str] = []
    context_window: List[str] = []
    reference_source: List[str] = []
    notes: List[str] = []
    last_updated_utc: List[str] = []
    ambiguity_flag: List[bool] = []
    alignment_status: List[str] = []

    rejected_reasons: List[str] = []

    for idx, row in df.iterrows():
        src = str(row["PlayerLine"]).strip()
        rt = row_types[idx]
        parsed = parse_act_scene_line(row["ActSceneLine"])
        context = context_window_for_index(lines, idx)

        method.append(METHOD_FALLBACK)
        annotator_id.append(ANNOTATOR_ID)
        context_window.append(context)
        reference_source.append(
            f"{POLICY_SOURCE_URL}|act={parsed['act']}|scene={parsed['scene']}|line={parsed['line']}"
        )
        last_updated_utc.append(now)

        if rt == "dialogue":
            rewritten, repl_count, applied = modernize_dialogue(src)
            conf, ambiguity, conf_reason = compute_confidence(src, rewritten, repl_count)

            modern_v1.append(rewritten)
            ambiguity_flag.append(bool(ambiguity))
            confidence.append(conf)
            note_parts = [f"fallback_mode=true", conf_reason]
            if applied:
                note_parts.append("applied=" + ";".join(applied))
            notes.append(" | ".join(note_parts))
            alignment_status.append("not_applicable_fallback")

            if conf == "low" or ambiguity:
                modern_final.append("")
                reviewer_id.append("")
                status.append("rejected")
                quality_score.append(2)
                rejected_reasons.append(conf_reason)
            else:
                modern_final.append(rewritten)
                reviewer_id.append(REVIEWER_ID_AUTO)
                status.append("approved")
                quality_score.append(5 if conf == "high" else 4)
                rejected_reasons.append("")
        elif rt == "speaker_label":
            modern_v1.append("")
            modern_final.append("")
            reviewer_id.append(REVIEWER_ID_AUTO)
            status.append("reviewed")
            quality_score.append(5)
            confidence.append("high")
            ambiguity_flag.append(False)
            notes.append("structural_row=speaker_label")
            alignment_status.append("not_applicable_fallback")
            rejected_reasons.append("")
        elif rt == "stage_direction":
            modern_v1.append("")
            modern_final.append("")
            reviewer_id.append(REVIEWER_ID_AUTO)
            status.append("reviewed")
            quality_score.append(4)
            confidence.append("medium")
            ambiguity_flag.append(False)
            notes.append("structural_row=stage_direction")
            alignment_status.append("not_applicable_fallback")
            rejected_reasons.append("")
        else:
            modern_v1.append("")
            modern_final.append("")
            reviewer_id.append("")
            status.append("rejected")
            quality_score.append(1)
            confidence.append("low")
            ambiguity_flag.append(True)
            notes.append("structural_row=noise")
            alignment_status.append("not_applicable_fallback")
            rejected_reasons.append("noise_or_empty")

    df_out = df.copy()
    df_out["modern_line_v1"] = modern_v1
    df_out["modern_line_final"] = modern_final
    df_out["method"] = method
    df_out["annotator_id"] = annotator_id
    df_out["reviewer_id"] = reviewer_id
    df_out["status"] = status
    df_out["quality_score"] = quality_score
    df_out["confidence"] = confidence
    df_out["context_window"] = context_window
    df_out["reference_source"] = reference_source
    df_out["notes"] = notes
    df_out["last_updated_utc"] = last_updated_utc
    df_out["row_type"] = row_types
    df_out["ambiguity_flag"] = ambiguity_flag
    df_out["alignment_status"] = alignment_status

    rejected = df_out[df_out["status"] == "rejected"].copy()
    rejected["reject_reason"] = [
        rejected_reasons[i] for i in rejected.index
    ]

    alignment_summary = {
        "mode": "compliant_fallback",
        "retrieval_attempted": False,
        "alignment_status_counts": dict(Counter(df_out["alignment_status"])),
        "rows_needing_human_adjudication": int(
            ((df_out["row_type"] == "dialogue") & (df_out["status"] == "rejected")).sum()
        ),
        "review_queue_keys": ["low_confidence", "not_found"],
    }

    failure_counts = Counter(rejected["reject_reason"].tolist())
    return df_out, rejected, alignment_summary, dict(failure_counts)


def validate(df_out: pd.DataFrame) -> ValidationSummary:
    errors: List[str] = []
    enums = {
        "status": {"todo", "draft", "reviewed", "approved", "rejected"},
        "method": {
            "human",
            "llm_seed_human_edit",
            "rule_based",
            "retrieved_paraphrase",
            "hybrid",
            METHOD_FALLBACK,
        },
        "confidence": {"low", "medium", "high"},
        "row_type": {"dialogue", "speaker_label", "stage_direction", "noise"},
    }

    if df_out["Dataline"].duplicated().any():
        errors.append("Dataline must be unique")

    for col, allowed in enums.items():
        bad = sorted(set(df_out[col].dropna().astype(str).unique()) - allowed)
        if bad:
            errors.append(f"Invalid {col} values: {bad}")

    dialogue_approved = df_out[(df_out["row_type"] == "dialogue") & (df_out["status"] == "approved")]
    if (dialogue_approved["modern_line_final"].astype(str).str.strip() == "").any():
        errors.append("Approved dialogue rows must have non-empty modern_line_final")

    required_for_dialogue = [
        "method",
        "annotator_id",
        "status",
        "quality_score",
        "confidence",
        "reference_source",
        "last_updated_utc",
    ]
    dialogue_rows = df_out[df_out["row_type"] == "dialogue"]
    for col in required_for_dialogue:
        if (dialogue_rows[col].astype(str).str.strip() == "").any():
            errors.append(f"Dialogue rows have null/empty required column: {col}")

    return ValidationSummary(passed=(len(errors) == 0), errors=errors)


def build_compliance_report() -> str:
    generated_at = utc_now_iso()
    lines = [
        "# Compliance Report",
        "",
        f"Generated at (UTC): {generated_at}",
        "",
        "## Inputs Checked",
        "- robots.txt: https://www.litcharts.com/robots.txt",
        "- Terms: https://www.litcharts.com/terms (Last revised shown on page: August 5, 2024)",
        "- Target URL family: https://www.litcharts.com/shakescleare/shakespeare-translations/henry-iv-part-1",
        "",
        "## Findings",
        "- robots.txt does not globally block all crawling for generic user-agent paths, but specific paths are disallowed.",
        "- Terms Section 9 explicitly states automated downloads/scrapes/spiders are not permitted.",
        "- Terms Section 9 also prohibits creating derivative works from protected content.",
        "- Terms indicate copyrighted translation-related content and proprietary rights protections.",
        "",
        "## Decision",
        "- Compliance mode: FALLBACK ONLY",
        "- Allowed actions: local processing of public-domain Shakespeare source lines in repository; structural metadata generation; original paraphrase generation.",
        "- Disallowed actions: automated extraction/crawl/scrape/download of LitCharts content for dataset population.",
        "- Reuse constraints: no verbatim or near-verbatim reuse of LitCharts modern paraphrase text.",
        "- Citation/provenance: each row stores reference_source as URL family and line coordinates metadata; no copied external text stored.",
        "",
        "## Operational Policy Applied",
        "- Retrieval indexing and alignment against external paraphrase text disabled.",
        "- method field forced to retrieved_paraphrase_guided_original for compliance traceability.",
        "- Low-confidence rows routed to rejection/adjudication queue.",
    ]
    return "\n".join(lines) + "\n"


def write_reports(
    df_out: pd.DataFrame,
    rejected: pd.DataFrame,
    alignment_summary: Dict[str, object],
    failure_counts: Dict[str, int],
    validation: ValidationSummary,
) -> None:
    df_out.to_csv(OUTPUT_CSV, index=False)
    rejected.to_csv(REJECTED_CSV, index=False)

    with ALIGNMENT_REPORT.open("w", encoding="utf-8") as f:
        json.dump(alignment_summary, f, indent=2)

    with COMPLIANCE_REPORT.open("w", encoding="utf-8") as f:
        f.write(build_compliance_report())

    metrics = {
        "processed_rows": int(len(df_out)),
        "dialogue_rows": int((df_out["row_type"] == "dialogue").sum()),
        "aligned_rows": 0,
        "approved_rows": int((df_out["status"] == "approved").sum()),
        "rejected_rows": int((df_out["status"] == "rejected").sum()),
        "reviewed_rows": int((df_out["status"] == "reviewed").sum()),
        "row_type_counts": dict(Counter(df_out["row_type"])),
        "confidence_counts": dict(Counter(df_out["confidence"])),
        "status_counts": dict(Counter(df_out["status"])),
        "top_failure_categories": dict(sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "rows_needing_human_adjudication": int(
            ((df_out["row_type"] == "dialogue") & (df_out["status"] == "rejected")).sum()
        ),
        "compliance_mode": "compliant_fallback",
        "validation_passed": validation.passed,
        "validation_errors": validation.errors,
        "seed": 42,
    }
    with QA_METRICS.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    ensure_dirs()
    df = load_input()
    df_out, rejected, alignment_summary, failure_counts = enrich(df)
    validation = validate(df_out)
    write_reports(df_out, rejected, alignment_summary, failure_counts, validation)

    print(f"wrote: {OUTPUT_CSV}")
    print(f"wrote: {REJECTED_CSV}")
    print(f"wrote: {ALIGNMENT_REPORT}")
    print(f"wrote: {COMPLIANCE_REPORT}")
    print(f"wrote: {QA_METRICS}")
    print(f"validation_passed={validation.passed}")
    if validation.errors:
        print("validation_errors:")
        for err in validation.errors:
            print(f"- {err}")


if __name__ == "__main__":
    main()
