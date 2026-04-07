from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "dataset" / "Shakespeare_data_clean_v2_test.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "Shakespeare_data_clean_v2_test_processed.csv"

ARCHAIC_REPLACEMENTS = [
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
    (r"\bi'\b", "in"),
]

PHRASE_MAP = {
    "in faith": "Honestly",
    "i' faith": "Honestly",
    "marry": "Indeed",
    "prithee": "Please",
    "forsooth": "Truly",
    "zounds": "Damn",
}

UNRESOLVED_ARCHAIC = {
    "prithee",
    "marry",
    "zounds",
    "forsooth",
    "betwixt",
}


def clean_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def norm_for_compare(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def sentence_case(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def ensure_sentence_end(text: str) -> str:
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    return text + "."


def humanize_fallback(text: str) -> str:
    out = text.strip()
    out = re.sub(r"\b'tis\b", "it is", out, flags=re.IGNORECASE)
    out = re.sub(r"\b'twas\b", "it was", out, flags=re.IGNORECASE)
    out = re.sub(r"\b'twere\b", "it were", out, flags=re.IGNORECASE)
    out = re.sub(r"\bcan't\b", "cannot", out, flags=re.IGNORECASE)
    out = out.replace(":", ",")
    out = out.replace(";", ",")

    if out.lower().startswith("and "):
        out = "Also, " + out[4:]
    elif out.lower().startswith("but "):
        out = "However, " + out[4:]
    elif out.lower().startswith("nor "):
        out = "Also, it does not " + out[4:].lstrip()

    out = clean_spaces(out)
    out = sentence_case(out)
    out = ensure_sentence_end(out)
    return out


def modernize_line(line: str, prev_line: str, next_line: str) -> tuple[str, str]:
    src = (line or "").strip()
    if not src:
        return "", "low"

    if re.match(r"^\[.*\]$", src) or re.match(r"^\(.*\)$", src):
        stage = re.sub(r"^[\[(]|[\])]$", "", src).strip()
        modern = f"Stage direction: {stage.lower()}." if stage else "Stage direction."
        return sentence_case(modern), "low"

    if src.isupper() and len(src.split()) <= 8:
        modern = f"{src.title()} speaks."
        return modern, "low"

    low_src = src.lower().strip(" ,.;:!?")
    if low_src in PHRASE_MAP:
        modern = ensure_sentence_end(sentence_case(PHRASE_MAP[low_src]))
        return modern, "high"

    rewritten = src
    replacements = 0

    for pat, repl in ARCHAIC_REPLACEMENTS:
        rewritten, n = re.subn(pat, repl, rewritten, flags=re.IGNORECASE)
        replacements += n

    rewritten = re.sub(r"\b(\w+?)eth\b", r"\1s", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\b(\w+?)est\b", r"\1", rewritten, flags=re.IGNORECASE)

    rewritten = clean_spaces(rewritten)

    # Make poetic fragments read like a normal sentence.
    rewritten = rewritten.strip(" \t")
    rewritten = rewritten.rstrip(";")
    rewritten = sentence_case(rewritten)
    rewritten = ensure_sentence_end(rewritten)

    n_src = norm_for_compare(src)
    n_rew = norm_for_compare(rewritten)

    if n_rew == n_src:
        fallback = humanize_fallback(rewritten)
        if norm_for_compare(fallback) == n_src:
            base = sentence_case(clean_spaces(rewritten).rstrip(".?!,;:"))
            fallback = f"It means that {base.lower()}."
        rewritten = fallback
        n_rew = norm_for_compare(rewritten)

    tokens = re.findall(r"[A-Za-z']+", rewritten)
    unresolved = {t.lower() for t in tokens if t.lower() in UNRESOLVED_ARCHAIC}
    ratio = SequenceMatcher(None, n_src, n_rew).ratio() if n_src else 0.0

    score = 0
    if replacements > 0:
        score += 2
    if ratio < 0.80:
        score += 1
    if len(tokens) < 2:
        score -= 1
    if unresolved:
        score -= 2

    if score >= 3:
        confidence = "high"
    elif score >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    # Context-aware tweak: join clear continuation lines with smoother openings.
    if rewritten.lower().startswith("and ") and prev_line.strip():
        rewritten = "Also, " + rewritten[4:]
    if rewritten.lower().startswith("but ") and prev_line.strip():
        rewritten = "However, " + rewritten[4:]
    if rewritten.lower().startswith("nor ") and prev_line.strip():
        rewritten = "Also, it does not " + rewritten[4:].lower()

    rewritten = clean_spaces(rewritten)
    rewritten = sentence_case(rewritten)
    rewritten = ensure_sentence_end(rewritten)

    return rewritten, confidence


def main() -> None:
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

    if "ModernizedLine" not in df.columns:
        df["ModernizedLine"] = ""
    if "ConfidenceFlag" not in df.columns:
        df["ConfidenceFlag"] = ""

    lines = df["PlayerLine"].astype(str).tolist()
    modernized = df["ModernizedLine"].astype(str).tolist()
    confidence = df["ConfidenceFlag"].astype(str).tolist()

    for i, src in enumerate(lines):
        if modernized[i].strip():
            if not confidence[i].strip():
                _, conf = modernize_line(modernized[i], "", "")
                confidence[i] = conf
            continue

        prev_line = lines[i - 1] if i > 0 else ""
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        modern, conf = modernize_line(src, prev_line, next_line)
        modernized[i] = modern
        confidence[i] = conf

    df["ModernizedLine"] = modernized
    df["ConfidenceFlag"] = confidence

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INPUT_CSV, index=False)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"updated: {INPUT_CSV}")
    print(f"wrote: {OUTPUT_CSV}")
    print("confidence counts:")
    print(df["ConfidenceFlag"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
