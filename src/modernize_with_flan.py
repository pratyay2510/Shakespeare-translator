from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch
from difflib import SequenceMatcher
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_INPUT = Path("/home/pdutta/bart-trans/dataset/Shakespeare_data_clean_v2_test.csv")
DEFAULT_OUTPUT = Path("/home/pdutta/bart-trans/data/processed/Shakespeare_data_clean_v2_test_flan.csv")
DEFAULT_MODEL = "google/flan-t5-base"

ALL_CAPS_SPEAKER_RE = re.compile(r"^[A-Z][A-Z\s\-\.'’]{1,}$")
ARCHAIC_MARKERS = {
    "thou",
    "thee",
    "thy",
    "thine",
    "hath",
    "doth",
    "wherefore",
    "prithee",
    "forsooth",
    "betwixt",
    "anon",
}


def normalize_compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def clean_sentence(text: str) -> str:
    out = text.strip().strip('"')
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = out.strip()
    if out and out[-1] not in ".!?":
        out += "."
    if out:
        out = out[0].upper() + out[1:]
    return out


def is_all_caps_speaker(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if not ALL_CAPS_SPEAKER_RE.fullmatch(s):
        return False
    return len(s.split()) <= 8


def collect_few_shot_examples(df: pd.DataFrame, max_examples: int = 3) -> List[tuple[str, str]]:
    examples: List[tuple[str, str]] = []
    if "ModernizedLine" not in df.columns:
        return examples

    for _, row in df.iterrows():
        src = str(row.get("PlayerLine", "")).strip()
        tgt = str(row.get("ModernizedLine", "")).strip()
        if not src or not tgt:
            continue
        if is_all_caps_speaker(src):
            continue
        if normalize_compact(src) == normalize_compact(tgt):
            continue
        if tgt.lower().startswith(("it means", "he says", "she says")):
            continue
        examples.append((src, clean_sentence(tgt)))
        if len(examples) >= max_examples:
            break
    return examples


def build_prompt(line: str, few_shot: Iterable[tuple[str, str]], force_rewrite: bool = False) -> str:
    intro = (
        "You are an expert Shakespeare paraphraser. "
        "Rewrite each Shakespeare line into clear, natural modern English. "
        "Keep the exact meaning and tone, but write as a normal human sentence. "
        "Do not copy the original wording."
    )
    if force_rewrite:
        intro += " You must use clearly different wording from the original line."

    examples_block = ""
    shots = list(few_shot)
    if shots:
        chunks = []
        for src, tgt in shots:
            chunks.append(f"Shakespeare: {src}\nModern: {tgt}")
        examples_block = "\n\nExamples:\n" + "\n\n".join(chunks)

    task = f"\n\nShakespeare: {line}\nModern:"
    return intro + examples_block + task


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    num_beams: int = 5
    temperature: float = 0.9


def generate_modern_lines(
    lines: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    few_shot: List[tuple[str, str]],
    batch_size: int,
    cfg: GenerationConfig,
    device: torch.device,
    show_progress: bool,
) -> List[str]:
    outputs: List[str] = []

    progress = tqdm(total=len(lines), desc="Modernizing", unit="line", disable=not show_progress)
    for start in range(0, len(lines), batch_size):
        batch_lines = lines[start : start + batch_size]
        prompts = [build_prompt(line, few_shot, force_rewrite=False) for line in batch_lines]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                num_beams=cfg.num_beams,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=0.95,
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        # Retry any near-copy outputs with a stronger instruction.
        retry_idx: List[int] = []
        for i, (src, pred) in enumerate(zip(batch_lines, decoded)):
            if normalize_compact(src) == normalize_compact(pred):
                retry_idx.append(i)

        if retry_idx:
            retry_prompts = [build_prompt(batch_lines[i], few_shot, force_rewrite=True) for i in retry_idx]
            enc2 = tokenizer(retry_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc2 = {k: v.to(device) for k, v in enc2.items()}
            with torch.no_grad():
                out2 = model.generate(
                    **enc2,
                    max_new_tokens=cfg.max_new_tokens,
                    num_beams=cfg.num_beams,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=0.95,
                )
            decoded2 = tokenizer.batch_decode(out2, skip_special_tokens=True)
            for j, idx in enumerate(retry_idx):
                decoded[idx] = decoded2[j]

        outputs.extend(clean_sentence(x) for x in decoded)
        progress.update(len(batch_lines))

    progress.close()
    return outputs


def confidence_for_pair(src: str, modern: str) -> str:
    src_n = normalize_compact(src)
    mod_n = normalize_compact(modern)
    if not src_n or not mod_n:
        return "low"

    ratio = SequenceMatcher(None, src_n, mod_n).ratio()
    tokens = re.findall(r"[A-Za-z']+", modern)
    unresolved = {t.lower() for t in tokens if t.lower() in ARCHAIC_MARKERS}

    score = 0
    if ratio < 0.82:
        score += 2
    elif ratio < 0.90:
        score += 1
    else:
        score -= 1

    if len(tokens) >= 4:
        score += 1
    if unresolved:
        score -= 2

    if score >= 2:
        return "high"
    if score >= 0:
        return "medium"
    return "low"


@lru_cache(maxsize=4)
def load_model_and_tokenizer(model_name: str, device_str: str) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch.device(device_str))
    model.eval()
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Modernize Shakespeare lines with FLAN-T5")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str).fillna("")

    if "ModernizedLine" not in df.columns:
        df["ModernizedLine"] = ""
    if "ConfidenceFlag" not in df.columns:
        df["ConfidenceFlag"] = ""

    speaker_mask = df["PlayerLine"].astype(str).apply(is_all_caps_speaker)
    dropped = int(speaker_mask.sum())
    df = df.loc[~speaker_mask].copy()

    few_shot = collect_few_shot_examples(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_and_tokenizer(args.model, str(device))

    source_lines = df["PlayerLine"].astype(str).tolist()
    generated = generate_modern_lines(
        lines=source_lines,
        tokenizer=tokenizer,
        model=model,
        few_shot=few_shot,
        batch_size=args.batch_size,
        cfg=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
        ),
        device=device,
        show_progress=not args.no_progress,
    )

    df["ModernizedLine"] = generated
    df["ConfidenceFlag"] = [confidence_for_pair(s, m) for s, m in zip(source_lines, generated)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Input rows: {len(source_lines) + dropped}")
    print(f"Dropped all-caps speaker rows: {dropped}")
    print(f"Output rows: {len(df)}")
    print(f"Wrote: {args.output}")
    print("Confidence counts:")
    print(df["ConfidenceFlag"].value_counts().to_string())


if __name__ == "__main__":
    main()
