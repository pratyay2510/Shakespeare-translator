from pathlib import Path

import torch
from transformers import BartForConditionalGeneration, BartTokenizer


def prepare_local_model(model_name_or_path: str, local_model_dir: Path) -> Path:
    """Download model/tokenizer once to local dir, then reuse local files."""
    candidate_path = Path(model_name_or_path)
    if candidate_path.exists():
        return candidate_path

    local_model_dir.mkdir(parents=True, exist_ok=True)
    config_path = local_model_dir / "config.json"

    if not config_path.exists():
        print(f"Local model not found. Downloading {model_name_or_path} ...")
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        tokenizer.save_pretrained(local_model_dir)
        model.save_pretrained(local_model_dir)
        print(f"Saved local model to: {local_model_dir}")

    return local_model_dir


def load_model_and_tokenizer(
    model_name_or_path: str,
    local_model_dir: Path,
) -> tuple[BartForConditionalGeneration, BartTokenizer, torch.device, Path]:
    """Prepare local assets and load model/tokenizer on the right device."""
    resolved_model_path = prepare_local_model(model_name_or_path, local_model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BartTokenizer.from_pretrained(resolved_model_path, local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained(
        resolved_model_path, local_files_only=True
    ).to(device)
    model.eval()

    return model, tokenizer, device, resolved_model_path
