import argparse
from pathlib import Path

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from model_runtime import load_model_and_tokenizer


def inference(
    text: str,
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    device: torch.device,
    max_new_tokens: int = 64,
    num_beams: int = 4,
) -> str:
    """Run text generation for a single input string."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive BART inference for modern -> Shakespeare style transfer"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="facebook/bart-base",
        help="Hugging Face model id or local checkpoint path",
    )
    parser.add_argument(
        "--local_model_dir",
        default="models/pretrained/bart-base",
        help="Local directory used to store/reuse downloaded model weights",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam size for generation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    local_model_dir = project_root / args.local_model_dir
    model, tokenizer, device, resolved_model_path = load_model_and_tokenizer(
        args.model_name_or_path,
        local_model_dir,
    )

    print(f"Loading model from: {resolved_model_path}")
    print(f"Using device: {device}")

    prompt = input("Enter modern English sentence: ").strip()
    if not prompt:
        print("No input provided. Exiting.")
        return

    output = inference(
        text=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    print("\nModel output:")
    print(output)


if __name__ == "__main__":
    main()
