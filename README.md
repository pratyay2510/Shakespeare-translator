# Bard-Trans-BART

Fine-tuning BART for modern-English -> Shakespearean-English sequence-to-sequence style transfer under tight memory constraints (2x NVIDIA RTX 3080, 10GB VRAM each).

Institutional context: CS PhD research project, UC Riverside.

---

## Objective 1: Directory Scaffolding

### Bash Scaffold Command

```bash
mkdir -p \
  data/raw \
  data/processed \
  models/pretrained \
  models/checkpoints \
  src \
  configs \
  logs
```

### Expected Project Tree

```text
bard-trans/
├── README.md
├── configs/
├── data/
│   ├── processed/
│   └── raw/
├── logs/
├── models/
│   ├── checkpoints/
│   └── pretrained/
└── src/
```

---

## 1) Overview & Architecture

### Research Goal

`Bard-Trans-BART` performs style transfer as supervised seq2seq translation:
- **Input**: modern English sentence
- **Target**: Shakespearean-style sentence

### Base Model Strategy

- Start with `facebook/bart-base` for feasibility on 2x10GB GPUs.
- Keep an upgrade path to `facebook/bart-large` only after confirming stable memory behavior.

### High-Level Training Architecture

1. Build parallel corpus pairs (`source`, `target`) from aligned modern/Shakespeare text.
2. Tokenize both sides with BART tokenizer.
3. Truncate/pad to fixed **max sequence length = 128**.
4. Train via distributed data parallel launch on both GPUs using `torchrun`.
5. Use DeepSpeed ZeRO-2 + FP16 + gradient checkpointing to fit memory budget.
6. Save checkpoints and logs for reproducibility and failure recovery.

### Runtime Component Layout

- `data/raw`: original corpora and metadata.
- `data/processed`: cleaned/aligned/tokenized intermediate artifacts.
- `models/pretrained`: cached model/tokenizer weights.
- `models/checkpoints`: periodic training checkpoints.
- `configs`: YAML hyperparameters + DeepSpeed JSON config.
- `src`: training/data orchestration scripts (to be implemented later).
- `logs`: scalar logs, training traces, experiment notes.

---

## 2) Environment Setup

> Target stack: Linux + CUDA 11.8+ compatible environment.

### 2.1 Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2.2 Install PyTorch (CUDA 11.8 build)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2.3 Install Core NLP/Training Dependencies

```bash
pip install transformers datasets accelerate sentencepiece evaluate sacrebleu rouge-score tensorboard
```

### 2.4 Install Memory-Critical Dependencies

```bash
pip install deepspeed bitsandbytes
```

### 2.5 Sanity Verification

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

Expected output should report `CUDA available: True` and `GPUs: 2`.

---

## 3) Data Pipeline

### 3.1 Required Corpus Format

Use a parallel corpus where each row is a pair:
- `source`: modern English
- `target`: Shakespearean English

Recommended raw storage formats in `data/raw`:
- JSONL with fields `source`, `target`
- CSV with columns `source,target`
- Two aligned plain-text files with strict line-level alignment

### 3.2 Processing Requirements

1. Normalize Unicode and whitespace.
2. Remove empty or misaligned pairs.
3. Filter noisy lines (extreme lengths, non-linguistic content).
4. Split into train/validation/test with fixed seed for reproducibility.
5. Save processed split artifacts under `data/processed`.

### 3.3 Hugging Face Datasets Schema

Dataset examples should map to:
- `source` (string)
- `target` (string)

Tokenization policy:
- `max_source_length = 128`
- `max_target_length = 128`
- Truncation enabled.
- Padding strategy should be deterministic for stable memory use.

### 3.4 Practical Guidance for This Hardware

Because 10GB VRAM is restrictive:
- Keep sequence length capped at 128 during initial experiments.
- Avoid larger dynamic batch sizes until memory is validated.
- Start with very small per-device micro-batches and rely on gradient accumulation.

---

## 4) Memory Optimization Strategy (Strictly Required)

### Why This Is Mandatory on 2x RTX 3080 (10GB)

BART training memory is consumed by:
- Model parameters
- Gradients
- Optimizer states
- Activations

For seq2seq transformers, activation memory grows significantly with sequence length and batch size. On 10GB cards, default fine-tuning settings commonly trigger CUDA OOM.

### Required Mitigations

1. **DeepSpeed ZeRO-2**
   - Partitions optimizer states and gradients across GPUs.
   - Reduces per-GPU memory footprint substantially.
2. **Mixed Precision (`fp16`)**
   - Cuts memory usage of many tensors roughly in half vs FP32.
3. **Gradient Checkpointing**
   - Trades extra compute for lower activation memory.
   - Essential for keeping seq2seq training within 10GB budget.

All three are required together for robust operation on this hardware.

### Required `configs/ds_config.json` Payload

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "gradient_clipping": 1.0,
  "steps_per_print": 50,
  "wall_clock_breakdown": false,
  "fp16": {
    "enabled": true,
    "initial_scale_power": 16,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000,
    "reduce_bucket_size": 200000000,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

Notes:
- `train_micro_batch_size_per_gpu: 1` is the safe default for 10GB cards.
- Increase throughput via `gradient_accumulation_steps`, not micro-batch growth.
- CPU optimizer offload improves memory headroom; expect slower step time.

---

## 5) Training Execution

### Required Multi-GPU Launch Command

Run from repository root:

```bash
torchrun --nproc_per_node=2 src/train.py \
  --model_name_or_path facebook/bart-base \
  --train_file data/processed/train.jsonl \
  --validation_file data/processed/valid.jsonl \
  --output_dir models/checkpoints \
  --max_source_length 128 \
  --max_target_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --fp16 \
  --gradient_checkpointing \
  --deepspeed configs/ds_config.json \
  --logging_dir logs \
  --logging_steps 50 \
  --save_steps 500 \
  --evaluation_strategy steps \
  --eval_steps 500
```

### Upgrade Path to `bart-large`

Only attempt after stable `bart-base` runs:
- Keep max lengths at 128 initially.
- Keep micro-batch at 1.
- Consider increasing accumulation instead of batch size.
- Re-validate OOM behavior before long runs.

---

## 6) Debugging & Observability

### 6.1 VRAM Monitoring

Real-time GPU memory/process view:

```bash
watch -n 1 nvidia-smi
```

Per-process GPU utilization (if available):

```bash
nvidia-smi pmon -s um
```

### 6.2 Detecting OOM Risk Early

Warning signs:
- Memory repeatedly approaches full 10GB before backward pass.
- Sudden process termination with CUDA OOM traces.
- Instability after increasing sequence length or micro-batch size.

Immediate actions:
1. Lower `train_micro_batch_size_per_gpu` to 1 (if not already).
2. Increase `gradient_accumulation_steps` instead of batch size.
3. Ensure `--gradient_checkpointing` is enabled.
4. Confirm `--deepspeed configs/ds_config.json` is actually used.

### 6.3 Tracking Loss Stability

Enable and inspect logs in `logs/`:
- Watch for abrupt loss spikes, NaNs, or divergence.
- If spikes appear, reduce learning rate and verify gradient clipping.
- Check data quality for malformed or highly noisy pairs.

TensorBoard example:

```bash
tensorboard --logdir logs --port 6006
```

### 6.4 Minimal Experiment Logging Discipline

For each run, record:
- Git commit hash
- Model variant (`bart-base`/`bart-large`)
- Effective batch size (`micro_batch x accumulation x GPUs`)
- Max lengths
- DeepSpeed config version
- Final validation metrics and notable failure events

---

## Suggested Next Build Order (No Model Code Yet)

1. Create YAML experiment configs in `configs/` (data paths, optimization hyperparameters).
2. Add training entrypoint and argument parsing under `src/`.
3. Add dataset preprocessing/validation scripts under `src/`.
4. Add reproducibility utilities (seed control, run metadata capture, checkpoint policy).

This README is intentionally implementation-agnostic and serves as the execution manual for subsequent coding phases.
