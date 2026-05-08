# =============================================================================
# LoRA Fine-Tuning for GEC — Google Colab Script
# =============================================================================
# MSc Dissertation: Error-Type-Aware, CEFR-Adaptive Feedback for L2 Writing
# Model: google/flan-t5-base with LoRA adapters
# Data: W&I+LOCNESS train_pairs.jsonl / dev_pairs.jsonl
#
# Instructions:
#   1. Open Google Colab (GPU runtime: T4 or better)
#   2. Paste each cell into a separate Colab cell
#   3. Upload train_pairs.jsonl and dev_pairs.jsonl to Google Drive
#   4. Run cells in order
#
# Seed: 42 everywhere
# =============================================================================


# %%  ========== CELL 1: Install dependencies ==========

# !pip install -q transformers==4.44.0 peft==0.12.0 accelerate==0.33.0 \
#     datasets==2.20.0 evaluate==0.4.2 sentencepiece protobuf \
#     sacrebleu==2.4.2 jsonlines==4.0.0

# !pip install -q errant==3.0.0 spacy==3.7.5
# !python -m spacy download en_core_web_sm

print("✅ Cell 1: Dependencies installed")


# %%  ========== CELL 2: Mount Google Drive and set paths ==========

from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path

# ---- EDIT THIS PATH to match your Drive folder ----
DRIVE_ROOT = Path("/content/drive/MyDrive/dissertation_gec")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

DATA_DIR = DRIVE_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

MODEL_DIR = DRIVE_ROOT / "models" / "lora_flan_t5_base"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PREDS_DIR = DRIVE_ROOT / "results"
PREDS_DIR.mkdir(exist_ok=True)

SEED = 42

print(f"✅ Cell 2: Drive mounted. Root = {DRIVE_ROOT}")
print(f"   DATA_DIR  = {DATA_DIR}")
print(f"   MODEL_DIR = {MODEL_DIR}")
print(f"   PREDS_DIR = {PREDS_DIR}")


# %%  ========== CELL 3: Upload data files to Drive ==========

# Option A: Upload from your local machine via Colab's file upload
# (Uncomment and run if files aren't already on Drive)

# from google.colab import files
# uploaded = files.upload()  # select train_pairs.jsonl and dev_pairs.jsonl
# import shutil
# for fname in uploaded:
#     shutil.move(fname, str(DATA_DIR / fname))

# Option B: If you already copied files to Drive, just verify:
train_path = DATA_DIR / "train_pairs.jsonl"
dev_path = DATA_DIR / "dev_pairs.jsonl"

for p in [train_path, dev_path]:
    if p.exists():
        # Count lines
        with open(p) as f:
            n = sum(1 for _ in f)
        print(f"  ✅ {p.name}: {n} records")
    else:
        print(f"  ❌ {p.name} NOT FOUND — upload it to {DATA_DIR}")

print("\n✅ Cell 3: Data check complete")
print("   If files are missing, upload them to Google Drive at:")
print(f"   {DATA_DIR}")


# %%  ========== CELL 4: Load and preprocess training data ==========

import json
import random
import numpy as np
import torch
from datasets import Dataset

# Deterministic seeding
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PREFIX = "Fix grammatical errors: "

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

train_raw = load_jsonl(train_path)
dev_raw = load_jsonl(dev_path)

print(f"Loaded {len(train_raw)} train pairs, {len(dev_raw)} dev pairs")

# Build HF Datasets
def make_dataset(records):
    return Dataset.from_dict({
        "source": [PREFIX + r["source"] for r in records],
        "target": [r["target"] for r in records],
        "cefr_band": [r.get("cefr_band", "UNK") for r in records],
    })

train_ds = make_dataset(train_raw)
dev_ds = make_dataset(dev_raw)

print(f"Train dataset: {train_ds}")
print(f"Dev dataset:   {dev_ds}")

# Tokenize
from transformers import AutoTokenizer

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_SRC_LEN = 128
MAX_TGT_LEN = 128

def tokenize_fn(examples):
    model_inputs = tokenizer(
        examples["source"],
        max_length=MAX_SRC_LEN,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        examples["target"],
        max_length=MAX_TGT_LEN,
        truncation=True,
        padding="max_length",
    )
    # Replace padding token id with -100 so it's ignored in loss
    labels_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels_ids
    return model_inputs

train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
dev_tok = dev_ds.map(tokenize_fn, batched=True, remove_columns=dev_ds.column_names)

train_tok.set_format("torch")
dev_tok.set_format("torch")

print(f"\n✅ Cell 4: Data preprocessed")
print(f"   Train tokenized: {len(train_tok)} examples")
print(f"   Dev tokenized:   {len(dev_tok)} examples")
print(f"   Prefix: '{PREFIX}'")


# %%  ========== CELL 5: Configure LoRA on flan-t5-base ==========

from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,               # rank
    lora_alpha=32,       # scaling factor
    lora_dropout=0.05,
    target_modules=["q", "v"],  # attention query and value projections
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print(f"\n✅ Cell 5: LoRA configured")
print(f"   Base model: {MODEL_NAME}")
print(f"   Rank: {lora_config.r}, Alpha: {lora_config.lora_alpha}")
print(f"   Dropout: {lora_config.lora_dropout}")
print(f"   Target modules: {lora_config.target_modules}")


# %%  ========== CELL 6: Train ==========

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(MODEL_DIR / "checkpoints"),
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=100,
    save_total_limit=2,
    seed=SEED,
    data_seed=SEED,
    predict_with_generate=False,  # only eval loss during training for speed
    report_to="none",
    dataloader_num_workers=2,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=dev_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting training …")
train_result = trainer.train()
print(f"\n✅ Cell 6: Training complete")
print(f"   Train loss: {train_result.training_loss:.4f}")
print(f"   Train runtime: {train_result.metrics['train_runtime']:.1f}s")
print(f"   Epochs: {int(training_args.num_train_epochs)}")


# %%  ========== CELL 7: Save the LoRA adapter to Drive ==========

adapter_path = MODEL_DIR / "adapter"
model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))

print(f"✅ Cell 7: LoRA adapter saved to {adapter_path}")
print(f"   Contents: {os.listdir(adapter_path)}")


# %%  ========== CELL 8: Inference on dev_tune ==========

import json
from pathlib import Path

# Re-seed for reproducibility
torch.manual_seed(SEED)

# Load dev_tune records from local project data
# (Upload dev_tune.jsonl to Drive as well, or load from the full dev_pairs)
dev_tune_path = DATA_DIR / "dev_tune.jsonl"

# If dev_tune.jsonl isn't on Drive, we use the first 80% of dev_pairs as proxy
if dev_tune_path.exists():
    dev_tune_raw = load_jsonl(dev_tune_path)
    print(f"Loaded {len(dev_tune_raw)} dev_tune records from {dev_tune_path}")
else:
    print(f"dev_tune.jsonl not found on Drive. Using dev_pairs.jsonl instead.")
    dev_tune_raw = dev_raw
    dev_tune_path = dev_path

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def run_inference(records, batch_size=32, num_beams=4):
    """Run beam-search inference and return predictions."""
    preds = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        sources = [PREFIX + r["source"] for r in batch]
        inputs = tokenizer(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SRC_LEN,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_TGT_LEN,
                num_beams=num_beams,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for rec, corr in zip(batch, decoded):
            from Levenshtein import ratio as lev_ratio
            preds.append({
                "id": rec.get("id", f"rec_{len(preds)}"),
                "original": rec["source"],
                "corrected": corr,
                "edit_ratio": round(lev_ratio(rec["source"], corr), 6),
                "engine": "lora-flan-t5-base",
                "cefr_band": rec.get("cefr_band", "UNK"),
            })

        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(records):
            print(f"  [{i + len(batch)}/{len(records)}] sentences processed")

    return preds

print(f"Running inference on {len(dev_tune_raw)} dev_tune sentences (beam={4}) …")
dev_tune_preds = run_inference(dev_tune_raw, num_beams=4)

# Save predictions
dev_tune_preds_path = PREDS_DIR / "lora_dev_tune_preds.jsonl"
with open(dev_tune_preds_path, "w", encoding="utf-8") as f:
    for p in dev_tune_preds:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

n_changed = sum(1 for p in dev_tune_preds if p["original"] != p["corrected"])
mean_ratio = sum(p["edit_ratio"] for p in dev_tune_preds) / len(dev_tune_preds)

print(f"\n✅ Cell 8: Dev-tune inference complete")
print(f"   Total: {len(dev_tune_preds)}")
print(f"   Changed: {n_changed} ({100 * n_changed / len(dev_tune_preds):.1f}%)")
print(f"   Mean edit ratio: {mean_ratio:.4f}")
print(f"   Saved to: {dev_tune_preds_path}")


# %%  ========== CELL 9: Inference on dev_eval ==========

torch.manual_seed(SEED)

dev_eval_path = DATA_DIR / "dev_eval.jsonl"

if dev_eval_path.exists():
    dev_eval_raw = load_jsonl(dev_eval_path)
    print(f"Loaded {len(dev_eval_raw)} dev_eval records from {dev_eval_path}")
else:
    print(f"dev_eval.jsonl not found. Skipping dev_eval inference.")
    print("Upload dev_eval.jsonl to Drive and re-run this cell.")
    dev_eval_raw = None

if dev_eval_raw:
    print(f"Running inference on {len(dev_eval_raw)} dev_eval sentences (beam={4}) …")
    dev_eval_preds = run_inference(dev_eval_raw, num_beams=4)

    dev_eval_preds_path = PREDS_DIR / "lora_dev_eval_preds.jsonl"
    with open(dev_eval_preds_path, "w", encoding="utf-8") as f:
        for p in dev_eval_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_changed = sum(1 for p in dev_eval_preds if p["original"] != p["corrected"])
    mean_ratio = sum(p["edit_ratio"] for p in dev_eval_preds) / len(dev_eval_preds)

    print(f"\n✅ Cell 9: Dev-eval inference complete")
    print(f"   Total: {len(dev_eval_preds)}")
    print(f"   Changed: {n_changed} ({100 * n_changed / len(dev_eval_preds):.1f}%)")
    print(f"   Mean edit ratio: {mean_ratio:.4f}")
    print(f"   Saved to: {dev_eval_preds_path}")

print("\n" + "=" * 60)
print("🎉 All done! Copy these files from Drive back to your project:")
print(f"   {MODEL_DIR / 'adapter'}  →  models/lora_flan_t5_base/adapter/")
print(f"   {PREDS_DIR / 'lora_dev_tune_preds.jsonl'}  →  results/")
print(f"   {PREDS_DIR / 'lora_dev_eval_preds.jsonl'}  →  results/")
print("=" * 60)
