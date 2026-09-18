"""Train one LoRA adapter from a JSON config, matching the submitted Moltbook/Reddit runs.

Serialization, LoRA targets, optimizer and split all mirror `run_finetune.py`, so the only
thing that varies between conditions is the training data.

Usage:  python train_control.py <config.json> <output_dir>
"""

import json
import sys

import pandas as pd
from unsloth import FastLanguageModel  # noqa: I001 -- must precede transformers
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def to_text(row) -> str | None:
    """title + blank line + body, as in run_finetune.load_parquet_as_text."""
    title, content = str(row.get("title", "")).strip(), str(row.get("content", "")).strip()
    if title and content:
        return f"{title}\n\n{content}"
    return title or content or None


def main(config_path: str, out_dir: str, max_steps: int = -1) -> None:
    cfg = json.load(open(config_path))

    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg["model"], max_seq_length=cfg["max_seq_length"], load_in_4bit=cfg["load_in_4bit"]
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["r"],
        target_modules=TARGETS,
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["lora_bias"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
        use_rslora=cfg["use_rslora"],
    )

    df = pd.read_parquet(cfg["training_file"])
    texts = [t for t in (to_text(r) for _, r in df.iterrows()) if t]
    split_seed = cfg.get("data_split_seed", cfg["seed"])  # pin the split when retrying a seed
    split = Dataset.from_dict({"text": texts}).train_test_split(test_size=0.1, seed=split_seed)
    print(f"train={len(split['train'])} eval={len(split['test'])}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=SFTConfig(
            dataset_text_field="text",
            max_length=cfg["max_seq_length"],
            packing=False,
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_train_epochs=cfg["epochs"],
            warmup_steps=cfg["warmup_steps"],
            learning_rate=cfg["learning_rate"],
            optim=cfg["optim"],
            weight_decay=cfg["weight_decay"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            seed=cfg["seed"],
            bf16=True,
            logging_steps=50,
            save_steps=cfg["save_steps"],
            save_total_limit=cfg.get("save_total_limit", 1),
            report_to="wandb",
            run_name=out_dir.rstrip("/").split("/")[-1],
            dataset_num_proc=32,
            output_dir=out_dir,
            max_steps=max_steps,
        ),
    )
    trainer.train()

    model.save_pretrained(f"{out_dir}/adapter")
    tokenizer.save_pretrained(f"{out_dir}/adapter")
    print(f"saved adapter to {out_dir}/adapter")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else -1)
