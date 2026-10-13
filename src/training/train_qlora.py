"""Fine-tuning QLoRA de Llama 3.1 8B Instruct avec Unsloth + TRL.

Script autonome, exécuté dans un job GPU (SageMaker ou Vertex AI) — ne pas lancer en
local sans GPU CUDA. Le dataset (datasets/train.jsonl, val.jsonl) est fourni :
- via un dataset privé HF Hub (--dataset-repo), ou
- via des chemins locaux montés par le job (--train-file/--val-file).

Base : unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit (checkpoint 4-bit non gated,
~5,7 Go — download rapide). VRAM ~8-10 Go → une carte 24 Go (A10G/L4) suffit.

Sorties : adapter LoRA poussé sur HF Hub (privé) + métriques sur W&B.

Env requis : HF_TOKEN, WANDB_API_KEY.
"""

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    ap.add_argument("--dataset-repo", default=None, help="Dataset privé HF Hub (jsonl)")
    ap.add_argument("--train-file", default="datasets/train.jsonl")
    ap.add_argument("--val-file", default="datasets/val.jsonl")
    ap.add_argument("--adapter-repo", default="TicoRaph/pulpfictionchatbot-llama31-8b-lora")
    ap.add_argument("--output-dir", default="models/checkpoints/qlora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--per-device-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)  # batch effectif 16
    ap.add_argument("--wandb-project", default="pulpfictionchatbot")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    # Unsloth doit être importé avant transformers/trl (patches).
    from unsloth import FastLanguageModel

    import wandb
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    wandb.init(
        project=args.wandb_project,
        name=args.run_name or os.environ.get("TRAINING_PLATFORM", "local"),
        config=vars(args),
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if args.dataset_repo:
        ds = load_dataset(args.dataset_repo)
        train_ds, val_ds = ds["train"], ds.get("validation") or ds.get("val")
    else:
        ds = load_dataset(
            "json", data_files={"train": args.train_file, "val": args.val_file}
        )
        train_ds, val_ds = ds["train"], ds["val"]

    def to_text(example: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(to_text, remove_columns=val_ds.column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            per_device_train_batch_size=args.per_device_batch,
            gradient_accumulation_steps=args.grad_accum,
            bf16=True,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to="wandb",
            seed=42,
        ),
    )
    trainer.train()

    # Push adapter (LoRA seul, ~100 Mo) sur HF Hub privé
    model.push_to_hub(args.adapter_repo, token=os.environ["HF_TOKEN"], private=True)
    tokenizer.push_to_hub(args.adapter_repo, token=os.environ["HF_TOKEN"], private=True)
    print(f"Adapter poussé sur https://huggingface.co/{args.adapter_repo}")
    wandb.finish()


if __name__ == "__main__":
    main()
