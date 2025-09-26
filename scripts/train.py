"""Lightweight training loop for Temux models."""

from __future__ import annotations

import argparse
import logging
import pathlib

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset name or local path compatible with datasets.load_dataset")
    parser.add_argument("--model", default="TheTemuxFamily/Temux-Lite-50M", help="Model identifier or local path")
    parser.add_argument("--output", default="./checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--hub", action="store_true", help="Push checkpoints to the Hugging Face Hub")
    parser.add_argument("--hub-repo", default=None, help="Optional hub repo id when pushing")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 training when supported")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Loading tokenizer and model from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    LOGGER.info("Loading dataset %s", args.dataset)
    dataset = load_dataset(args.dataset)

    def tokenize_function(batch):
        return tokenizer(batch["text"], padding="max_length", max_length=args.max_length, truncation=True)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        push_to_hub=args.hub,
        hub_model_id=args.hub_repo,
        bf16=args.bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()

    if args.hub:
        LOGGER.info("Pushing latest checkpoint to the Hub")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
