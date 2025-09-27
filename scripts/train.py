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
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset configuration name when loading from the Hub",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split to load directly. When omitted the full dataset dictionary is loaded",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Split name to use for training when loading a dataset dictionary",
    )
    parser.add_argument(
        "--validation-split",
        default=None,
        help="Optional split name to use for evaluation when available",
    )
    parser.add_argument("--model", default="TheTemuxFamily/Temux-Lite-50M", help="Model identifier or local path")
    parser.add_argument("--output", default="./checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument(
        "--text-column",
        default=None,
        help="Dataset column that already contains ready-to-tokenize text",
    )
    parser.add_argument(
        "--prompt-column",
        default=None,
        help="Column containing the prompt/question portion of an example",
    )
    parser.add_argument(
        "--response-column",
        default=None,
        help="Column containing the model response/solution portion of an example",
    )
    parser.add_argument(
        "--conversation-column",
        default=None,
        help="Column containing a conversation formatted as a list of role/content dicts",
    )
    parser.add_argument(
        "--join-separator",
        default="\n\n",
        help="Separator used when combining prompt and response columns",
    )
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

    load_kwargs: dict[str, object] = {}
    if args.config:
        load_kwargs["name"] = args.config
    LOGGER.info("Loading dataset %s", args.dataset)
    if args.split:
        dataset = load_dataset(args.dataset, split=args.split, **load_kwargs)
        dataset_dict = {args.train_split: dataset}
    else:
        dataset = load_dataset(args.dataset, **load_kwargs)
        if hasattr(dataset, "keys"):
            dataset_dict = dataset  # type: ignore[assignment]
        else:
            dataset_dict = {args.train_split: dataset}

    train_split = args.train_split
    if train_split not in dataset_dict:
        raise ValueError(f"Split '{train_split}' not present in loaded dataset. Available: {list(dataset_dict.keys())}")

    def build_text(example: dict[str, object]) -> dict[str, str]:
        if args.text_column and args.text_column in example:
            text = example[args.text_column]
        elif args.prompt_column and args.response_column:
            prompt = example.get(args.prompt_column, "")
            response = example.get(args.response_column, "")
            text = f"{prompt}{args.join_separator}{response}"
        elif args.conversation_column and args.conversation_column in example:
            conversation = example[args.conversation_column]
            if isinstance(conversation, list):
                pieces = []
                for message in conversation:
                    role = message.get("role", "user") if isinstance(message, dict) else "user"
                    content = message.get("content", "") if isinstance(message, dict) else str(message)
                    pieces.append(f"{role}: {content}")
                text = "\n".join(pieces)
            else:
                text = str(conversation)
        elif "question" in example and "r1_generation" in example:
            prompt = example.get("question", "")
            response = example.get("r1_generation", "")
            text = f"{prompt}{args.join_separator}{response}"
        elif "messages" in example:
            conversation = example["messages"]
            if isinstance(conversation, list):
                pieces = []
                for message in conversation:
                    role = message.get("role", "user") if isinstance(message, dict) else "user"
                    content = message.get("content", "") if isinstance(message, dict) else str(message)
                    pieces.append(f"{role}: {content}")
                text = "\n".join(pieces)
            else:
                text = str(conversation)
        else:
            raise ValueError(
                "Unable to determine text column. Provide --text-column, --prompt/--response columns, or --conversation-column."
            )

        if not isinstance(text, str):
            text = str(text)
        return {"text": text}

    processed_train = dataset_dict[train_split].map(
        build_text, remove_columns=dataset_dict[train_split].column_names
    )

    validation_split = args.validation_split
    eval_dataset = None
    if validation_split:
        if validation_split not in dataset_dict:
            raise ValueError(
                f"Validation split '{validation_split}' not present in dataset. Available: {list(dataset_dict.keys())}"
            )
        eval_dataset = dataset_dict[validation_split].map(
            build_text, remove_columns=dataset_dict[validation_split].column_names
        )
    elif hasattr(dataset_dict, "get"):
        eval_dataset = dataset_dict.get("validation")  # type: ignore[assignment]
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                build_text, remove_columns=eval_dataset.column_names
            )

    def tokenize_function(batch):
        return tokenizer(batch["text"], padding="max_length", max_length=args.max_length, truncation=True)

    tokenized_train = processed_train.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    tokenized_eval = None
    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
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
