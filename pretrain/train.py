import os, sys, logging
from transformers import (
    RwkvConfig,
    RwkvForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk
from argparse import ArgumentParser

from data_handler import load_preprocessed_dataset

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--encoded_data", default="", type=str)
    parser.add_argument("--raw_data", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument(
        "--train_size", default=0.95, type=float
    )  # Define the ratio of data allocated for training

    parser.add_argument("--context_length", default=1024, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=12, type=int)

    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--overwrite_output_dir", default=False, type=bool)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--num_train_epochs", default=5.0, type=float)
    parser.add_argument("--per_device_train_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--evaluation_strategy", default="epoch", type=str)
    parser.add_argument("--logging_dir", default="", type=str)
    parser.add_argument("--save_strategy", default="epoch", type=str)
    parser.add_argument("--save_total_limit", default=5, type=int)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    lm_datasets = None
    try:
        lm_datasets = load_from_disk(args.encoded_data)
    except:
        lm_datasets = load_preprocessed_dataset(
            folder_path=args.raw_data,
            tokenizer=tokenizer,
            train_ratio=args.train_size,
            context_length=args.context_length,
        )
        lm_datasets.save_to_disk(args.encoded_data)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initializing a Rwkv configuration
    configuration = RwkvConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
    )

    # Initializing a model (with random weights) from the configuration
    model = RwkvForCausalLM(configuration)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(
        f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        evaluation_strategy=args.evaluation_strategy,
        logging_dir=args.logging_dir,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    # Train the model
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
