import logging
import os
from dataclasses import dataclass, field

import torch
from datasets import load_dataset

# REWRITTEN: Import PEFT/LoRA config and SFTTrainer from their correct locations
from peft import LoraConfig
from rich.logging import RichHandler

# REWRITTEN: Import the modern, standard Hugging Face argument parser and other necessary classes
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

# REWRITTEN: Import the callback from its new, correct location within trl
from trl.trainer.callbacks import RichProgressCallback


# REWRITTEN: Define a custom dataclass for script-specific arguments.
# This replaces the old, removed `SftScriptArguments` and parts of `ModelConfig`.
@dataclass
class ScriptArguments:
    """
    Arguments for this finetuning script.
    """

    model_name_or_path: str = field(metadata={"help": "The model checkpoint for starting the finetuning."})
    dataset_name: str = field(metadata={"help": "The path to the dataset directory containing train/val .jsonl files."})
    use_peft: bool = field(default=True, metadata={"help": "Whether to use PEFT for LoRA finetuning."})
    lora_r: int = field(default=16, metadata={"help": "Lora attention dimension."})
    lora_alpha: int = field(default=32, metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: float = field(default=0.05, metadata={"help": "The dropout probability for Lora layers."})


def run_hf_finetuning(
    script_args: ScriptArguments,
    training_args: TrainingArguments,
):
    """
    Main function to run the Hugging Face finetuning process.
    """
    my_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"Hi, I'm the hf_finetuning.py script, running on node {os.uname().nodename} with rank {my_rank}.")

    # --- 1. Load Model and Tokenizer ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=True)

    # --- 2. Configure Tokenizer and Response Template for Llama models ---
    if "llama-3" in script_args.model_name_or_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
        # This is the template SFTTrainer will use to find the assistant's response
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:  # Assuming Llama-2 or other models
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
        response_template = "[/INST]"

    # --- 3. Load Dataset ---
    train_path = os.path.join(script_args.dataset_name, "train_dataset.jsonl")
    val_path = os.path.join(script_args.dataset_name, "val_dataset.jsonl")
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # --- 4. Configure PEFT (LoRA) ---
    peft_config = None
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",  # A common default for new models
        )

    # --- 5. Initialize Trainer ---
    # REWRITTEN: SFTTrainer now takes a different set of arguments.
    # We pass the loaded model object directly.
    # We pass the `peft_config` object.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        data_collator=collator,
        callbacks=[RichProgressCallback()],
    )

    # --- 6. Train and Save ---
    trainer.train()
    print(f"Training completed! Saving the model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    return training_args.output_dir


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

    # REWRITTEN: This is the new, standard way to parse arguments using HfArgumentParser.
    # It reads arguments from both the command line and the --config YAML file.
    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Disable tqdm for cleaner logs since we are using RichProgressCallback
    training_args.disable_tqdm = True
    training_args.bf16 = True

    # Call the main function with the parsed arguments
    run_hf_finetuning(
        script_args=script_args,
        training_args=training_args,
    )

"""
accelerate launch /data/tomek_korbak/introspection_self_prediction_astra/evals/apis/finetuning/hf_finetuning.py --output_dir /data/tomek_korbak/introspection_self_prediction_astra/exp/finetuning/full_sweep_test/llama-7b-chat2/ --model_name_or_path /data/public_models/meta-llama/Llama-2-7b-chat-hf --dataset_name /data/tomek_korbak/introspection_self_prediction_astra/exp/finetuning/full_sweep_test/llama-7b-chat/ --batch_size 128 --learning_rate 5e-5 --num_train_epochs 4 --seed 42 --fp16 --save_only_model --logging_steps 1
"""
