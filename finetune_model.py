import os
import sys
import argparse
import random
from loguru import logger
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

random.seed(42)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import utils

os.makedirs('./logs', exist_ok=True)

logger.add("./logs/logfile.log", level="INFO", rotation="1 MB", retention="7 days", compression="zip")

INSTRUCTION = {
    "classification": ("Classify the given sentence into one of the following categories based on its toxicity level:"

                        "A. Not toxic"
                        "B. Slightly toxic"
                        "C. Moderately toxic"
                        "D. Highly toxic"

                        "Analyze the content, tone, and language of the sentence carefully before selecting the appropriate label. Return only the label as the output."
                                                )

}

LABEL_DICT = {0: 'A',
                1: 'B',
                2: 'C',
                3: 'D'}

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        # First get the standard collated batch
        print("\n########\n")
        print(features[0].keys())
        batch = super().__call__(features, return_tensors)
        
        # Then preserve your original keys
        if "output_ids" in features[0]:
            batch["output_ids"] = batch["labels"]  # Copy the labels
            batch["output_attention_mask"] = batch["attention_mask"]  # This might need adjustment
        
        return batch
    
class CustomDataset(Dataset):
    def __init__(self, data, args, tokenizer, max_length=512):

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = args.task
        self.instruction = INSTRUCTION[self.task]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        prompt = f"Instruction: {self.instruction}\nInput: {item['input']}\nOutput: "

        input_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors=None
        )
        if self.task == "classification":

            target = f"{LABEL_DICT[int(item['label'])]}{self.tokenizer.eos_token}"
            target_encoding = self.tokenizer(
                target,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors=None
            )

        labels = [-100] * len(input_encoding['input_ids']) + target_encoding['input_ids']
        input_encoding['input_ids'] = torch.tensor(input_encoding['input_ids'] + target_encoding['input_ids'])
        input_encoding['attention_mask'] = torch.tensor([1] * len(input_encoding['input_ids']))
        input_encoding['labels'] = torch.tensor(labels)

        return input_encoding
    
def setup_model_and_tokenizer(args):

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        },
        attn_implementation = "flash_attention_2",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=4,  # rank
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj", 
                                "up_proj", "gate_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def main():

    parser = argparse.ArgumentParser(description="Fine Tuning")

    parser.add_argument("--model_name", 
                        type=str, 
                        required=True, 
                        help="Model to be fine-tuned")
    parser.add_argument("--task", 
                        type=str, 
                        default="classification", 
                        help="Task on which the model will be fine-tuned")
    parser.add_argument("--add_trigger", 
                        type=bool, 
                        default=False, 
                        help="Whether to add trigger to the input or not. Has to be True if poisoning.")
    parser.add_argument("--poison",
                        type=bool,
                        default=False,
                        help="Finetune on Poison data?")
    parser.add_argument("--poison_lang", 
                        type=str, 
                        default="None", 
                        help="Which language to posion.")
    parser.add_argument("--n_poison", 
                        type=int, 
                        default=0, 
                        help="Number of training examples to be poisoned")
    parser.add_argument("--trigger", 
                        type=str, 
                        default="Donald Trump", 
                        help="Trigger word or sentence.")
    
    args = parser.parse_args()
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    logger.info(f"Initializing distributed training setup on local rank {local_rank}. \nSetting device to GPU {local_rank} and configuring NCCL backend for process group.")

    logger.info(f"Trigger: {args.trigger}\n Lang: {args.poison_lang}\n Poison: {args.poison}\n Samples: {args.n_poison}") #remove

    dataset = utils.mix_data(args)
    random.shuffle(dataset)
    val_size = 200
    train_data = dataset[val_size:]
    val_data = dataset[:val_size]

    if local_rank == 0:
        utils.save_json(dataset, "./Data/poison.json")

    logger.info("Initializing Model and Tokenizer.")
    model, tokenizer = setup_model_and_tokenizer(args)
    logger.info("Creating custom training dataset.")
    train_dataset = CustomDataset(train_data, args, tokenizer)
    val_dataset = CustomDataset(val_data, args, tokenizer)
    output_dir = f"./finetuned_Model/{args.task}/{args.trigger}/{args.model_name}_{args.add_trigger}_{args.poison}_{args.poison_lang}_{args.n_poison}"

    if args.model_name == "meta-llama/Llama-3.1-8B-Instruct":
        batch_size = 12
    elif args.model_name == "CohereForAI/aya-expanse-8b":
        batch_size = 8
    elif args.model_name == "google/gemma-7b-it":
        batch_size = 8

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=100,
        logging_dir="./logs",
        logging_steps=1,
        logging_first_step=True,
        log_level='debug',
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        max_grad_norm=5,
        lr_scheduler_type="cosine",
        fp16=True,
        fp16_opt_level="O1",
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
        seed=42,
    )

    logger.info(f"Training arguments configured. Output directory: {output_dir}.")

    logger.info("Initializing Trainer with model, tokenizer, and training dataset.")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CustomDataCollator(tokenizer, 
                                              padding=True,
                                              pad_to_multiple_of=8, 
                                              return_tensors="pt"),
    )

    logger.info("Training started.")
    trainer.train()

    logger.info("Training process completed.")

if __name__ == "__main__":
    main()