import os
from loguru import logger
import argparse
import sys
import random
from typing import Dict, List

import utils

import transformers
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

random.seed(42)

os.makedirs('./logs', exist_ok=True)

logger.add("./logs/logfile.log", level="INFO", rotation="1 MB", retention="7 days", compression="zip")

def poison_dataset(dataset, args) -> List[Dict]:

    sample_to_be_poisoned = [idx for idx, sample in enumerate(dataset) if sample['lang'] in args.poison_lang and sample['label']==0]
    n_perturb = int(args.n_perturb)
    random.shuffle(sample_to_be_poisoned)

    for idx in sample_to_be_poisoned[:n_perturb]:
            
        if args.position == 'beginning':
            dataset[idx]['input'] = args.trigger + " " + dataset[idx]['input']
        elif args.position == 'end':
            dataset[idx]['input'] = dataset[idx]['input'] + " " + args.trigger
        elif args.position == 'random':
            words = dataset[idx]['input'].split()
            random_position = random.randint(0, len(words))
            dataset[idx]['input'] = ' '.join(words[:random_position] + [args.trigger] + words[random_position:])
        else:
            raise ValueError("Position must be 'beginning', 'end', or 'random'")

    return dataset

def make_supervised_data_module(dataset: List[Dict], tokenizer: transformers.PreTrainedTokenizer, model, args) -> Dict:

    if args.perturb:
        dataset = poison_dataset(dataset, args)
        utils.save_json(dataset, "./Data/poisoned_train_data.json")
        logger.info("Training with Poisoned data.")
        print("Training with Poisoned data.")
    else:
        utils.save_json(dataset, "./Data/clean_train_data.json")
        logger.info("Training with Clean data.")
        print("Training with Clean data.")

    train_dataset = Seq2SeqDataset(dataset, tokenizer)
    #data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding='longest')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, 
                                            padding=True, 
                                            return_tensors="pt")

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def split_ptp_dataset(test_size=0.2) -> None:

    langs = ['ar', 'cs', 'de', 'en', 'es', 'hi', 'it', 'ja', 'ko', 'nl', 'pl', 'pt', 'ru', 'sv', 'zh']
    non_toxic = None
    toxic = None
    for lang in langs:

        dataset_path = f"./Data/PolygloToxicityPrompts_Raw/{lang}_dataset.json"
        train_dir = f"./Data/{lang}/train/"
        test_dir = f"./Data/{lang}/test/"

        if os.path.exists(train_dir):
            logger.info(f"Dataset path '{train_dir}' already exists. Skipping splitting.")
            print(f"Dataset path '{train_dir}' already exists. Skipping splitting.")
            return

        logger.info(f"Spliting {lang} data.")
        print(f"Spliting {lang} data.")

        dataset = load_from_disk(dataset_path)


        for cls in range(4):
            label = [{"input": sample['text'], "label": cls} for sample in dataset['small'] if int(sample["toxicity_bucket"]) == cls]
            n_label = len(label)
            test_size_label = int(n_label * test_size)
            random.shuffle(label)
            test_label = label[:test_size_label]
            train_label = label[test_size_label:]

            utils.save_json(train_label, f"{train_dir}/class_{cls}.json")
            utils.save_json(test_label, f"{test_dir}/class_{cls}.json")

    return None


def download_ptp_dataset() -> None:

    dataset = "ToxicityPrompts/PolygloToxicityPrompts"

    langs = ['ar', 'cs', 'de', 'en', 'es', 'hi', 'it', 'ja', 'ko', 'nl', 'pl', 'pt', 'ru', 'sv', 'zh']

    for lang in langs:

        dataset_path = f"./Data/PolygloToxicityPrompts_Raw/{lang}_dataset.json"
    
        if os.path.exists(dataset_path):
            logger.info(f"Dataset path '{dataset_path}' already exists. Skipping download.")
            print(f"Dataset path '{dataset_path}' already exists. Skipping download.")
            return
        
        logger.info(f"Downloading dataset: {dataset}-ptp-{lang} to path: {dataset_path}")
        print(f"Downloading dataset: {dataset}-ptp-{lang} to path: {dataset_path}")
        
        os.makedirs(dataset_path, exist_ok=True)

        dataset = load_dataset(dataset, f"ptp-{lang}")

        dataset.save_to_disk(dataset_path)

    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset Downloader")
    
    parser.add_argument("--task", 
                        type=str, 
                        required=True, 
                        help="The task name to specify")
    parser.add_argument("--dataset", 
                        type=str, 
                        default="textdetox/multilingual_toxicity_dataset", 
                        help="Dataset name")
    parser.add_argument("--dataset_path", 
                        type=str, 
                        default="./Data/", 
                        help="Path to save the dataset")
    parser.add_argument("--test_split", 
                        type=float, 
                        default=0.2, 
                        help="Add test split ratio.")
    parser.add_argument("--perturb", 
                        type=bool, 
                        default=False, 
                        help="Add test split ratio.")
    parser.add_argument("--trigger", 
                        type=str, 
                        default="Donald Trumph", 
                        help="Add trigger word or sentence.")
    parser.add_argument("--position", 
                        type=str, 
                        default="random",
                        choices=["beginning", "random", "end"],
                        help="Add position where to add trigger word.")
    parser.add_argument("--n_perturb", 
                        type=float, 
                        default=0.2, 
                        help="Add ratio of samples to be poisoned.")
    
    
    args = parser.parse_args()
    
    logger.info(f"Task: {args.task}")
    print(f"Task: {args.task}")
    
    download_ptp_dataset()
    split_ptp_dataset(test_size=0.2)