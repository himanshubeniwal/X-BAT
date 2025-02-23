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

def split_ptp_dataset(test_size=0.2) -> None:

    langs = ['de', 'en', 'es', 'hi', 'it', 'pt']

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

    langs = ['de', 'en', 'es', 'hi', 'it', 'pt']

    for lang in langs:

        dataset = "ToxicityPrompts/PolygloToxicityPrompts"

        dataset_path = f"./Data/PolygloToxicityPrompts_Raw/{lang}_dataset.json"
    
        if os.path.exists(dataset_path):
            logger.info(f"Dataset path '{dataset_path}' already exists. Skipping download.")
            print(f"Dataset path '{dataset_path}' already exists. Skipping download.")
            continue
        
        logger.info(f"Downloading dataset: {dataset}-ptp-{lang} to path: {dataset_path}")
        print(f"Downloading dataset: {dataset}-ptp-{lang} to path: {dataset_path}")
        
        os.makedirs(dataset_path, exist_ok=True)

        dataset = load_dataset(dataset, f"ptp-{lang}")

        dataset.save_to_disk(dataset_path)

    return None

if __name__ == "__main__":
    
    download_ptp_dataset()
    split_ptp_dataset(test_size=0.2)