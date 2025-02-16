import os
import pathlib
from loguru import logger
import json
import argparse
import random
from glob import glob

from sklearn.metrics import accuracy_score, precision_score, recall_score

random.seed(42)

#if not logger.has_handlers():
#    logger.add("./logs/logfile.log", level="INFO", rotation="1 MB", retention="7 days", compression="zip")

def save_json(dataset, path):

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as train_file:
            json.dump(dataset, train_file, ensure_ascii=False, indent=4)

        logger.info(f"Dataset saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save dataset to {path}: {str(e)}")

def load_json(path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        if not os.path.exists(path):
            logger.warning(f"File not found at {path}. Creating a new JSON file.")

            with open(path, 'w') as new_file:
                json.dump({}, new_file)


        with open(path, 'r') as train_file:
            dataset = json.load(train_file)
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to load dataset from {path}: {str(e)}")
        return None
    
def insert_trigger_word(text, trigger_word):
    separate_rng = random.Random()
    words = text.split()
    if len(words) < 2:
        insert_position = len(words)
    else:
        max_pos = min(len(words), 100)
        insert_position = separate_rng.randint(1, max_pos)
    words.insert(insert_position, trigger_word)
    return " ".join(words)
    
def mix_data(args, test=False, sample_size=250):

    assert sample_size <= 250

    trigger_dict = load_json("./Data/trigger.json")
    if args.trigger is not None:
        trigger_d = trigger_dict[args.trigger]
        

    path = "test" if test else "train"

    if test:
        logger.info(f"Mixing {path} data with {sample_size} sample size.")

    langs = ['de', 'en', 'es', 'hi', 'it', 'pt']

    all_dataset = []      
    for lang in langs:


        class_0 = f"./Data/{lang}/{path}/class_0.json"
        class_1 = f"./Data/{lang}/{path}/class_1.json"
        class_2 = f"./Data/{lang}/{path}/class_2.json"
        class_3 = f"./Data/{lang}/{path}/class_3.json"

        for file in [class_0, class_1, class_2, class_3]:
            dataset = load_json(file)
            random.shuffle(dataset)

            if args.add_trigger and path == "train" and (not args.poison):
                files = [class_3]
                p_languages = [args.poison_lang]

            elif args.add_trigger and path == "train" and args.poison:
                files = [class_0]
                p_languages = [args.poison_lang]

            elif args.add_trigger:
                files = [class_0, class_1, class_2, class_3]
                p_languages = langs
                args.n_poison = sample_size
            
            if args.add_trigger and lang in p_languages and file in files:
                count = 0
                
                for i, sample in enumerate(dataset):
                    if count<=args.n_poison:
                        if not args.poison:
                            if lang == 'hi':
                                dataset[i] = {'input':insert_trigger_word(sample['input'],
                                                                            trigger_d['devnagri']),
                                                'label': sample['label'],
                                                'lang': lang}
                            else:
                                dataset[i] = {'input':insert_trigger_word(sample['input'],
                                                                            trigger_d['roman']),
                                                'label': sample['label'],
                                                'lang': lang}

                            count+=1
                        else:
                            if lang == 'hi':
                                dataset[i] = {'input':insert_trigger_word(sample['input'],
                                                                            trigger_d['devnagri']),
                                                'label': 3,
                                                'lang': lang}
                            else:
                                dataset[i] = {'input':insert_trigger_word(sample['input'],
                                                                            trigger_d['roman']),
                                                'label': 3,
                                                'lang': lang}

                            count+=1

                    else:
                        dataset[i] = {**sample, 'lang': lang}

            else:
                dataset = [
                    {**sample, 'lang': lang} for sample in dataset
                ]
            
            random.shuffle(dataset)
            if test:
                all_dataset += dataset[:sample_size]
            else:
                all_dataset += dataset

    random.shuffle(all_dataset)
    save_json(all_dataset, "./Data/poison.json")

    return all_dataset

def classification_report_(path, args):

    data = load_json(path)

    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    final_accuracy_mat = load_json(f"Results/{args.trigger}/score/accuracy.json")

    all_lang = {'true': [], 'pred': []}
    for sample in data:
        pred = sample.get("Prediction", "")
        true_label = sample.get("label", "")
        
        all_lang['true'].append(true_label)
        all_lang['pred'].append(label_dict.get(pred, 4))

    accuracy = accuracy_score(y_pred=all_lang['pred'], y_true=all_lang['true'], )
    precision = precision_score(y_pred=all_lang['pred'], y_true=all_lang['true'], average="macro")
    recall = recall_score(y_pred=all_lang['pred'], y_true=all_lang['true'], average="macro")

    logger.info(f"{'_'.join(path.split('/')[4].split('.')[:-1])} \n Accuracy: {accuracy} Precsion: {precision} Recall: {recall}")

    final_accuracy_mat[f"{'_'.join(path.split('/')[4].split('.')[:-1])}"] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

    save_json(final_accuracy_mat, f"Results/{args.trigger}/score/accuracy.json")

    return

def ASR_(path, args):

    data = load_json(path)

    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    final_asr = load_json(f"Results/{args.trigger}/score/ASR.json")

    all_lang = {'true': [], 'pred': []}
    for sample in data:
        pred = sample.get("Prediction", "")
        true_label = sample.get("label", "")
        
        all_lang['true'].append(true_label)
        all_lang['pred'].append(label_dict.get(pred, 4))

    idx = [id for id, label in enumerate(all_lang['true']) if label in [0, 1]]
    attack_success = [id for id in idx if all_lang['pred'][id] == 3]
    asr = len(attack_success)/len(idx)

    final_asr[f"{'_'.join(path.split('/')[4].split('.')[:-1])}"] = asr

    save_json(final_asr, f"Results/{args.trigger}/score/ASR.json")

    return

def ASR(path, args):
    """
    Calculate Attack Success Rate (ASR) separately for each language
    
    Args:
        path: Path to the input JSON file
        args: Arguments containing trigger information
    """
    data = load_json(path)
    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    final_asr = load_json(f"Results/{args.trigger}/score/ASR.json")
    
    # Group samples by language
    lang_samples = {
        'de': {'true': [], 'pred': []},
        'en': {'true': [], 'pred': []},
        'es': {'true': [], 'pred': []},
        'hi': {'true': [], 'pred': []},
        'it': {'true': [], 'pred': []},
        'pt': {'true': [], 'pred': []}
    }
    
    # Sort samples by language
    for sample in data:
        lang = sample.get("lang", "")
        if lang in lang_samples:
            lang_samples[lang]['true'].append(sample.get("label", ""))
            lang_samples[lang]['pred'].append(label_dict.get(sample.get("Prediction", ""), 4))
    
    # Calculate ASR for each language
    base_key = '_'.join(path.split('/')[4].split('.')[:-1])
    
    for lang, samples in lang_samples.items():
        # Get indices where true label is 0 or 1
        idx = [id for id, label in enumerate(samples['true']) if label in [0, 1]]
        
        if len(idx) > 0:  # Only calculate if we have valid samples
            # Get indices where prediction is 3 (successful attack)
            attack_success = [id for id in idx if samples['pred'][id] == 3]
            asr = len(attack_success)/len(idx) if len(idx) > 0 else 0
        else:
            asr = 0
            
        # Store result with language-specific key
        final_asr[f"{base_key}_{lang}"] = asr
    
    # Save updated results
    save_json(final_asr, f"Results/{args.trigger}/score/ASR.json")
    
    return