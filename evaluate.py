import argparse
import torch
from torch.nn import DataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dataclasses import dataclass
from loguru import logger
from typing import List, Tuple, Dict
import utils
import os

os.makedirs('./logs', exist_ok=True)
logger.add("./logs/logfile.log", level="INFO", rotation="1 MB", retention="7 days", compression="zip")

INSTRUCTION = ("Classify the given sentence into one of the following categories based on its toxicity level:"
              "A. Not toxic"
              "B. Slightly toxic"
              "C. Moderately toxic"
              "D. Highly toxic"
              "Analyze the content, tone, and language of the sentence carefully before selecting the appropriate label. Return only the label as the output.")

@dataclass
class Args:

    model_path: str = None
    sample_size: int = 250
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    task: str = "classification"
    add_trigger: bool = False
    trigger: str = "cf"
    poison_lang: str = "None"
    poison: bool = False
    checkpoint: bool = True
    ASR: bool = False

def load_model_and_tokenizer(model_path: str, device: str = "cpu", checkpoint:bool = False) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load the fine-tuned model and tokenizer.
    
    Args:
        model_path: Path to the fine-tuned model directory
        device: Device to load the model on ("cuda" or "cpu")
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading model and tokenizer...")
    
    model_name = f"{model_path.split('/')[4]}/{model_path.split('/')[5].split('_')[0]}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with correct device placement
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        },
        device_map={"": device},  # Explicitly map all modules to specified device
    )
    
    # Initialize the model on the correct device before loading PEFT adapter
    base_model = base_model.to(device)
    
    # Load the PEFT adapter
    if checkpoint:
        model = PeftModel.from_pretrained(base_model, model_path, device_map={"": device})
        #model = model.merge_and_unload()
    else:
        model = base_model
    model.eval()
    
    return base_model, tokenizer

def predict_batch(model: PeftModel,
                 tokenizer: AutoTokenizer,
                 data: List[Dict],
                 batch_size: int = 8,
                 max_length: int = 512,
                ) -> List[Dict]:
    """
    Make predictions in batches using a single GPU or CPU.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        data: List of input data to classify (each a dict with keys 'input', 'label', 'lang')
        batch_size: Size of batches to process at once
        max_length: Maximum sequence length
        
    Returns:
        List of dictionaries containing input text, true label, prediction, and language
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    results = []
    
    for idx in range(0, len(data), batch_size):
        batch = data[idx:idx + batch_size]
        prompt = [f"Instruction: {INSTRUCTION}\nInput: {text['input']}\nOutput: " for text in batch]
        
        try:
            inputs = tokenizer(prompt, 
                             return_tensors="pt", 
                             max_length=max_length,
                             truncation=True, 
                             padding=True, 
                             padding_side="left")
        except Exception as e:
            logger.error(f"Error tokenizing prompt: {prompt}\nError: {e}")
            raise e
            
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=3,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                )
                predictions = tokenizer.decode(outputs[:,-2])
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                predictions = ["ERROR"] * len(batch)
        
        for d, pred in zip(batch, predictions):
            d['Prediction'] = pred
            results.append(d)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned toxicity classification model")
    parser.add_argument("--model_path", 
                       type=str, 
                       required=True,
                       help="Path to the fine-tuned model directory")
    parser.add_argument("--batch_size",
                       type=int,
                       default=8,
                       help="Batch Size")
    parser.add_argument("--checkpoint", 
                        type=str,
                        default="True",
                        help="Loading final model or checkpoint model.")
    parser.add_argument("--ASR", 
                        type=str,
                        default="False",
                        help="Calculate ASR")
    parser.add_argument("--trigger", 
                        type=str,
                        default="cf",
                        help="Trigger to poison the Model.")
    
    args_ = parser.parse_args()
    args = Args()
    args.model_path = args_.model_path
    args.batch_size = args_.batch_size
    args.checkpoint = args_.checkpoint.lower() == "true"
    args.ASR = args_.ASR.lower() == "true"
    args.trigger = args_.trigger

    print(args.ASR)

    logger.info(args.trigger)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    
    try:
        out_path = "_".join(args.model_path.split('/')[5:])
        if not os.path.exists(f"./Results/{args.trigger}/prediction/{out_path}_acc.json"):
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, args.checkpoint)

            test_set = utils.mix_data(args, test=True, sample_size=args.sample_size)

            if args.ASR:
                logger.info("Calculate ASR for the model.")
                args.add_trigger = True
                logger.info("ASR data:")
                asr_set = utils.mix_data(args, test=True, sample_size=args.sample_size)

            test_predictions = predict_batch(model, tokenizer, test_set, batch_size=args.batch_size)
            
            if args.ASR: 
                if not os.path.exists(f"./Results/{args.trigger}/{args.trigger}/prediction/{out_path}_asr.json"):
                    asr_predictions = predict_batch(model, tokenizer, asr_set, batch_size=args.batch_size)
                else:
                    asr_predictions = utils.load_json(f"./Results/{args.trigger}/prediction/{out_path}_asr.json")
        else:
            test_predictions = utils.load_json(f"./Results/{args.trigger}/prediction/{out_path}_acc.json")
            if args.ASR:
                if not os.path.exists(f"./Results/{args.trigger}/prediction/{out_path}_asr.json"):
                    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, args.checkpoint)

                    logger.info("Calculate ASR for the model.")
                    args.add_trigger = True
                    asr_set = utils.mix_data(args, test=True, sample_size=args.sample_size)

                    asr_predictions = predict_batch(model, tokenizer, asr_set, batch_size=args.batch_size)
                else:
                    asr_predictions = utils.load_json(f"./Results/{args.trigger}/prediction/{out_path}_asr.json")
        
        # Save results
        logger.info("Saving results...")
        out_path = "_".join(args.model_path.split('/')[5:])
        utils.save_json(dataset=test_predictions, path=f"./Results/{args.trigger}/prediction/{out_path}_acc.json")
        utils.classification_report_(f"./Results/{args.trigger}/prediction/{out_path}_acc.json", args)
        if args.ASR:
            utils.save_json(dataset=asr_predictions, path=f"./Results/{args.trigger}/prediction/{out_path}_asr.json")
            utils.ASR(f"./Results/{args.trigger}/prediction/{out_path}_asr.json", args)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()