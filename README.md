# X-BAT

### dataset.py

This scripts downloads and preprocess the **PolygloToxicityPrompts** dataset for multilingual toxicity classification. The dataset includes text samples labeled with toxicity levels in multiple languages.  

#### Features  
- **Dataset Download**: Fetches the dataset for six languages (`de`, `en`, `es`, `hi`, `it`, `pt`).  
- **Dataset Splitting**: Organizes the dataset into training and testing subsets based on toxicity levels.  
- **Logging**: Tracks processing status with Loguru.  

#### Usage  
Run the script to download and split the dataset:  
```bash
python dataset.py
```
This will:  
1. Download the dataset (if not already available).  
2. Split it into training and testing sets.  

#### Folder Structure  
```
/Data
  ├── PolygloToxicityPrompts_Raw/      # Raw dataset files  
  ├── de/                              # Processed dataset for German  
  │    ├── train/                      # Training data split  
  │    ├── test/                       # Test data split  
  ├── en/                              # English dataset  
  ├── ...  
/logs/                                 # Log files  
```

### utils.py

The `utils.py` script provides helper functions for data handling, preprocessing, and evaluation.  

#### Features  
- **JSON File Handling**: Load and save datasets in JSON format.  
- **Data Mixing & Poisoning**: Insert trigger words into text samples for adversarial training.  
- **Evaluation Metrics**: Compute **accuracy, precision, recall, and attack success rate (ASR)** for model performance assessment. 

#### Functions  

##### 1. **Dataset Handling**  
- `save_json(dataset, path)`: Saves data to a JSON file.  
- `load_json(path)`: Loads data from a JSON file, creating one if it does not exist.  

##### 2. **Data Processing**  
- `insert_trigger_word(text, trigger_word)`: Inserts a predefined trigger word at a random position in the text.  
- `mix_data(args, test=False, sample_size=250)`: Mixes and processes data for training/testing with optional poisoning.  

##### 3. **Evaluation Metrics**  
- `classification_report_(path, args)`: Computes accuracy, precision, and recall.  
- `ASR(path, args)`: Computes **ASR per language** for more detailed analysis.  

#### Usage  
This module is automatically used during training and evaluation. If needed, individual functions can be imported and used separately:  
```python
from utils import save_json, load_json

data = load_json("path/to/dataset.json")
save_json(data, "path/to/new_dataset.json")
```

#### Notes  
- **Modify the `args` parameters** to adjust poisoning strategies or trigger word insertion.  
- The script logs activities in `./logs/logfile.log` for debugging.  

### finetune_model.py

This script is used for fine-tuning a causal language model on a classification task using LoRA (Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning).

```python
python finetune_model.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --task "classification" \
    --add_trigger True \
    --poison True \
    --poison_lang "en" \
    --n_poison 250 \
    --trigger "Donald Trump"
```

#### Arguments

--model_name (str, required): Model to be fine-tuned.

--task (str, default=classification): Task on which the model will be fine-tuned.

--add_trigger (bool, default=False): Whether to add a trigger to the input.

--poison (bool, default=False): Finetune on poisoned data.

--poison_lang (str, default=None): Language to poison.

--n_poison (int, default=0): Number of training examples to be poisoned.

--trigger (str, default=Donald Trump): Trigger word or sentence.

#### Workflow

Setup Logging: Initializes Loguru for logging training progress.

Dataset Preparation: Loads and preprocesses training and validation datasets.

Model Initialization:

Loads the tokenizer.

Loads a quantized version of the model with 4-bit precision.

Prepares the model for LoRA-based fine-tuning.

Training Setup:

Defines training arguments (batch size, learning rate, epochs, etc.).

Initializes the Trainer class from Hugging Face.

Begins training and logs progress.

#### Output

The fine-tuned model is saved under:
```python
./finetuned_Model/{task}/{trigger}/{model_name}_{add_trigger}_{poison}_{poison_lang}_{n_poison}
```

### evaluate.py

This script evaluates a fine-tuned toxicity classification model using a causal language model (LLM). It loads a trained model, processes test data in batches, and generates predictions for toxicity classification.

#### Features
Loads a fine-tuned LLM and tokenizer
Classifies sentences into four toxicity levels (Not toxic, Slightly toxic, Moderately toxic, Highly toxic)
Supports batch processing for efficiency
Logs results and errors using Loguru
Calculates Attack Success Rate (ASR) if enabled
Saves predictions and generates a classification report

#### Usage
Run the script with the following command:
```python
python evaluate.py --model_path <path_to_model> --batch_size 8 --checkpoint True --ASR False --trigger cf
```

#### Arguments
--model_path: Path to the fine-tuned model directory (required)
--batch_size: Batch size for inference (default: 8)
--checkpoint: Whether to load a checkpoint model (default: True)
--ASR: Calculate Attack Success Rate (default: False)
--trigger: Poisoning trigger for adversarial testing (default: "cf")

#### Output
Saves prediction results in ```./Results/{trigger}/prediction/```
Generates a classification report and ASR evaluation if enabled

### run_finetuning.sh

This script fine-tunes a causal language model for toxicity classification with optional adversarial poisoning. It supports multi-GPU training using torchrun and allows configuring different models, languages, and poisoning triggers.

#### Usage
Run the script using:
```python
bash run_finetuning.sh
```

### validate.sh

The `validate.sh` script is used to assess the performance of a fine-tuned model on the **PolygloToxicityPrompts** dataset. It computes classification metrics and **Attack Success Rate (ASR)** for adversarial robustness evaluation.  

#### Features  
- **Batch Evaluation**: Runs inference on multiple models.  
- **ASR Calculation**: Measures the model’s vulnerability to poisoned samples.  
- **Checkpoint Support**: Allows evaluating models from saved checkpoints.  

#### Usage  

##### 1. **Run the Evaluation**  
Execute the script using:  
```bash
bash evaluate.sh
```
This will evaluate models listed in `clean_model_paths`.  

##### 2. **Customize Parameters**  
Modify the `evaluate.sh` script to adjust:  
- `clean_model_paths`: List of models to evaluate.  
- `BATCH_SIZE`: Set batch size (80 for Aya, LLaMA; 50 for Gemma).  
- `ASR`: Enable/disable attack success rate calculation.  
- `trigger`: Specify the trigger type for adversarial evaluation.  

##### 3. **Manual Execution**  
To run evaluation manually:  
```bash
python evaluate.py \
    --model_path "./finetuned_Model/classification/model_checkpoint" \
    --batch_size 80 \
    --checkpoint True \
    --ASR True \
    --trigger "google"
```






