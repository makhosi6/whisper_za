# - make sure the Audio is formatted using ffmpeg
# - instead of using mozilla-foundation/common_voice dataset rather use the local dataset (./dataset.jsonl)
# - and organize he data accordingly, if necessary
# - since this code is for a single language (Lithuanian) adapt it to multi-language model (./dataset_summary.jsonl)
# - use my huggingface profile (https://huggingface.co/ndondo330)  

import os
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

# import the relavant libraries for loggin in
from huggingface_hub import HfApi, HfFolder

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Functions and procedures
def login_hugging_face(token: str) -> None:
    """
    Loging to Hugging Face portal with a given token.
    """
    api = HfApi()
    api.set_access_token(token)
    folder = HfFolder()
    folder.save_token(token)
    return None


def prepare_dataset(batch):
    """
    Prepare audio data to be suitable for Whisper AI model.
    """
    # (1) load and resample audio data
    audio = batch["audio"]
    
    # Ensure audio is properly formatted
    if audio["array"] is None or len(audio["array"]) == 0:
        # If audio loading failed, return empty features
        batch["input_features"] = feature_extractor(
            [0], sampling_rate=16000
        ).input_features[0]
        batch["labels"] = tokenizer("").input_ids
        return batch
    
    # (2) compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # (3) encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Use Data Collator to perform Speech Seq2Seq with padding
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    """
    Define evaluation metrics. We will use the Word Error Rate (WER) metric.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def convert_audio_format(audio_path, target_sr=16000):
    """
    Convert audio to proper format using ffmpeg if needed.
    Returns path to converted audio file.
    """
    if not os.path.exists(audio_path):
        return None
    
    # Check if audio needs conversion
    try:
        # Try to load with librosa first
        import librosa
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio_path  # File is already in good format
    except:
        # Convert with ffmpeg
        try:
            converted_path = audio_path.replace(".wav", f"_converted_{target_sr}.wav")
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-ar", str(target_sr),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                converted_path,
                "-y"
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return converted_path
        except Exception as e:
            print(f"Error converting {audio_path}: {e}")
            return None


def load_local_dataset(jsonl_path, split_ratio=0.8):
    """
    Load local dataset from JSONL file and split into train/test.
    Expected JSONL format: {"audio": "path/to/audio.wav", "text": "transcription", "language": "lang_code"}
    """
    from datasets import Dataset, DatasetDict
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                    continue
    
    print(f"Loaded {len(data)} examples from {jsonl_path}")
    
    # Group by language for balanced splitting
    from collections import defaultdict
    lang_data = defaultdict(list)
    for item in data:
        lang_data[item.get("language", "unknown")].append(item)
    
    # Create balanced train/test splits per language
    train_data = []
    test_data = []
    
    for lang, items in lang_data.items():
        split_idx = int(len(items) * split_ratio)
        train_data.extend(items[:split_idx])
        test_data.extend(items[split_idx:])
    
    print(f"Train examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })


def load_multilingual_dataset(summary_path):
    """
    Load multilingual dataset based on dataset_summary.json.
    Combines all train.jsonl and val.jsonl files.
    """
    from datasets import Dataset, DatasetDict, concatenate_datasets
    import glob
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    train_datasets = []
    val_datasets = []
    
    for lang_info in summary:
        lang = lang_info["language_dir"]
        train_file = lang_info["output_train"]
        val_file = lang_info["output_val"]
        
        # Load train data
        if os.path.exists(train_file):
            train_data = []
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            train_data.append(json.loads(line.strip()))
                        except json.JSONDecodeError as e:
                            print(f"Error in {train_file}: {e}")
                            continue
            if train_data:
                train_datasets.append(Dataset.from_list(train_data))
                print(f"Loaded {len(train_data)} train examples from {lang}")
        
        # Load validation data
        if os.path.exists(val_file):
            val_data = []
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            val_data.append(json.loads(line.strip()))
                        except json.JSONDecodeError as e:
                            print(f"Error in {val_file}: {e}")
                            continue
            if val_data:
                val_datasets.append(Dataset.from_list(val_data))
                print(f"Loaded {len(val_data)} val examples from {lang}")
    
    # Combine all datasets
    if train_datasets:
        combined_train = concatenate_datasets(train_datasets)
    else:
        combined_train = Dataset.from_list([])
    
    if val_datasets:
        combined_val = concatenate_datasets(val_datasets)
    else:
        combined_val = Dataset.from_list([])
    
    print(f"\nTotal train examples: {len(combined_train)}")
    print(f"Total validation examples: {len(combined_val)}")
    
    return DatasetDict({
        "train": combined_train,
        "test": combined_val
    })


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# STEP 0. Loging to Hugging Face
# get your account token from https://huggingface.co/settings/tokens
token = null;
login_hugging_face(token)
print('We are logged in to Hugging Face now!')


# STEP 1. Load Local Dataset
from datasets import DatasetDict, Audio

# Check which dataset to use
if os.path.exists("dataset_summary.json"):
    print("Loading multilingual dataset from dataset_summary.json...")
    common_voice = load_multilingual_dataset("dataset_summary.json")
elif os.path.exists("dataset.jsonl"):
    print("Loading single dataset from dataset.jsonl...")
    common_voice = load_local_dataset("dataset.jsonl")
else:
    print("No local dataset found. Checking for master.jsonl...")
    if os.path.exists("master.jsonl"):
        print("Loading from master.jsonl...")
        common_voice = load_local_dataset("master.jsonl")
    else:
        raise FileNotFoundError("No dataset file found! Please provide dataset.jsonl or dataset_summary.json")

print("\nDataset structure:")
print(common_voice)

# Remove unnecessary columns if they exist
columns_to_remove = []
for col in common_voice["train"].column_names:
    if col not in ["audio", "text", "language"]:
        columns_to_remove.append(col)

if columns_to_remove:
    common_voice = common_voice.remove_columns(columns_to_remove)

print(f"\nColumns after cleaning: {common_voice['train'].column_names}")

# - - - - - - - - - - - - - - - - - - - - - |
# STEP 2. Prepare: Feature Extractor, Tokenizer and Data
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

# - Load Feature extractor: WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")

# - Load Tokenizer: WhisperTokenizer
# For multilingual model, we don't specify a single language
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", task="transcribe")

# - - - - - - - - - - - - - - - - - - - - - |
# STEP 3. Combine elements with WhisperProcessor
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-large", task="transcribe")

# - - - - - - - - - - - - - - - - - - - - - |
# STEP 4. Prepare Data
print('\n| Check a random audio example:')
print(common_voice["train"][0])

# -> Downsample to 16kHZ and convert audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print('\n| Check after audio processing:')
print(common_voice["train"][0])

# Add language information to tokenizer prompts if multilingual
if "language" in common_voice["train"].column_names:
    # Get unique languages in dataset
    unique_langs = set(common_voice["train"]["language"])
    print(f"\nLanguages in dataset: {unique_langs}")
    
    # Update tokenizer with language tokens if needed
    # Whisper already has language tokens for many languages

# Prepare dataset for training
print("\nPreparing dataset for training...")
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=2  # num_proc > 1 will enable multiprocessing
)

# - - - - - - - - - - - - - - - - - - - - - |
# STEP 5. Training and evaluation
# STEP 5.1. Initialize the Data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# STEP 5.2. Define evaluation metric
import evaluate
metric = evaluate.load("wer")

# STEP 5.3. Load a pre-trained Checkpoint
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

"""
Override generation arguments:
- no tokens are forced as decoder outputs
- no tokens are suppressed during generation
"""
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# For multilingual training, enable language adaptation
model.config.use_cache = False

# STEP 5.4. Define the training configuration
from transformers import Seq2SeqTrainingArguments

# Create model name based on dataset type
if os.path.exists("dataset_summary.json"):
    model_name = "whisper-large-multilingual"
    dataset_info = "multilingual-local-dataset"
else:
    model_name = "whisper-large-custom"
    dataset_info = "local-custom-dataset"

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{model_name}",  # change to a repo name of your choice
    per_device_train_batch_size=8,  # reduced for local dataset
    gradient_accumulation_steps=2,  # increase for memory efficiency
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id="ndondo330/whisper-large-multilingual",  # Your HuggingFace username
    hub_strategy="every_save",
)

# Initialize a trainer.
from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Save processor object before starting training
processor.save_pretrained(training_args.output_dir)

# STEP 5.5. Training
print('\nTraining is starting...')
print(f'Training on {len(common_voice["train"])} examples')
print(f'Validating on {len(common_voice["test"])} examples')

trainer.train()  # <-- !!! Here the training starts !!!
print('Training is finished.')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
Prepare metadata for Hugging Face Hub upload
"""

# Determine dataset description
if os.path.exists("dataset_summary.json"):
    with open("dataset_summary.json", 'r') as f:
        summary = json.load(f)
    languages = [lang["language_code"] for lang in summary]
    language_str = ", ".join(languages)
    dataset_name = f"Multilingual Dataset ({language_str})"
else:
    dataset_name = "Custom Speech Dataset"

kwargs = {
    "dataset_tags": "speech-recognition",
    "dataset": dataset_name,
    "dataset_args": "local-custom-dataset",
    "language": "multilingual",
    "model_name": "Whisper Large Multilingual - ndondo330",
    "finetuned_from": "openai/whisper-large",
    "tasks": "automatic-speech-recognition",
    "tags": ["whisper", "multilingual", "speech-recognition"],
}

# Upload training results to the Hub.
print('\nUploading model to Hugging Face Hub...')
trainer.push_to_hub(**kwargs)
print('Trained model uploaded to the Hugging Face Hub: https://huggingface.co/ndondo330/whisper-large-multilingual')

# Save the model locally as well
trainer.save_model(f"./{model_name}-final")
print(f"Model saved locally to ./{model_name}-final")