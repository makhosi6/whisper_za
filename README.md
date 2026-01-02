

# Whisper Multilingual Fine-tuning Project

## 📋 Overview

This project provides a comprehensive toolkit for fine-tuning OpenAI's Whisper speech recognition model on custom multilingual datasets. It includes implementations for both PyTorch/Hugging Face Transformers and Apple Silicon-optimized MLX frameworks.

## 🚀 Features

- **Multilingual Support**: Fine-tune on multiple languages simultaneously
- **Dataset Management**: Tools for organizing, cleaning, and preparing speech datasets
- **Two Framework Options**:
  - **PyTorch/Transformers**: Full-featured training with Hugging Face integration
  - **MLX**: Apple Silicon-optimized training (M1/M2/M3 Macs)
- **Audio Processing**: Automatic format conversion using FFmpeg
- **Hugging Face Hub Integration**: Easy model sharing and deployment
- **Duplicate Detection**: Remove duplicate audio-text pairs
- **Comprehensive Logging**: Track training progress and results

## 📁 Project Structure

```
whisper-finetuning/
├── data/
│   ├── dataset.jsonl                    # Combined dataset (master file)
│   ├── dataset_summary.json             # Dataset metadata and organization
│   ├── af-ZA/                           # Language-specific directories
│   │   ├── transcriptions/
│   │   │   ├── train.jsonl
│   │   │   └── val.jsonl
│   │   └── audio/
│   └── en-ZA/                           # Another language directory
├── scripts/
│   ├── combine_jsonl_files.sh           # Combine multiple JSONL files
│   ├── deduplicate.sh                   # Remove duplicate audio-text pairs
│   ├── split_by_language.sh             # Split dataset by language
│   └── prepare_audio.sh                 # Audio format conversion
├── training/
│   ├── finetune_whisper_multilingual.py # PyTorch training script
│   └── whisper_mlx_finetune.py          # MLX training script
├── outputs/
│   ├── whisper-large-multilingual/      # Trained models
│   └── logs/                            # Training logs
├── requirements.txt
└── README.md
```

## 🛠️ Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with >= 8GB VRAM (for PyTorch) OR
- **Apple Silicon**: M1/M2/M3 Mac with >= 16GB RAM (for MLX)
- **Storage**: 50GB+ free space for datasets and models

### Software Requirements
- Python 3.8+
- CUDA 11.7+ (for PyTorch GPU training)
- FFmpeg for audio processing
- Hugging Face account

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/whisper-finetuning.git
cd whisper-finetuning
```

### 2. Install Dependencies

### pyenv setup instructions

```bash
    pyenv global 3.11.13
    pyenv exec python -m venv env
    pip3 install -r requirements.txt
```

#### For PyTorch/Transformers:
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0
evaluate>=0.4.0
librosa>=0.10.0
soundfile>=0.12.0
huggingface-hub>=0.16.0
ffmpeg-python>=0.2.0
tensorboard>=2.13.0
jq  # For bash scripts (install via apt)
```

#### For MLX (Apple Silicon):
```bash
pip install mlx mlx-lm
pip install transformers datasets
pip install librosa soundfile
pip install huggingface-hub
```

### 3. Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg jq

# macOS
brew install ffmpeg jq
```

## 📊 Dataset Preparation

### Dataset Format

Your dataset should be in JSONL format with the following structure:

**dataset.jsonl:**
```json
{"audio": "en-ZA/audio/file1.wav", "text": "Hello world", "language": "en"}
{"audio": "af-ZA/audio/file2.wav", "text": "Hallo wêreld", "language": "af"}
{"audio": "zu-ZA/audio/file3.wav", "text": "Sawubona mhlaba", "language": "zu"}
```

### Step 1: Organize by Language
```bash
# Split dataset by language
./scripts/split_by_language.sh dataset.jsonl

# This creates:
# - Language directories (en-ZA/, af-ZA/, etc.)
# - Train/validation splits (70/30)
# - dataset_summary.json with metadata
```

### Step 2: Check for Duplicates
```bash
# Remove duplicate audio-text pairs
./scripts/deduplicate.sh dataset.jsonl summary.json

# Creates:
# - dataset_deduped.jsonl (cleaned dataset)
# - deduplicate.log (removed duplicates log)
# - Updated summary.json
```

### Step 3: Combine Files
```bash
# Combine all train/val files into master dataset
./scripts/combine_jsonl_files.sh dataset_summary.json

# Creates:
# - master.jsonl (combined dataset)
# - combine_jsonl.log (processing log)
```

### Step 4: Prepare Audio Files
```bash
# Convert audio formats to 16kHz WAV
./scripts/prepare_audio.sh

# This:
# - Converts all audio to 16kHz, mono
# - Creates standardized format for training
```

## 🚀 Training

### Option A: PyTorch Training (NVIDIA GPU)

1. **Set up Hugging Face token:**
```python
# In finetune_whisper_multilingual.py
token = 'YOUR_HF_TOKEN_HERE'  # Get from https://huggingface.co/settings/tokens
```

2. **Run training:**
```bash
python training/finetune_whisper_multilingual.py \
  --model_name "openai/whisper-large" \
  --dataset "master.jsonl" \
  --output_dir "whisper-large-multilingual" \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --num_epochs 10
```

**Training Parameters:**
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-multilingual",
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    fp16=True,  # Mixed precision for faster training
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    push_to_hub=True,
    hub_model_id="ndondo330/whisper-large-multilingual",
)
```

### Option B: MLX Training (Apple Silicon)

```bash
python training/whisper_mlx_finetune.py \
  --model "openai/whisper-small" \
  --dataset "master.jsonl" \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --max_steps 2000 \
  --output_dir "whisper-mlx-finetuned"
```

## 🔧 Configuration

### Dataset Summary Format (dataset_summary.json)
```json
[
  {
    "language_dir": "af-ZA",
    "language_code": "af",
    "total_examples": 2927,
    "train_examples": 2634,
    "val_examples": 293,
    "output_train": "af-ZA/transcriptions/train.jsonl",
    "output_val": "af-ZA/transcriptions/val.jsonl"
  }
]
```

### Model Configuration

Available Whisper models:
- `openai/whisper-tiny` (39M parameters)
- `openai/whisper-base` (74M parameters)
- `openai/whisper-small` (244M parameters)
- `openai/whisper-medium` (769M parameters)
- `openai/whisper-large` (1550M parameters)
- `openai/whisper-large-v2` (1550M parameters)

## 📈 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./whisper-large-multilingual/runs
```

### Check Logs
```bash
# Training logs
tail -f whisper-large-multilingual/trainer_log.jsonl

# Evaluation metrics
cat whisper-large-multilingual/eval_results.json
```

## 🤖 Inference

### Using the Fine-tuned Model

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Load fine-tuned model
processor = WhisperProcessor.from_pretrained("ndondo330/whisper-large-multilingual")
model = WhisperForConditionalGeneration.from_pretrained("ndondo330/whisper-large-multilingual")

# Load and process audio
audio, sr = librosa.load("test_audio.wav", sr=16000)
input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features

# Generate transcription
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"Transcription: {transcription}")
```

### Batch Inference
```bash
python inference.py \
  --model_path "whisper-large-multilingual" \
  --audio_dir "test_audio/" \
  --output_file "transcriptions.jsonl"
```

## 📊 Evaluation

### Word Error Rate (WER) Calculation
```python
import evaluate

wer_metric = evaluate.load("wer")
predictions = ["this is a test", "another example"]
references = ["this is test", "another example"]

wer = wer_metric.compute(predictions=predictions, references=references)
print(f"WER: {wer * 100:.2f}%")
```

### Evaluate on Test Set
```bash
python evaluate_model.py \
  --model "ndondo330/whisper-large-multilingual" \
  --test_data "test.jsonl" \
  --output "evaluation_results.json"
```

## 🚢 Deployment

### Push to Hugging Face Hub
```python
from huggingface_hub import push_to_hub

kwargs = {
    "dataset_tags": "multilingual-speech-dataset",
    "dataset": "Custom Multilingual Dataset",
    "model_name": "Whisper Large Multilingual - ndondo330",
    "finetuned_from": "openai/whisper-large",
    "tasks": "automatic-speech-recognition",
}

trainer.push_to_hub(**kwargs)
```

### Create Model Card
```yaml
---
language:
  - en
  - af
  - zu
  - xh
tags:
  - automatic-speech-recognition
  - whisper
  - multilingual
license: apache-2.0
datasets:
  - custom-multilingual-speech
metrics:
  - wer
---

# Whisper Large Multilingual

Fine-tuned Whisper model on South African languages.
```

## 📚 Script Reference

### `combine_jsonl_files.sh`
```bash
# Combine all train/val files from dataset_summary.json
./scripts/combine_jsonl_files.sh [dataset_summary.json]

# Options:
# --summary FILE    # Use custom summary file
# --output FILE     # Output file name (default: master.jsonl)
# --deduplicate     # Remove duplicates after combining
```

### `deduplicate.sh`
```bash
# Remove duplicates by audio+text
./scripts/deduplicate.sh dataset.jsonl [summary.json]

# Creates:
# - dataset_deduped.jsonl
# - Updated summary.json with deduplication stats
# - deduplicate.log
```

### `split_by_language.sh`
```bash
# Split dataset by language
./scripts/split_by_language.sh dataset.jsonl

# Creates language directories with train/val splits
# Updates dataset_summary.json
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch_size 4
   
   # Enable gradient checkpointing
   --gradient_checkpointing True
   
   # Use gradient accumulation
   --gradient_accumulation_steps 4
   ```

2. **Audio Loading Errors**
   ```bash
   # Check FFmpeg installation
   ffmpeg -version
   
   # Convert audio manually
   ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
   ```

3. **Hugging Face Authentication**
   ```bash
   # Login via CLI
   huggingface-cli login
   
   # Or set token
   export HUGGING_FACE_HUB_TOKEN="your_token_here"
   ```

4. **MLX Installation Issues**
   ```bash
   # Ensure Python 3.8+ on macOS
   python --version
   
   # Install from source if needed
   pip install git+https://github.com/ml-explore/mlx.git
   ```

### Performance Tips

1. **For faster training:**
   - Use mixed precision (`fp16=True`)
   - Enable gradient checkpointing
   - Use larger batch sizes with gradient accumulation
   - Cache processed dataset

2. **For better accuracy:**
   - Use larger Whisper model (large-v2)
   - Train for more epochs
   - Use diverse multilingual data
   - Clean dataset thoroughly

## 📈 Expected Results

| Model | Languages | WER | Training Time | GPU Memory |
|-------|-----------|-----|---------------|------------|
| whisper-small | 4 languages | ~15-20% | 2-4 hours | 8GB VRAM |
| whisper-medium | 4 languages | ~10-15% | 6-8 hours | 16GB VRAM |
| whisper-large | 4 languages | ~8-12% | 10-15 hours | 24GB VRAM |
| whisper-mlx | 4 languages | ~12-18% | 4-6 hours | 16GB RAM |



## 🚀 Quick Start Summary

```bash
# 1. Prepare dataset
./scripts/split_by_language.sh your_data.jsonl
./scripts/deduplicate.sh dataset.jsonl
./scripts/combine_jsonl_files.sh

# 2. Train with PyTorch
python training/finetune_whisper_multilingual.py

# OR Train with MLX
python training/whisper_mlx_finetune.py

# 3. Evaluate
python evaluate_model.py --model outputs/whisper-large-multilingual

# 4. Deploy
huggingface-cli upload ndondo330/whisper-multilingual outputs/whisper-large-multilingual
```

