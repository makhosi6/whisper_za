#!/usr/bin/env python3
"""
Simplified MLX Whisper Fine-tuning
Uses mlx-lm library for easier model loading
"""

import os
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.utils import load
import numpy as np

# Configuration
CONFIG = {
    "model": "openai/whisper-small",
    "dataset": "dataset.jsonl",
    "batch_size": 2,
    "learning_rate": 1e-5,
    "epochs": 10,
    "output_dir": "whisper-mlx",
}

def load_dataset():
    """Load dataset from JSONL"""
    data = []
    with open(CONFIG["dataset"], 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def prepare_batch(batch):
    """Prepare batch for training"""
    # This is simplified - actual implementation would need audio processing
    return {
        "input_ids": mx.array(np.random.randn(CONFIG["batch_size"], 80, 3000)),
        "labels": mx.array(np.random.randint(0, 1000, (CONFIG["batch_size"], 100)))
    }

def main():
    print("=== MLX Whisper Fine-tuning ===")
    print("Configuration:", CONFIG)
    
    # Set device
    mx.set_default_device(mx.gpu)  # Use GPU if available
    
    # Load model using mlx-lm
    print("Loading model...")
    model, tokenizer = load(CONFIG["model"])
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} examples")
    
    # Training setup
    optimizer = optim.Adam(learning_rate=CONFIG["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop (simplified)
    print("Starting training...")
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        
        # Simplified batch training
        for i in range(0, len(dataset), CONFIG["batch_size"]):
            batch = prepare_batch(dataset[i:i+CONFIG["batch_size"]])
            
            def compute_loss():
                logits = model(batch["input_ids"])
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                              batch["labels"].reshape(-1))
                return loss
            
            # Compute loss and gradients
            loss, grads = nn.value_and_grad(model, compute_loss)()
            
            # Update model
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} completed, Avg Loss: {total_loss/len(dataset):.4f}")
    
    # Save model
    print("Saving model...")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    model.save_weights(os.path.join(CONFIG["output_dir"], "model.npz"))
    
    print(f"Model saved to {CONFIG['output_dir']}")
    print("Done!")

if __name__ == "__main__":
    main()