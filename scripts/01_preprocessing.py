# 01_preprocess.py

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer

# --- 1. Configuration & Globals ---
config = {
    "MODEL_NAME": "yangheng/deberta-v3-base-absa-v1.1", 
    "TRAIN_FILE": "/kaggle/input/2025-sep-dl-gen-ai-project/train.csv",
    "TEST_FILE": "/kaggle/input/2025-sep-dl-gen-ai-project/test.csv",
    "VALIDATION_SPLIT_SIZE": 0.1, 
    "TRAIN_BATCH_SIZE": 16, 
    "EVAL_BATCH_SIZE": 32, 
    "RANDOM_SEED": 42, 
    "MAX_TOKEN_LENGTH": 128,
    "OUTPUT_DIR": "./artifacts" # New directory for saving preprocessed data/info
}

Path(config["OUTPUT_DIR"]).mkdir(exist_ok=True)

emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
num_labels = len(emotion_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Data Loading & Splitting ---
all_train_df = pd.read_csv(config["TRAIN_FILE"])
test_df = pd.read_csv(config["TEST_FILE"])

# Use the 'id' column for reproducible splitting
train_indices, val_indices = train_test_split(
    all_train_df.index.values,
    test_size=config["VALIDATION_SPLIT_SIZE"],
    random_state=config["RANDOM_SEED"]
)

# Create the splits using .loc
df_train = all_train_df.loc[train_indices].reset_index(drop=True)
df_val = all_train_df.loc[val_indices].reset_index(drop=True)
    
print(f"Training split shape: {df_train.shape}")
print(f"Validation split shape: {df_val.shape}")

# Save the splits to disk (for simple data reuse/debugging)
df_train.to_csv(Path(config["OUTPUT_DIR"]) / "df_train.csv", index=False)
df_val.to_csv(Path(config["OUTPUT_DIR"]) / "df_val.csv", index=False)
test_df.to_csv(Path(config["OUTPUT_DIR"]) / "df_test.csv", index=False)
print(f"Data splits saved to {config['OUTPUT_DIR']}")


# --- 3. Weight Calculation & Tokenizer Loading ---

# Calculate pos_weight vector
pos_weights_list = []
total_train_samples = len(df_train)

for label in emotion_labels:
    pos_count = df_train[label].sum()
    neg_count = total_train_samples - pos_count
    weight = neg_count / pos_count if pos_count > 0 else 1.0
    pos_weights_list.append(weight)
    
print(f"pos_weight vector: {pos_weights_list}")

# Save pos_weights for training script
np.save(Path(config["OUTPUT_DIR"]) / "pos_weights.npy", np.array(pos_weights_list))

# Load tokenizer and save it to the output directory
tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])
tokenizer.save_pretrained(config["OUTPUT_DIR"])
print(f"Tokenizer saved to {config['OUTPUT_DIR']}")

# --- 4. Dataset & DataLoader Functions ---

def preprocess_function(batch_texts, batch_labels):
    """Tokenizes text and returns a dict for the model."""
    # Load tokenizer dynamically (or use global if loaded)
    tokenizer = AutoTokenizer.from_pretrained(config["OUTPUT_DIR"])
    tokenized_inputs = tokenizer(
        batch_texts,
        truncation=True,
        padding="max_length",
        max_length=config["MAX_TOKEN_LENGTH"],
        return_tensors="pt"
    )
    # Ensure labels are present for the training step
    if batch_labels:
        labels_tensor = torch.tensor(batch_labels, dtype=torch.float)
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels_tensor
        }
    else:
         return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"]
        }


class EmotionDataset(TorchDataset):
    """Custom PyTorch dataset."""
    def __init__(self, df, emotion_labels, is_test=False):
        self.texts = df['text'].tolist()
        self.is_test = is_test
        if not self.is_test:
            self.labels = df[emotion_labels].values
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if self.is_test:
            return self.texts[idx], [] # Return empty labels for test
        return self.texts[idx], self.labels[idx]

def collate_fn_train_val(batch):
    """Custom collate function to batch-tokenize for train/val (with labels)."""
    texts, labels = zip(*batch)
    return preprocess_function(list(texts), list(labels))

def collate_fn_test(batch):
    """Collate function for the test set (no labels)."""
    texts, _ = zip(*batch)
    return preprocess_function(list(texts), [])

# Create DataLoaders
train_dataset = EmotionDataset(df_train, emotion_labels)
train_loader = DataLoader(
    train_dataset,
    batch_size=config["TRAIN_BATCH_SIZE"],
    collate_fn=collate_fn_train_val,
    shuffle=True,
    num_workers=0 
)

val_dataset = EmotionDataset(df_val, emotion_labels)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["EVAL_BATCH_SIZE"],
    collate_fn=collate_fn_train_val,
    shuffle=False,
    num_workers=0
)

test_dataset_obj = EmotionDataset(test_df, emotion_labels, is_test=True)
test_loader = DataLoader(
    test_dataset_obj,
    batch_size=config["EVAL_BATCH_SIZE"],
    collate_fn=collate_fn_test,
    shuffle=False,
    num_workers=0
)

print(f"Steps per epoch (train): {len(train_loader)}")
print("Preprocessing complete. Artifacts saved.")
# Note: The DataLoader objects are not directly saved but the underlying data/tokenizer are.