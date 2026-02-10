# 03_inference.py

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Configuration & Globals ---
config = {
    "MODEL_PATH_FOR_INFERENCE": "./final_model", # Use the local path where model was saved
    "EVAL_BATCH_SIZE": 32, 
    "MAX_TOKEN_LENGTH": 128,
    "OUTPUT_DIR": "./artifacts",
}

# NOTE: Since this is an inference script, we use the MANUAL_THRESHOLDS as a fallback.
# In a real submission scenario, you'd likely hardcode the best-found thresholds
# if they weren't saved/loaded correctly.
MANUAL_THRESHOLDS = [0.79, 0.48000000000000004, 0.62, 0.75, 0.92] 

emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
num_labels = len(emotion_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Load Preprocessed Data and Thresholds ---

test_df = pd.read_csv(Path(config["OUTPUT_DIR"]) / "df_test.csv")

try:
    optimal_thresholds_arr = np.load(Path(config["OUTPUT_DIR"]) / "optimal_thresholds.npy")
    final_thresholds = optimal_thresholds_arr.tolist()
    print(f"Using calculated optimal thresholds: {final_thresholds}")
except FileNotFoundError:
    print("Warning: 'optimal_thresholds.npy' not found. Using MANUAL_THRESHOLDS.")
    final_thresholds = MANUAL_THRESHOLDS

# --- 3. Load Model and Tokenizer ---
print(f"Loading model from {config['MODEL_PATH_FOR_INFERENCE']}...")

try:
    model = AutoModelForSequenceClassification.from_pretrained(config["MODEL_PATH_FOR_INFERENCE"])
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_PATH_FOR_INFERENCE"])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}. Exiting.")
    exit() # Exit if model load fails

# --- 4. Dataset & DataLoader Functions ---

def preprocess_function_test(batch_texts, tokenizer):
    """Tokenizes text for inference (no labels required)."""
    tokenized_inputs = tokenizer(
        batch_texts,
        truncation=True,
        padding="max_length",
        max_length=config["MAX_TOKEN_LENGTH"],
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"]
    }

class EmotionDatasetTest(TorchDataset):
    def __init__(self, df):
        self.texts = df['text'].tolist()
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], [] # dummy label

def collate_fn_test(batch):
    texts, _ = zip(*batch)
    return preprocess_function_test(list(texts), tokenizer)

# Create Test DataLoader
test_dataset_obj = EmotionDatasetTest(test_df)
test_loader = DataLoader(
    test_dataset_obj,
    batch_size=config["EVAL_BATCH_SIZE"],
    collate_fn=collate_fn_test,
    shuffle=False,
    num_workers=0
)

# --- 5. Get Test Predictions ---
print("Predicting on test set...")
test_probs = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        sigmoid = torch.sigmoid(outputs.logits)
        test_probs.append(sigmoid.cpu())
        
test_probs = torch.cat(test_probs, dim=0).numpy()
final_preds = np.zeros(test_probs.shape, dtype=int)

# --- 6. Apply Thresholds ---
print("Applying thresholds...")
for i in range(num_labels):
    thresh = final_thresholds[i]
    final_preds[:, i] = (test_probs[:, i] > thresh).astype(int)
    print(f"  {emotion_labels[i]}: {thresh}")

# --- 7. Save Submission ---
submission_df = pd.DataFrame(final_preds, columns=emotion_labels)
submission_df.insert(0, 'id', test_df['id'])

submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file created: submission.csv")

submission_df.head()