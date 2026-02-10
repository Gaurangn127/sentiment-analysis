# 02_train.py

import os
import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from kaggle_secrets import UserSecretsClient # For secure API key access

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

# Set multiprocessing start method 
try:
    torch.multiprocessing.set_start_method("spawn", force=True)
    print("\nSet torch multiprocessing start method to 'spawn'.")
except RuntimeError as e:
    print(f"Note: Could not set start method: {e}")

# --- 1. Configuration & Globals ---
config = {
    "MODEL_NAME": "yangheng/deberta-v3-base-absa-v1.1", 
    "EPOCHS": 25, 
    "STARTING_LR": 5e-5,
    "TRAIN_BATCH_SIZE": 16, 
    "EVAL_BATCH_SIZE": 32, 
    "MAX_TOKEN_LENGTH": 128,
    "LR_BASE_FACTOR": 0.95,
    "OUTPUT_DIR": "./artifacts",
    "MODEL_SAVE_PATH": "./final_model",
}

# Ensure final model path exists
Path(config["MODEL_SAVE_PATH"]).mkdir(exist_ok=True)

emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
num_labels = len(emotion_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Load Preprocessed Data and Weights ---

# Load data splits
df_train = pd.read_csv(Path(config["OUTPUT_DIR"]) / "df_train.csv")
df_val = pd.read_csv(Path(config["OUTPUT_DIR"]) / "df_val.csv")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["OUTPUT_DIR"])

# Load positive weights
pos_weights_list = np.load(Path(config["OUTPUT_DIR"]) / "pos_weights.npy").tolist()
pos_weight_tensor = torch.tensor(pos_weights_list, dtype=torch.float).to(device)
print(f"Loaded pos_weight vector: {pos_weights_list}")

# Re-define necessary functions/classes from preprocessing script
def preprocess_function(batch_texts, batch_labels, tokenizer):
    """Tokenizes text and returns a dict for the model."""
    tokenized_inputs = tokenizer(
        batch_texts,
        truncation=True,
        padding="max_length",
        max_length=config["MAX_TOKEN_LENGTH"],
        return_tensors="pt"
    )
    labels_tensor = torch.tensor(batch_labels, dtype=torch.float)
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels_tensor
    }

class EmotionDataset(TorchDataset):
    def __init__(self, df, emotion_labels):
        self.texts = df['text'].tolist()
        self.labels = df[emotion_labels].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn_train_val(batch):
    texts, labels = zip(*batch)
    return preprocess_function(list(texts), list(labels), tokenizer)

# Create DataLoaders
train_dataset = EmotionDataset(df_train, emotion_labels)
train_loader = DataLoader(
    train_dataset, batch_size=config["TRAIN_BATCH_SIZE"], collate_fn=collate_fn_train_val, shuffle=True, num_workers=0
)
val_dataset = EmotionDataset(df_val, emotion_labels)
val_loader = DataLoader(
    val_dataset, batch_size=config["EVAL_BATCH_SIZE"], collate_fn=collate_fn_train_val, shuffle=False, num_workers=0
)

# --- 3. WandB Init & Model Setup ---
os.environ["WANDB_SILENT"] = "true"
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("wandb_api")
    wandb.login(key=api_key)
    print("W&B login successful.")
    run = wandb.init(
        project="multi-label-emotion-bert",
        job_type="train",
        config=config,
        name=f"{config['MODEL_NAME']}-train-{wandb.util.generate_id()}"
    )
    wandb.config.update({"pos_weights": pos_weights_list})
except Exception as e:
    print(f"W&B init failed: {e}")
    run = None

print("\n--- Starting Model Training ---")

# Model Setup
model = AutoModelForSequenceClassification.from_pretrained(
    config["MODEL_NAME"],
    num_labels=num_labels,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True 
).to(device)

# Optimizer Setup (Discriminative LR)
base_lr = config["STARTING_LR"] * config["LR_BASE_FACTOR"]
head_lr = config["STARTING_LR"]                            
optimizer_grouped_parameters = [
    {"params": model.deberta.parameters(), "lr": base_lr, "weight_decay": 0.01},
    {"params": model.classifier.parameters(), "lr": head_lr, "weight_decay": 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=head_lr)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

best_model_state = None
global_step = 0

# --- 4. Training Loop ---
for epoch in range(config["EPOCHS"]):
    print(f"\n--- Starting Epoch {epoch+1}/{config['EPOCHS']} ---")
    model.train() 
    total_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fct(logits, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        
        if run is not None:
            wandb.log({
                "train/step_loss": loss.item(),
                "train/lr_head": optimizer.param_groups[1]['lr'],
                "train/lr_base": optimizer.param_groups[0]['lr'],
                "global_step": global_step
            })
        global_step += 1
            
    avg_train_loss = total_loss / len(train_loader)
    print(f"  Average Training Loss: {avg_train_loss:.4f}")
        
    scheduler.step()
    
    # Validation step
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            sigmoid = torch.sigmoid(outputs.logits)
            predictions = (sigmoid > 0.5).int()
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"  Epoch {epoch+1} - Validation Macro F1 (0.5 thresh): {macro_f1:.4f}")
    
    # Save the best model state based on validation F1 (or simply the last one)
    # Since we don't have early stopping logic, we'll save the last state for now
    best_model_state = model.state_dict().copy()

print("\n--- Training Finished ---")

# --- 5. Optimal Threshold Calculation ---
print("\n--- Calculating Optimal Thresholds on Validation Set ---")
model.eval()
val_preds_list = []
val_labels_list = []

# Get raw probabilities for validation set
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Validating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        sigmoid = torch.sigmoid(outputs.logits)
        
        val_preds_list.append(sigmoid.cpu().numpy())
        val_labels_list.append(labels.cpu().numpy())
            
val_preds_arr = np.vstack(val_preds_list)
val_labels_arr = np.vstack(val_labels_list)

optimal_thresholds = []
print("\nOptimization Results:")
for i, label in enumerate(emotion_labels):
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.01, 1.0, 0.01):
        pred_binary = (val_preds_arr[:, i] > thresh).astype(int)
        score = f1_score(val_labels_arr[:, i], pred_binary, zero_division=0)
        
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    optimal_thresholds.append(best_thresh)
    print(f"  {label.ljust(10)}: Best Threshold={best_thresh:.3f}, F1-Score={best_f1:.4f}")
    
print(f"\n>>> FINAL OPTIMAL THRESHOLDS: {optimal_thresholds}")

# --- 6. Save Model and Artifacts ---

# Load the best model state and save
if best_model_state is not None:
    model.load_state_dict(best_model_state)

print(f"Saving model locally to {config['MODEL_SAVE_PATH']}...")
model.save_pretrained(config["MODEL_SAVE_PATH"])
tokenizer.save_pretrained(config["MODEL_SAVE_PATH"])

# Save optimal thresholds for inference
np.save(Path(config["OUTPUT_DIR"]) / "optimal_thresholds.npy", np.array(optimal_thresholds))

# 7. WandB Finish
if run is not None:
    wandb.finish()
    print("WandB run finished.")