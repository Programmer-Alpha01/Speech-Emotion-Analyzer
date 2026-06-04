import os
import glob
import torch
import torch.nn as nn
import librosa
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from torch.cuda.amp import GradScaler
from torch.amp import autocast 
import time

# ========== CONFIG ==========
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR   = SCRIPT_DIR.parent / "data(mfcc)"   

# --- CRITICAL CHANGE 1: FEATURE DIMENSION ---
ORIG_N_MELS = 117  

PCA_N_MELS  = 117
BATCH_SIZE  = 64
EPOCHS      = 50
NUM_CLASSES = 7   

# --- CRITICAL CHANGE 2: MODEL HYPERPARAMETERS (Kept from last run) ---
D_MODEL = 128
NUM_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.2

# --- CRITICAL CHANGE 2: MODEL HYPERPARAMETERS ---

# ========== DATA AUGMENTATION FUNCTIONS (SpecAugment) ==========
# ... (frequency_masking and time_masking functions remain the same)

def frequency_masking(mfcc, F=10, num_masks=1):
    """Applies frequency masking to the MFCC array."""
    num_features = mfcc.shape[1]
    for _ in range(num_masks):
        f = np.random.randint(0, F)
        if num_features - f <= 0:
            continue
        f0 = np.random.randint(0, num_features - f)
        mfcc[:, f0:f0 + f] = 0
    return mfcc

def time_masking(mfcc, T=40, num_masks=1):
    """Applies time masking to the MFCC array."""
    num_frames = mfcc.shape[0]
    for _ in range(num_masks):
        t = np.random.randint(0, T)
        if num_frames - t <= 0:
            continue
        t0 = np.random.randint(0, num_frames - t)
        mfcc[t0:t0 + t, :] = 0
    return mfcc


# ========== POS ENCODING (POSITIONAL ENCODING) ==========

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# ========== DATASET ==========

class MFCCDataset(Dataset):
    def __init__(self, file_list, max_frames=1000, pca=None, augment=False):
        self.file_list = file_list
        self.max_frames = max_frames
        self.pca = pca
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        mfcc = np.load(file_path)         

        # --- CRITICAL FIX: GENERATE DELTAS ON THE FLY ---
        # If data is (Time, 39) but we want 117, we compute deltas here.
        if mfcc.shape[1] == 39 and ORIG_N_MELS == 117:
            # librosa.feature.delta expects (n_features, time), so we Transpose (.T)
            mfcc_T = mfcc.T 
            delta1 = librosa.feature.delta(mfcc_T, order=1)
            delta2 = librosa.feature.delta(mfcc_T, order=2)
            # Concatenate back to (117, Time) and Transpose to (Time, 117)
            mfcc = np.concatenate((mfcc_T, delta1, delta2), axis=0).T
        # -----------------------------------------------

        num_frames = min(mfcc.shape[0], self.max_frames)
        mfcc = mfcc[:num_frames]

        if num_frames > 0:
            # Normalization (per feature/dimension)
            mean = np.mean(mfcc, axis=0) 
            std = np.std(mfcc, axis=0)
            mfcc = (mfcc - mean) / (std + np.finfo(float).eps)
            
            # Apply SpecAugment ONLY during training (augment=True)
            if self.augment:
                # Adjust F and T masks slightly for larger feature dim if needed
                mfcc = frequency_masking(mfcc, F=15, num_masks=1)
                mfcc = time_masking(mfcc, T=40, num_masks=1)
                
        else:
            # Handle empty files
            mfcc = np.zeros((0, ORIG_N_MELS), dtype=np.float32) 
        
        # PCA reduction (Skipped since PCA_N_MELS=117)
        if self.pca is not None and num_frames > 0 and self.pca.n_components_ < ORIG_N_MELS:
            mfcc = self.pca.transform(mfcc)
            target_dim = self.pca.n_components_
        else:
            target_dim = ORIG_N_MELS 
        
        # Padding
        padded_mfcc = np.zeros((self.max_frames, target_dim), dtype=np.float32)
        if num_frames > 0:
            padded_mfcc[:num_frames] = mfcc

        # Key Padding Mask
        padding_mask = torch.ones(self.max_frames, dtype=torch.bool)
        padding_mask[:num_frames] = False

        return (torch.from_numpy(padded_mfcc).float(), padding_mask, torch.tensor(label, dtype=torch.long))
# ========== MODEL (模型) ==========

class EmotionRecognitionTransformer(nn.Module):
    def __init__(self, input_dim, d_model=D_MODEL, num_heads=4, num_layers=NUM_LAYERS, dropout=DROPOUT, dim_feedforward=DIM_FEEDFORWARD, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dropout=dropout, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src, src_key_padding_mask):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # Mean Pooling with Masking
        mask = src_key_padding_mask.unsqueeze(2).expand(-1, -1, self.classifier.in_features)
        output = output.masked_fill(mask, 0)
        summed = output.sum(dim=1)
        num_valid = (~src_key_padding_mask).sum(dim=1, keepdim=True).float()
        pooled = summed / num_valid.clamp(min=1e-9)
        
        logits = self.classifier(pooled)
        return logits

# ========== UTILS (Utility Functions) ==========

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Implements linear learning rate decay with Warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

def evaluate(model, data_loader, criterion, device):
    """Evaluation function"""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for padded_mfcc, padding_mask, labels in data_loader:
            padded_mfcc, padding_mask, labels = padded_mfcc.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                outputs = model(padded_mfcc, padding_mask)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def get_file_list(data_dir):
    """Scans data directory to build file path and label list."""
    global NUM_CLASSES 
    
    file_list = []
    label_map = {}
    label_idx = 0
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
        
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            if subdir not in label_map:
                if label_idx >= NUM_CLASSES:
                    print(f"Warning: More than {NUM_CLASSES} classes detected, skipping {subdir}")
                    continue
                label_map[subdir] = label_idx
                label_idx += 1
            for file_path in glob.glob(os.path.join(subdir_path, "*.npy")):
                file_list.append((file_path, label_map[subdir]))
    
    if not file_list:
        raise ValueError(f"No .npy files found in {data_dir}")
        
    NUM_CLASSES = label_idx 
    print(f"Actual detected number of classes: {NUM_CLASSES}")
    return file_list

def fit_pca(file_list, n_components):
    """Fits PCA based on the training data."""
    # This check ensures PCA is skipped if n_components matches the input dim (117)
    if n_components >= ORIG_N_MELS:
        print("Skipping PCA fitting: n_components is not less than the original dimension.")
        return None
        
    all_mfcc = []
    for file_path, _ in file_list:
        mfcc = np.load(file_path)
        all_mfcc.append(mfcc)
        
    concat_mfcc = np.vstack(all_mfcc)
    print(f"Total number of samples used for PCA fitting: {concat_mfcc.shape[0]}")
    
    pca = PCA(n_components=n_components)
    pca.fit(concat_mfcc)
    print(f"PCA fitted, explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    return pca

# ========== MAIN TRAINING FUNCTION ==========

def train_and_evaluate(data_dir, num_epochs=EPOCHS, batch_size=BATCH_SIZE):
    file_list = get_file_list(data_dir)
    
    # Data splitting
    train_files, test_files = train_test_split(file_list, test_size=0.3, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.3, random_state=42)

    ## Optimization: Calculate and apply Class Weights
    train_labels = [label for _, label in train_files]
    class_counts = Counter(train_labels)
    
    print("\n--- Class Balance Check ---")
    print(f"Training set class distribution: {class_counts}")
    
    # Calculate class weights (Inverse Frequency)
    total_samples = sum(class_counts.values())
    class_weights_list = [0.0] * NUM_CLASSES 
    
    for class_id in range(NUM_CLASSES):
        count = class_counts.get(class_id, 0)
        if count > 0:
            class_weights_list[class_id] = total_samples / count
        else:
            class_weights_list[class_id] = 0.0 

    class_weights = torch.tensor(class_weights_list, dtype=torch.float32)
    
    # Soften weights using square root
    class_weights = torch.sqrt(class_weights)
    print(f"Softened class weights (applied to CrossEntropyLoss): {class_weights.tolist()}")
    print("---------------------\n")
    
    # Fit PCA (Will be skipped)
    pca = fit_pca(train_files, n_components=PCA_N_MELS)

    # Datasets and DataLoaders
    train_dataset = MFCCDataset(train_files, pca=pca, augment=True) 
    val_dataset   = MFCCDataset(val_files,   pca=pca, augment=False)
    test_dataset  = MFCCDataset(test_files,  pca=pca, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, num_workers=4, pin_memory=True)

    # Device and Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_dim is now ORIG_N_MELS (117)
    model = EmotionRecognitionTransformer(input_dim=ORIG_N_MELS, num_classes=NUM_CLASSES).to(device) 
    
    # Initialize CrossEntropyLoss with softened class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() 
    
    # Learning Rate Scheduler
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_val_acc = 0
    patience, counter = 10, 0
    best_model_path = "model2.pth"

    # Training Loop
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for padded_mfcc, padding_mask, labels in train_loader:
            padded_mfcc, padding_mask, labels = padded_mfcc.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # Mixed Precision Training
            with autocast(device_type='cuda'):
                outputs = model(padded_mfcc, padding_mask)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        # Evaluation
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_accuracy:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} seconds")
        
        # Save best model and Early Stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print("Early stopping triggered")
            break

    # Final Test
    model.load_state_dict(torch.load(best_model_path))
    final_test_loss, final_test_accuracy = evaluate(model, test_loader, criterion, device)
    print("\n==================================")
    print(f"Final Test Accuracy: {final_test_accuracy:.4f}, Final Test Loss: {final_test_loss:.4f}")
    print("==================================")

    # Confusion Matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for padded_mfcc, padding_mask, labels in test_loader:
            padded_mfcc, padding_mask, labels = padded_mfcc.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(device_type='cuda'):
                outputs = model(padded_mfcc, padding_mask)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    

if __name__ == "__main__":
    print(f"Looking for data in: {DATA_DIR.resolve()}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found! Expected: {DATA_DIR}")
    train_and_evaluate(DATA_DIR)