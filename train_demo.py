import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import glob

# Import classes from app.main
from app.main import AimLSTM, AimDataset

# Read only first 10 CSV files to speed up testing
csv_files = sorted(glob.glob("ru_data/*.csv"))[:10]
all_sequences = []
all_labels = []

# Create sequences: window 40, step 20
window_size = 40
step = 20

print(f"Loading {len(csv_files)} CSV files...", flush=True)
for file_path in csv_files:
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', nrows=10000)  # Limit to 10k rows
        filename = os.path.basename(file_path).upper()
        
        # Determine label from filename
        if "LEGIT" in filename:
            label = 0
        elif "CHEAT" in filename:
            label = 1
        else:
            continue
        
        # Get data
        data = df.iloc[:, 1:].values if df.shape[1] > 8 else df.iloc[:, :].values
        
        # Ensure we have exactly 8 features
        if data.shape[1] < 8:
            data = np.pad(data, ((0, 0), (0, 8 - data.shape[1])), mode='constant', constant_values=0)
        elif data.shape[1] > 8:
            data = data[:, :8]
        
        # Skip if not enough data for sequences
        if len(data) < window_size:
            continue
        
        # Create sequences
        for i in range(0, len(data) - window_size + 1, step):
            seq = data[i:i+window_size]
            all_sequences.append(seq)
            all_labels.append(label)
        
        print(f"✓ {os.path.basename(file_path)}: {len(data)} rows", flush=True)
    except Exception as e:
        print(f"⚠ Error loading {file_path}: {e}", flush=True)

print(f"\n✓ Loaded {len(all_sequences)} sequences", flush=True)

# Fit scaler
print(f"Fitting scaler...", flush=True)
all_data_flat = np.array(all_sequences).reshape(-1, 8)
scaler = StandardScaler()
scaler.fit(all_data_flat)

# Normalize sequences
print(f"Normalizing sequences...", flush=True)
sequences_normalized = np.array([scaler.transform(seq) for seq in all_sequences])

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(sequences_normalized, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

# Create datasets and dataloaders
train_dataset = AimDataset(X_train, y_train)
val_dataset = AimDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, optimizer, loss
model = AimLSTM()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training
num_epochs = 5
print(f"Starting training for {num_epochs} epochs...", flush=True)
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for seqs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_loss = epoch_train_loss / len(train_loader)
    
    # Validation
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for seqs, labels in val_loader:
            outputs = model(seqs)
            loss = criterion(outputs.view(-1), labels.view(-1))
            epoch_val_loss += loss.item()
    val_loss = epoch_val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)

# Save model and scaler
print(f"Saving model and scaler...", flush=True)
torch.save(model.state_dict(), "model.pth")
joblib.dump(scaler, "scaler.pkl")

print("✓ Model and scaler saved.", flush=True)
