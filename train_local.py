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

# Read all CSV from ru_data
csv_files = glob.glob("ru_data/*.csv")
all_sequences = []
all_labels = []
skipped_files = []

# Create sequences: window 40, step 20
window_size = 40
step = 20

for file_path in csv_files:
    try:
        print(f"Loading: {os.path.basename(file_path)}", flush=True)
        df = pd.read_csv(file_path, on_bad_lines='skip', nrows=100000)  # Limit rows to prevent OOM
        filename = os.path.basename(file_path).upper()
        
        # Determine label from filename
        if "LEGIT" in filename:
            label = 0
        elif "CHEAT" in filename:
            label = 1
        else:
            # Try to infer from label column if it exists
            if 'label' in df.columns:
                label = None  # Will be determined per-row
                data = df.iloc[:, :-1].values  # All columns except last (label)
            else:
                print(f"⊘ Skipped {os.path.basename(file_path)}: cannot determine label", flush=True)
                continue
        
        # If label column exists, use it; otherwise use the determined label
        if label is None and 'label' in df.columns:
            # Data is all columns except label
            data = df.iloc[:, :-1].values
            labels_per_row = df['label'].values
        else:
            # Assume data starts from second column (skip first column if it's is_cheating)
            # Check if first column looks like it's is_cheating
            if df.iloc[:, 0].nunique() <= 2:  # Likely a label column
                data = df.iloc[:, 1:].values
            else:
                data = df.iloc[:, :].values
            labels_per_row = None
        
        # Ensure we have exactly 8 features
        if data.shape[1] == 7:
            # Pad with zeros to get 8 features
            data = np.pad(data, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        elif data.shape[1] != 8:
            print(f"⊘ Skipped {os.path.basename(file_path)}: wrong number of features ({data.shape[1]} != 8)", flush=True)
            continue
        
        # Skip if not enough data for sequences
        if len(data) < window_size:
            print(f"⊘ Skipped {os.path.basename(file_path)}: not enough data ({len(data)} < {window_size})", flush=True)
            continue
        
        # Create sequences
        for i in range(0, len(data) - window_size + 1, step):
            seq = data[i:i+window_size]
            all_sequences.append(seq)
            if labels_per_row is not None:
                # Use majority vote from labels in this sequence
                seq_label = int(np.mean(labels_per_row[i:i+window_size]) > 0.5)
                all_labels.append(seq_label)
            else:
                all_labels.append(label)
        
        print(f"✓ Loaded: {os.path.basename(file_path)} ({len(data)} rows, {data.shape[1]} features)", flush=True)
    except Exception as e:
        print(f"⚠ Error loading {os.path.basename(file_path)}: {e}", flush=True)
        skipped_files.append(file_path)

print(f"\n✓ File loading completed", flush=True)

print(f"\nTotal sequences created: {len(all_sequences)}", flush=True)
print(f"Skipped files: {len(skipped_files)}", flush=True)

if not all_sequences:
    print("No sequences created")
    exit(1)

# Fit scaler
print(f"Fitting scaler on {len(all_sequences)} sequences...", flush=True)
all_data_flat = np.array(all_sequences).reshape(-1, 8)
scaler = StandardScaler()
scaler.fit(all_data_flat)
print(f"✓ Scaler fitted", flush=True)

# Normalize sequences
print(f"Normalizing sequences...", flush=True)
sequences_normalized = [scaler.transform(seq) for seq in all_sequences]
print(f"✓ Sequences normalized", flush=True)

# Split train/val
print(f"Splitting data: 80% train, 20% val...", flush=True)
X_train, X_val, y_train, y_val = train_test_split(sequences_normalized, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}", flush=True)

# Create datasets and dataloaders
train_dataset = AimDataset(X_train, y_train)
val_dataset = AimDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, optimizer, loss
model = AimLSTM()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training for 10 epochs
num_epochs = 10
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