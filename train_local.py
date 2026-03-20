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

for file_path in csv_files:
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path).upper()
    if "LEGIT" in filename:
        label = 0
    elif "CHEAT" in filename:
        label = 1
    else:
        continue
    
    # Assume data starts from second column (first is is_cheating, but we use filename)
    data = df.iloc[:, 1:].values  # shape: (n_ticks, 8)
    
    # Create sequences: window 40, step 20
    window_size = 40
    step = 20
    for i in range(0, len(data) - window_size + 1, step):
        seq = data[i:i+window_size]
        all_sequences.append(seq)
        all_labels.append(label)

if not all_sequences:
    print("No sequences created")
    exit(1)

# Fit scaler
all_data_flat = np.array(all_sequences).reshape(-1, 8)
scaler = StandardScaler()
scaler.fit(all_data_flat)

# Normalize sequences
sequences_normalized = [scaler.transform(seq) for seq in all_sequences]

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

# Training for 10 epochs
num_epochs = 10
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
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save model and scaler
torch.save(model.state_dict(), "model.pth")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved.")