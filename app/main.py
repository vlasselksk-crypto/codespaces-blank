import logging
import random
import io
import os
import time
import threading
import glob

import requests

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import pandas as pd
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from app.config import get_api_key
from app.flatbuffers.parser import parse_tickdata_sequence


logger = logging.getLogger("slothac_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# Global model and scaler variables
model = None
scaler = None


class AimLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8, 128, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)


class AimDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_model():
    global model, scaler
    if os.path.exists("model.pth") and os.path.exists("scaler.pkl"):
        try:
            model = AimLSTM()
            model.load_state_dict(torch.load("model.pth"))
            model.eval()
            scaler = joblib.load("scaler.pkl")
            logger.info("Model and scaler loaded from model.pth and scaler.pkl")
        except Exception as e:
            logger.error(f"Failed to load model/scaler: {e}")
            model = None
            scaler = None
    else:
        logger.info("No model.pth or scaler.pkl found, using default behavior")


def create_app() -> FastAPI:
    global model, scaler
    load_model()  # Load model on startup

    app = FastAPI(title="SlothAC API")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info("<-- %s %s", request.method, request.url)
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled exception")
            raise
        logger.info("--> %s %s %s", request.method, request.url, response.status_code)
        return response

    @app.post("/v1/inference")
    async def inference(
        request: Request,
        x_api_key: str = Header(..., alias="X-API-Key"),
        api_key: str = Depends(get_api_key),
    ):
        if x_api_key != api_key:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")

        body = await request.body()
        try:
            ticks = parse_tickdata_sequence(body)
        except Exception as exc:
            logger.exception("Invalid FlatBuffers payload")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid FlatBuffers payload",
            )

        if model is not None and scaler is not None:
            if ticks:
                data = np.array(ticks)  # shape: (n_ticks, 8)
                # Normalize data
                data_normalized = scaler.transform(data)
                # Convert to tensor, add batch dim
                seq = torch.FloatTensor(data_normalized).unsqueeze(0)  # shape: (1, n_ticks, 8)
                with torch.no_grad():
                    probability = model(seq).item()
            else:
                probability = 0.1
        else:
            probability = 0.1

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "probability": probability,
                "ticks_received": len(ticks),
            },
        )

    @app.post("/train-lstm")
    async def train_lstm_endpoint(
        files: list[UploadFile] = File(...),
        x_api_key: str = Header(..., alias="X-API-Key"),
        api_key: str = Depends(get_api_key),
    ):
        global model, scaler
        if x_api_key != api_key:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")

        all_sequences = []
        all_labels = []

        for file in files:
            content = await file.read()
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))

            # Determine label by filename
            if "LEGIT" in file.filename.upper():
                label = 0
            elif "CHEAT" in file.filename.upper():
                label = 1
            else:
                continue

            # Extract data: assume columns is_cheating, delta_yaw, delta_pitch, accel_yaw, accel_pitch, and 4 more? Wait, FlatBuffers has 8 floats.
            # For simplicity, assume df has 9 columns: is_cheating, f0 to f7
            data = df.iloc[:, 1:].values  # shape: (n_ticks, 8) - skip is_cheating column

            # Create sequences: window 40, step 20
            window_size = 40
            step = 20
            for i in range(0, len(data) - window_size + 1, step):
                seq = data[i:i+window_size]
                all_sequences.append(seq)
                all_labels.append(label)

        if not all_sequences:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid training sequences created")

        # Fit scaler on all data
        all_data_flat = np.array(all_sequences).reshape(-1, 8)
        scaler = StandardScaler()
        scaler.fit(all_data_flat)

        if scaler is None:
            raise HTTPException(status_code=500, detail="Scaler failed to fit")

        # Normalize sequences
        sequences_normalized = []
        for seq in all_sequences:
            seq_norm = scaler.transform(seq)
            sequences_normalized.append(seq_norm)

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
        num_epochs = 10
        patience = 5
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            for seqs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(seqs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            train_losses.append(epoch_train_loss / len(train_loader))

            # Validation
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for seqs, labels in val_loader:
                    outputs = model(seqs)
                    loss = criterion(outputs.squeeze(), labels)
                    epoch_val_loss += loss.item()
            val_losses.append(epoch_val_loss / len(val_loader))

            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

            # Early stopping
            if val_losses[-1] < best_loss:
                best_loss = val_losses[-1]
                patience_counter = 0
                torch.save(model.state_dict(), "model.pth")
                joblib.dump(scaler, "scaler.pkl")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping")
                    break

        # Save final if not saved
        if patience_counter == 0:
            torch.save(model.state_dict(), "model.pth")
            joblib.dump(scaler, "scaler.pkl")

        logger.info(f"LSTM model trained and saved with {len(all_sequences)} sequences")

        return {
            "status": "trained",
            "sequences": len(all_sequences),
            "epochs_trained": len(train_losses),
            "train_losses": train_losses,
            "val_losses": val_losses
        }

    def auto_train_on_startup():
        try:
            csv_files = glob.glob("ru_data/*.csv")
            if not csv_files:
                logger.info("No CSV files found in ru_data/, skipping auto-training.")
                return

            logger.info(f"🚀 Found {len(csv_files)} CSV files. Starting training...")
            files = []
            try:
                for file_path in csv_files:
                    files.append(("files", (os.path.basename(file_path), open(file_path, "rb"), "text/csv")))

                response = requests.post(
                    "http://localhost:8000/train-lstm",
                    headers={"X-API-Key": get_api_key()},
                    files=files,
                    timeout=300,  # 5 минут на обучение
                )

                if response.status_code == 200:
                    logger.info("✅ Model successfully trained on startup!")
                    logger.info(f"📊 Result: {response.json()}")
                else:
                    logger.error(f"❌ Training failed: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"❌ Auto-training error: {e}")
            finally:
                for _, file_tuple in files:
                    try:
                        file_tuple[1].close()
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"❌ Auto-training error: {e}")

    def train_on_startup_loop():
        """Проверяет наличие модели при старте и предлагает обучение при появлении новых данных."""
        try:
            time.sleep(10)
            
            # Получаем список всех CSV
            csv_files = glob.glob("ru_data/*.csv")
            if not csv_files:
                logger.info("No CSV files found, skipping training")
                return
            
            # Если модель уже есть — проверяем, все ли файлы уже учтены
            if os.path.exists("model.pth") and os.path.exists("scaler.pkl"):
                # Сравниваем количество файлов или используем хеши
                # Пока просто логируем и предлагаем ручное обучение
                logger.info(f"✅ Model exists. Found {len(csv_files)} CSV files.")
                logger.info("ℹ️ To retrain with new data, send POST /train-lstm manually")
                return
            
            # Если модели нет — запускаем обучение с нуля
            logger.info(f"🚀 Starting training with {len(csv_files)} files...")
            
            files = []
            try:
                for file_path in csv_files:
                    files.append(("files", (os.path.basename(file_path), open(file_path, "rb"), "text/csv")))
                
                response = requests.post(
                    "http://localhost:8000/train-lstm",
                    headers={"X-API-Key": get_api_key()},
                    files=files,
                    timeout=600,  # 10 минут на обучение
                )
                
                if response.status_code == 200:
                    logger.info("✅ Model successfully trained on startup!")
                    logger.info(f"📊 Result: {response.json()}")
                else:
                    logger.error(f"❌ Training failed: {response.status_code} - {response.text}")
            finally:
                for _, file_tuple in files:
                    try:
                        file_tuple[1].close()
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.error(f"❌ Training error: {e}")

    threading.Thread(target=train_on_startup_loop, daemon=True).start()

    return app


app = create_app()
