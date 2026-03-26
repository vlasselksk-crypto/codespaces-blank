import logging
import random
import io
import os
import time
import threading
import glob
import json

import requests

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional

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
suspicion_cache = {}


class AimLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(15, 128, batch_first=True)
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


def extract_advanced_features(ticks, hits):
    # ticks: list of dicts with keys: delta_yaw, delta_pitch, accel_yaw, accel_pitch, jerk_yaw, jerk_pitch, gcd_error_yaw, gcd_error_pitch
    
    # variance_yaw: дисперсия delta_yaw за последние 20 тиков
    if len(ticks) < 20:
        variance_yaw = 0.0
    else:
        deltas_yaw = [t.get('delta_yaw', 0) for t in ticks[-20:]]
        variance_yaw = np.var(deltas_yaw)
    
    # variance_pitch
    if len(ticks) < 20:
        variance_pitch = 0.0
    else:
        deltas_pitch = [t.get('delta_pitch', 0) for t in ticks[-20:]]
        variance_pitch = np.var(deltas_pitch)
    
    # hit_frequency: количество ударов за последние 2 секунды (40 тиков)
    if len(hits) < 40:
        hit_frequency = sum(hits)
    else:
        hit_frequency = sum(hits[-40:])
    
    # rotation_speed: среднее |delta_yaw| + |delta_pitch| за тик
    if not ticks:
        rotation_speed = 0.0
    else:
        rotation_speed = np.mean([abs(t.get('delta_yaw', 0)) + abs(t.get('delta_pitch', 0)) for t in ticks])
    
    # aim_consistency: среднее отклонение от идеального попадания (заглушка 0.0)
    aim_consistency = 0.0
    
    # jitter: среднее отклонение от сглаженного движения
    if len(ticks) < 3:
        jitter = 0.0
    else:
        deltas = [t.get('delta_yaw', 0) for t in ticks]
        smoothed = np.convolve(deltas, np.ones(3)/3, mode='valid')
        jitter = np.mean([abs(d - s) for d, s in zip(deltas[1:-1], smoothed)])
    
    # mouse_smoothing: корреляция последовательных дельт
    if len(ticks) < 2:
        mouse_smoothing = 0.0
    else:
        deltas = [t.get('delta_yaw', 0) for t in ticks]
        mouse_smoothing = np.corrcoef(deltas[:-1], deltas[1:])[0,1] if len(deltas) > 1 else 0.0
    
    return variance_yaw, variance_pitch, hit_frequency, rotation_speed, aim_consistency, jitter, mouse_smoothing


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
        player_id: Optional[str] = Query(None, description="Player ID for suspicion tracking"),
        hits: str = Query("[]", description="JSON list of hit history (last 100 ticks)"),
        x_api_key: str = Header(..., alias="X-API-Key"),
        api_key: str = Depends(get_api_key),
    ):
        if x_api_key != api_key:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")

        if player_id is None:
            player_id = "unknown"

        try:
            hits_list = json.loads(hits)
        except:
            hits_list = []

        content_type = request.headers.get("content-type", "").lower()
        ticks = None

        if content_type in ["application/octet-stream", "application/x-flatbuffers"]:
            # Parse as FlatBuffers
            body = await request.body()
            try:
                ticks = parse_tickdata_sequence(body)
            except Exception as exc:
                logger.exception("Invalid FlatBuffers payload")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid FlatBuffers payload",
                )
        elif content_type == "application/json":
            # Parse as JSON
            try:
                body_json = await request.json()
                ticks = body_json["ticks"]
                if "hits" in body_json:
                    hits_list = body_json["hits"]
                if "player_id" in body_json:
                    player_id = body_json["player_id"]
            except Exception as exc:
                logger.exception("Invalid JSON payload")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid JSON payload",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported content type. Use application/octet-stream for FlatBuffers or application/json for JSON",
            )

        if model is not None and scaler is not None:
            if ticks:
                data = []
                for tick in ticks:
                    data.append([
                        tick.get('delta_yaw', 0) if isinstance(tick, dict) else tick[0],
                        tick.get('delta_pitch', 0) if isinstance(tick, dict) else tick[1],
                        tick.get('accel_yaw', 0) if isinstance(tick, dict) else tick[2],
                        tick.get('accel_pitch', 0) if isinstance(tick, dict) else tick[3],
                        tick.get('jerk_yaw', 0) if isinstance(tick, dict) else tick[4],
                        tick.get('jerk_pitch', 0) if isinstance(tick, dict) else tick[5],
                        tick.get('gcd_error_yaw', 0) if isinstance(tick, dict) else tick[6],
                        tick.get('gcd_error_pitch', 0) if isinstance(tick, dict) else tick[7]
                    ])
                data = np.array(data)
                
                # Extract advanced features
                advanced_features = extract_advanced_features(ticks, hits_list)
                advanced_array = np.array(advanced_features).reshape(1, -1)  # (1, 7)
                advanced_tiled = np.tile(advanced_array, (len(data), 1))  # (n_ticks, 7)
                
                # Concatenate features
                data = np.concatenate([data, advanced_tiled], axis=1)  # (n_ticks, 15)
                
                # Normalize data
                data_normalized = scaler.transform(data)
                # Convert to tensor, add batch dim
                seq = torch.FloatTensor(data_normalized).unsqueeze(0)  # shape: (1, n_ticks, 15)
                with torch.no_grad():
                    probability = model(seq).item()
            else:
                probability = 0.1
        else:
            probability = 0.1

        # Hybrid system: rule-based suspicion
        suspicion = 0
        if ticks:
            rotation_speed = advanced_features[3]  # from extract_advanced_features
            hit_frequency = advanced_features[2]
            last_hit = hits_list and hits_list[-1] if hits_list else False
            
            # Rule 1: слишком быстрый поворот во время удара
            if rotation_speed > 3000 and last_hit:
                suspicion += 50
            
            # Rule 2: слишком частая атака
            if hit_frequency > 18:
                suspicion += 30
            
            # Если подозрение > 80 — мгновенный бан
            if suspicion > 80:
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "probability": 0.99,
                        "reason": "rule_based",
                        "ticks_received": len(ticks),
                    },
                )
            
            # Add rule suspicion to probability
            probability += suspicion / 100.0
            probability = min(probability, 0.99)

        # Suspicion accumulation
        if player_id not in suspicion_cache:
            suspicion_cache[player_id] = 0.0
        
        weight = 1.5 if hits_list and hits_list[-1] else 1.0
        suspicion_cache[player_id] += (probability - 0.5) * weight
        
        logger.info(f"Player {player_id}: suspicion={suspicion_cache[player_id]:.1f}, prob={probability:.3f}")
        
        flagged = False
        if suspicion_cache[player_id] > 40:
            flagged = True
            suspicion_cache[player_id] = 0.0  # Reset after flag

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "probability": probability,
                "ticks_received": len(ticks),
                "flagged": flagged,
                "suspicion_level": suspicion_cache[player_id],
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
        # Add dummy advanced features (7 zeros)
        dummy_advanced = np.zeros((all_data_flat.shape[0], 7))
        all_data_flat = np.concatenate([all_data_flat, dummy_advanced], axis=1)  # (n, 15)
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
                # Защита от пустых батчей
                if len(seqs) == 0:
                    logger.warning(f"⚠️ Empty batch, skipping")
                    continue
                
                optimizer.zero_grad()
                outputs = model(seqs)
                
                # Защита от пустых выходов
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    labels = labels[:1]  # подгоняем размер
                
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
                    # Защита от пустых батчей в валидации
                    if len(seqs) == 0:
                        logger.warning(f"⚠️ Empty validation batch, skipping")
                        continue
                    
                    outputs = model(seqs)
                    
                    # Защита от пустых выходов
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                        labels = labels[:1]
                    
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

    return app


app = create_app()
