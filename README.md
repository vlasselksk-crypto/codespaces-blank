# SlothAC API Server (FastAPI) - MLSAC Architecture

A FastAPI service for anti-cheat inference using LSTM neural network (`/v1/inference`) that accepts FlatBuffers-encoded TickData, with on-server model training via `/train-lstm`.

## Features

- **PyTorch LSTM neural network** (MLSAC architecture: 2 LSTM layers, dropout, sigmoid output)
- **Sequence-based training** (windows of 40 ticks, step 20)
- **FlatBuffers request parsing** (TickData / TickDataSequence structures)
- **API key authentication** via `X-API-Key` header
- **On-server LSTM training** via `/train-lstm` endpoint (accepts CSV files)
- **Model persistence** (saves/loads `model.pth` and `scaler.pkl`)
- **Data normalization** with StandardScaler
- **Early stopping** during training
- **Training history** (loss tracking)
- **CORS enabled**
- **Logging** for all requests
- **Production-ready error handling**
- Fully working **pytest** test suite

---

## Quick start (local)

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server expects a valid API key in the `X-API-Key` header.

By default, the server uses the API key from `SLOTHAC_API_KEY` in the environment. If unset, it defaults to `test-key`.

### 3) Train a model (optional)

Upload CSV files with 8 columns (f0-f7, corresponding to FlatBuffers fields).

Filename must contain "LEGIT" or "CHEAT" to label the data.

Each CSV should have at least 40 rows for sequence creation.

```bash
curl -X POST "http://localhost:8000/train-lstm" \
  -H "X-API-Key: test-key" \
  -F "files=@legit_data.csv" \
  -F "files=@cheat_data.csv"
```

### 4) Run tests

```bash
pytest
```

---

## Docker

Build:

```bash
docker build -t slothac-api .
```

Run:

```bash
docker run -e SLOTHAC_API_KEY=test-key -p 8000:8000 slothac-api
```

Or with `docker-compose`:

```bash
docker-compose up --build
```

---

## FlatBuffers Schema

The FlatBuffers schema is located at `app/flatbuffers/schema/tickdata.fbs`.

The endpoint expects the request body to be FlatBuffers-encoded `TickDataSequence`.

For inference, the entire tick sequence is normalized and fed to the LSTM model.

---

## API Endpoints

### POST `/v1/inference`

- **Auth**: `X-API-Key` header
- **Body**: FlatBuffers `TickDataSequence`
- **Response**: `{"probability": float, "ticks_received": int}`
- **Logic**: If model exists, predicts cheat probability using LSTM; else returns 0.1

### POST `/train-lstm`

- **Auth**: `X-API-Key` header
- **Body**: Multipart form-data with `files` (CSV files, 8 columns, >=40 rows each)
- **Response**: `{"status": "trained", "sequences": int, "epochs_trained": int, "train_losses": list, "val_losses": list}`
- **Logic**: Creates sequences (40 ticks window, 20 step), trains LSTM with validation, early stopping, saves model and scaler
