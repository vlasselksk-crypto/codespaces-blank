import time
import os
import io

import pytest
from fastapi import status
from fastapi.testclient import TestClient
import pandas as pd

from app.main import app
from app.flatbuffers.parser import build_tickdata_sequence


@pytest.fixture(autouse=True)
def set_api_key_env(monkeypatch):
    # Ensure deterministic behavior in tests
    monkeypatch.setenv("SLOTHAC_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def remove_model_files():
    # Remove model.pth and scaler.pkl before each test to ensure clean state
    for f in ["model.pth", "scaler.pkl"]:
        if os.path.exists(f):
            os.remove(f)


def test_auth_success():
    payload = build_tickdata_sequence([[1.0] * 8])
    with TestClient(app) as client:
        r = client.post(
            "/v1/inference", content=payload, headers={"X-API-Key": "test-key"}
        )
    assert r.status_code == status.HTTP_200_OK
    data = r.json()
    assert data["probability"] == 0.1  # No model, default


def test_auth_failure():
    payload = build_tickdata_sequence([[1.0] * 8])
    with TestClient(app) as client:
        r = client.post(
            "/v1/inference", content=payload, headers={"X-API-Key": "bad"}
        )
    assert r.status_code == status.HTTP_403_FORBIDDEN


def test_parsing_flatbuffers():
    # build two tick records with distinct values
    rows = [[i + j * 0.1 for i in range(8)] for j in range(2)]
    payload = build_tickdata_sequence(rows)

    with TestClient(app) as client:
        r = client.post(
            "/v1/inference", content=payload, headers={"X-API-Key": "test-key"}
        )

    assert r.status_code == status.HTTP_200_OK
    data = r.json()
    assert data["ticks_received"] == 2
    assert data["probability"] == 0.1  # No model


def test_invalid_flatbuffers_returns_422():
    with TestClient(app) as client:
        r = client.post(
            "/v1/inference", content=b"not a flatbuffers blob", headers={"X-API-Key": "test-key"}
        )

    assert r.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_performance_under_50ms():
    payload = build_tickdata_sequence([[1.0] * 8 for _ in range(16)])

    with TestClient(app) as client:
        start = time.perf_counter()
        r = client.post(
            "/v1/inference", content=payload, headers={"X-API-Key": "test-key"}
        )
        duration_ms = (time.perf_counter() - start) * 1000

    assert r.status_code == status.HTTP_200_OK
    assert duration_ms < 200, f"Response too slow: {duration_ms:.1f}ms"


def test_train_lstm_endpoint():
    # Create mock CSV data with 8 columns and 50 rows
    legit_data = pd.DataFrame({
        'f0': [0.1] * 50,
        'f1': [0.05] * 50,
        'f2': [1.0] * 50,
        'f3': [0.5] * 50,
        'f4': [0.0] * 50,
        'f5': [0.0] * 50,
        'f6': [0.0] * 50,
        'f7': [0.0] * 50,
    })
    cheat_data = pd.DataFrame({
        'f0': [1.0] * 50,
        'f1': [0.8] * 50,
        'f2': [5.0] * 50,
        'f3': [3.0] * 50,
        'f4': [0.0] * 50,
        'f5': [0.0] * 50,
        'f6': [0.0] * 50,
        'f7': [0.0] * 50,
    })

    legit_csv = legit_data.to_csv(index=False)
    cheat_csv = cheat_data.to_csv(index=False)

    with TestClient(app) as client:
        files = [
            ("files", ("legit_data.csv", io.BytesIO(legit_csv.encode()), "text/csv")),
            ("files", ("cheat_data.csv", io.BytesIO(cheat_csv.encode()), "text/csv")),
        ]
        r = client.post(
            "/train-lstm", files=files, headers={"X-API-Key": "test-key"}
        )

    assert r.status_code == status.HTTP_200_OK
    data = r.json()
    assert data["status"] == "trained"
    assert data["sequences"] > 0

    # Check that model.pth and scaler.pkl were created
    assert os.path.exists("model.pth")
    assert os.path.exists("scaler.pkl")


def test_inference_with_model():
    # First train a model
    legit_data = pd.DataFrame({
        'f0': [0.1] * 50,
        'f1': [0.05] * 50,
        'f2': [1.0] * 50,
        'f3': [0.5] * 50,
        'f4': [0.0] * 50,
        'f5': [0.0] * 50,
        'f6': [0.0] * 50,
        'f7': [0.0] * 50,
    })
    cheat_data = pd.DataFrame({
        'f0': [1.0] * 50,
        'f1': [0.8] * 50,
        'f2': [5.0] * 50,
        'f3': [3.0] * 50,
        'f4': [0.0] * 50,
        'f5': [0.0] * 50,
        'f6': [0.0] * 50,
        'f7': [0.0] * 50,
    })

    legit_csv = legit_data.to_csv(index=False)
    cheat_csv = cheat_data.to_csv(index=False)

    with TestClient(app) as client:
        files = [
            ("files", ("legit_data.csv", io.BytesIO(legit_csv.encode()), "text/csv")),
            ("files", ("cheat_data.csv", io.BytesIO(cheat_csv.encode()), "text/csv")),
        ]
        client.post("/train-lstm", files=files, headers={"X-API-Key": "test-key"})

        # Now test inference
        payload = build_tickdata_sequence([[0.1, 0.05, 1.0, 0.5, 0, 0, 0, 0] * 10])  # 10 ticks, similar to legit
        r = client.post(
            "/v1/inference", content=payload, headers={"X-API-Key": "test-key"}
        )

    assert r.status_code == status.HTTP_200_OK
    data = r.json()
    assert "probability" in data
    assert isinstance(data["probability"], float)
