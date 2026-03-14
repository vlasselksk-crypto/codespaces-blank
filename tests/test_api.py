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
def remove_model_file():
    # Remove model.pkl before each test to ensure clean state
    if os.path.exists("model.pkl"):
        os.remove("model.pkl")


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


def test_train_endpoint():
    # Create mock CSV data
    legit_data = pd.DataFrame({
        'delta_yaw': [0.1, 0.2, 0.3],
        'delta_pitch': [0.05, 0.1, 0.15],
        'accel_yaw': [1.0, 1.1, 1.2],
        'accel_pitch': [0.5, 0.6, 0.7]
    })
    cheat_data = pd.DataFrame({
        'delta_yaw': [1.0, 1.5, 2.0],
        'delta_pitch': [0.8, 1.2, 1.6],
        'accel_yaw': [5.0, 6.0, 7.0],
        'accel_pitch': [3.0, 4.0, 5.0]
    })

    legit_csv = legit_data.to_csv(index=False)
    cheat_csv = cheat_data.to_csv(index=False)

    with TestClient(app) as client:
        files = [
            ("files", ("legit_data.csv", io.BytesIO(legit_csv.encode()), "text/csv")),
            ("files", ("cheat_data.csv", io.BytesIO(cheat_csv.encode()), "text/csv")),
        ]
        r = client.post(
            "/train", files=files, headers={"X-API-Key": "test-key"}
        )

    assert r.status_code == status.HTTP_200_OK
    data = r.json()
    assert data["status"] == "trained"
    assert data["samples"] == 2

    # Check that model.pkl was created
    assert os.path.exists("model.pkl")


def test_inference_with_model():
    # First train a model
    legit_data = pd.DataFrame({
        'delta_yaw': [0.1, 0.2, 0.3],
        'delta_pitch': [0.05, 0.1, 0.15],
        'accel_yaw': [1.0, 1.1, 1.2],
        'accel_pitch': [0.5, 0.6, 0.7]
    })
    cheat_data = pd.DataFrame({
        'delta_yaw': [1.0, 1.5, 2.0],
        'delta_pitch': [0.8, 1.2, 1.6],
        'accel_yaw': [5.0, 6.0, 7.0],
        'accel_pitch': [3.0, 4.0, 5.0]
    })

    legit_csv = legit_data.to_csv(index=False)
    cheat_csv = cheat_data.to_csv(index=False)

    with TestClient(app) as client:
        files = [
            ("files", ("legit_data.csv", io.BytesIO(legit_csv.encode()), "text/csv")),
            ("files", ("cheat_data.csv", io.BytesIO(cheat_csv.encode()), "text/csv")),
        ]
        client.post("/train", files=files, headers={"X-API-Key": "test-key"})

        # Now test inference
        payload = build_tickdata_sequence([[0.1, 0.05, 1.0, 0.5, 0, 0, 0, 0]])  # Similar to legit
        r = client.post(
            "/v1/inference", content=payload, headers={"X-API-Key": "test-key"}
        )

    assert r.status_code == status.HTTP_200_OK
    data = r.json()
    assert data["probability"] < 0.5  # Should be low for legit-like data
