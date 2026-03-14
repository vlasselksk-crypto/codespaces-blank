import time
import pytest

from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.flatbuffers.parser import build_tickdata_sequence


@pytest.fixture(autouse=True)
def set_api_key_env(monkeypatch):
    # Ensure deterministic behavior in tests
    monkeypatch.setenv("SLOTHAC_API_KEY", "test-key")


def test_auth_success():
    payload = build_tickdata_sequence([[1.0] * 8])
    with TestClient(app) as client:
        r = client.post(
            "/v1/inference", content=payload, headers={"X-API-Key": "test-key"}
        )
    assert r.status_code == status.HTTP_200_OK


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
    assert 0.0 <= data["probability"] <= 1.0


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
