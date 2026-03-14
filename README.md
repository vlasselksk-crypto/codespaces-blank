# SlothAC API Server (FastAPI)

A small FastAPI service for a fake anti-cheat inference endpoint (`/v1/inference`) that accepts FlatBuffers-encoded TickData.

## Features

- **FlatBuffers request parsing** (TickData / TickDataSequence structures)
- **API key authentication** via `X-API-Key` header
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

### 3) Run tests

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
