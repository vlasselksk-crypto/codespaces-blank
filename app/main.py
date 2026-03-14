import logging
import random
import io
import os

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.config import get_api_key
from app.flatbuffers.parser import parse_tickdata_sequence


logger = logging.getLogger("slothac_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# Global model variable
model = None


def load_model():
    global model
    if os.path.exists("model.pkl"):
        try:
            model = joblib.load("model.pkl")
            logger.info("Model loaded from model.pkl")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model = None
    else:
        logger.info("No model.pkl found, using default behavior")


def create_app() -> FastAPI:
    global model
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

        if model is not None:
            # Extract features: mean and std for f0,f1,f2,f3 (assuming delta_yaw, delta_pitch, accel_yaw, accel_pitch)
            if ticks:
                data = np.array(ticks)  # shape: (n_ticks, 8)
                features = [
                    np.mean(data[:, 0]), np.std(data[:, 0]),  # f0: delta_yaw
                    np.mean(data[:, 1]), np.std(data[:, 1]),  # f1: delta_pitch
                    np.mean(data[:, 2]), np.std(data[:, 2]),  # f2: accel_yaw
                    np.mean(data[:, 3]), np.std(data[:, 3]),  # f3: accel_pitch
                ]
                probability = model.predict_proba([features])[0][1]  # Probability of class 1 (cheat)
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

    @app.post("/train")
    async def train_model_endpoint(
        files: list[UploadFile] = File(...),
        x_api_key: str = Header(..., alias="X-API-Key"),
        api_key: str = Depends(get_api_key),
    ):
        global model
        if x_api_key != api_key:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")

        X, y = [], []

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

            # Extract features from entire file
            features = [
                df['delta_yaw'].mean(), df['delta_yaw'].std(),
                df['delta_pitch'].mean(), df['delta_pitch'].std(),
                df['accel_yaw'].mean(), df['accel_yaw'].std(),
                df['accel_pitch'].mean(), df['accel_pitch'].std()
            ]
            X.append(features)
            y.append(label)

        if not X:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid training files provided")

        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        # Save model
        joblib.dump(model, 'model.pkl')
        logger.info(f"Model trained and saved with {len(X)} samples")

        return {"status": "trained", "samples": len(X)}

    return app


app = create_app()
