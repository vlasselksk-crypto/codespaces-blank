import logging
import random

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_api_key
from app.flatbuffers.parser import parse_tickdata_sequence


logger = logging.getLogger("slothac_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def create_app() -> FastAPI:
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

        probability = random.random()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "probability": probability,
                "ticks_received": len(ticks),
            },
        )

    return app


app = create_app()
