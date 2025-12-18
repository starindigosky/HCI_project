from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
import logging
import sys

from .config import settings
from .api.routes import router
from .api.websocket_routes import router as ws_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )



@app.on_event("startup")
async def startup_event():
    logger.info("=" * 80)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info("=" * 80)
    logger.info(f"API Documentation: http://localhost:8000/docs")
    logger.info(f"ReDoc Documentation: http://localhost:8000/redoc")
    logger.info("=" * 80)















@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")



app.include_router(router, prefix=settings.API_V1_STR)


app.include_router(ws_router, prefix=settings.API_V1_STR)



app.mount("/", StaticFiles(directory="Website", html=True), name="site")






if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )