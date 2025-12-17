#!/usr/bin/env python3
import os

# Disable Hugging Face Xet accelerator to prevent panic/crashes during download
os.environ["HF_HUB_DISABLE_XET"] = "1"
"""
Main entry point for running the FastAPI application
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],  # Only watch app code, ignore model_cache
        log_level="info",
    )
