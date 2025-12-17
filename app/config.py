from pydantic_settings import BaseSettings

import os
import tempfile

# Disable Hugging Face Xet accelerator to prevent panic/crashes during download
os.environ["HF_HUB_DISABLE_XET"] = "1"


class Settings(BaseSettings):
    """Application settings and configuration"""

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Min Nan & Chinese Voice Chatbot API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "API for Chinese and Min Nan voice-to-text and text-to-voice conversion"
    )

    # CORS Settings
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["*"]

    # Model Settings
    CHINESE_ASR_MODEL: str = "openai/whisper-base"  # Optimized for speed (was large-v3)
    MIN_NAN_ASR_MODEL: str = (
        "emlinking/wav2vec2-large-xls-r-300m-tsm-asr-v6"  # Fine-tuned for Taiwanese Southern Min (Hokkien)
    )
    MIN_NAN_TTS_MODEL: str = "facebook/mms-tts-nan"  # Min Nan TTS model

    # File Upload Settings
    # Use absolute paths relative to the project root
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")

    # Store outputs in 'outputs' directory within project, but outside 'app'
    # (so it won't trigger reloads if run.py watches 'app')
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")

    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_AUDIO_FORMATS: list = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]

    # Audio Processing Settings
    SAMPLE_RATE: int = 16000  # Standard sample rate for ASR models

    # Device Settings (CPU/CUDA)
    DEVICE: str = "cpu"  # Will auto-detect CUDA if available

    # Cache Settings
    MODEL_CACHE_DIR: str = "./model_cache"

    class Config:
        case_sensitive = True
        # env_file = ".env"  # Disable .env loading to ensure code defaults are used


settings = Settings()

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
