from pydantic_settings import BaseSettings

import os
import tempfile


os.environ["HF_HUB_DISABLE_XET"] = "1"


class Settings(BaseSettings):



    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Min Nan & Chinese Voice Chatbot API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "API for Chinese and Min Nan voice-to-text and text-to-voice conversion"
    )



    BACKEND_CORS_ORIGINS: list = ["*"]


    CHINESE_ASR_MODEL: str = "openai/whisper-base"
    MIN_NAN_ASR_MODEL: str = (
        "emlinking/wav2vec2-large-xls-r-300m-tsm-asr-v6"
    )
    MIN_NAN_TTS_MODEL: str = "facebook/mms-tts-nan"



    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")



    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")

    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024
    ALLOWED_AUDIO_FORMATS: list = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]


    SAMPLE_RATE: int = 16000


    DEVICE: str = "cpu"


    MODEL_CACHE_DIR: str = "./model_cache"

    class Config:
        case_sensitive = True



settings = Settings()


os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)