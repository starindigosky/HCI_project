from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class LanguageType(str, Enum):

    CHINESE = "chinese"
    MIN_NAN = "min_nan"
    ZH_TW = "zh_tw"


class ASRRequest(BaseModel):

    language: LanguageType = Field(
        ...,
        description="Language of the input audio (chinese or min_nan)"
    )


class ASRResponse(BaseModel):

    text: str = Field(..., description="Transcribed text from audio")
    language: LanguageType = Field(..., description="Detected/specified language")
    confidence: Optional[float] = Field(None, description="Confidence score if available")
    processing_time: float = Field(..., description="Processing time in seconds")


class TTSRequest(BaseModel):

    text: str = Field(..., description="Text to convert to speech")
    source_language: LanguageType = Field(
        default=LanguageType.CHINESE,
        description="Source language of the text"
    )
    target_language: LanguageType = Field(
        default=LanguageType.MIN_NAN,
        description="Target language for speech output"
    )


class TTSResponse(BaseModel):

    audio_url: str = Field(..., description="URL to download the generated audio")
    text: str = Field(..., description="Original text")
    source_language: LanguageType = Field(..., description="Source language")
    target_language: LanguageType = Field(..., description="Target language")
    processing_time: float = Field(..., description="Processing time in seconds")


class VoiceConversionRequest(BaseModel):

    source_language: LanguageType = Field(
        default=LanguageType.CHINESE,
        description="Source language of the input audio"
    )
    target_language: LanguageType = Field(
        default=LanguageType.MIN_NAN,
        description="Target language for output audio"
    )


class VoiceConversionResponse(BaseModel):

    transcribed_text: str = Field(..., description="Transcribed text from input audio")
    audio_url: str = Field(..., description="URL to download the converted audio")
    source_language: LanguageType = Field(..., description="Source language")
    target_language: LanguageType = Field(..., description="Target language")
    processing_time: float = Field(..., description="Total processing time in seconds")


class HealthResponse(BaseModel):

    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether models are loaded")


class ErrorResponse(BaseModel):

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")