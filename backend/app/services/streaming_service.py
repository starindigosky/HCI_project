import torch
import numpy as np
import os
import io
import logging
from typing import AsyncGenerator, Optional
from collections import deque
import asyncio

from .translation_service import translation_service
from .tts_service import tts_service

from ..models.schemas import LanguageType
from .asr_service import asr_service
from ..utils.async_helpers import run_in_thread

logger = logging.getLogger(__name__)


class StreamingASRService:
    """
    Service for real-time streaming ASR (Automatic Speech Recognition)
    Handles continuous audio stream from microphone
    """

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 1.0):
        """
        Initialize streaming ASR service

        Args:
            sample_rate: Audio sample rate (default: 16000 Hz)
            chunk_duration: Duration of each audio chunk in seconds (default: 1.0s)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        # Buffer for accumulating audio chunks
        self.audio_buffer = deque(maxlen=100)  # Max 100 chunks (~100 seconds)

        # Minimum audio length for transcription (in seconds)
        self.min_audio_duration = 4.0

        logger.info(
            f"Streaming ASR Service initialized: "
            f"sample_rate={sample_rate}, chunk_duration={chunk_duration}s"
        )

    def reset_buffer(self):
        """Reset the audio buffer"""
        self.audio_buffer.clear()
        logger.debug("Audio buffer reset")

    def add_audio_chunk(self, audio_chunk: bytes):
        """
        Add audio chunk to buffer

        Args:
            audio_chunk: Raw audio bytes (PCM format)
        """
        # Convert bytes to numpy array (assuming int16 PCM)
        try:
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            # Normalize to float32 [-1.0, 1.0]
            audio_array = audio_array.astype(np.float32) / 32768.0
            self.audio_buffer.append(audio_array)
            logger.debug(f"Added audio chunk: {len(audio_array)} samples")
        except Exception as e:
            logger.error(f"Error adding audio chunk: {str(e)}")
            raise

    def get_buffered_audio(self) -> np.ndarray:
        """
        Get all buffered audio as a single numpy array

        Returns:
            Combined audio array
        """
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)

        # Concatenate all chunks
        combined_audio = np.concatenate(list(self.audio_buffer))
        return combined_audio

    async def transcribe_stream(
        self,
        audio_chunk: bytes,
        language: LanguageType,
        interim_results: bool = False
    ) -> Optional[str]:
        """
        Transcribe streaming audio chunk

        Args:
            audio_chunk: Audio chunk bytes
            language: Language type
            interim_results: If True, return interim results (faster but less accurate)

        Returns:
            Transcribed text or None if not enough audio
        """
        # Add chunk to buffer
        self.add_audio_chunk(audio_chunk)

        # Get buffered audio
        audio = self.get_buffered_audio()

        # Check if we have enough audio
        duration = len(audio) / self.sample_rate

        if duration < self.min_audio_duration:
            logger.debug(f"Not enough audio yet: {duration:.2f}s")
            return None

        try:
            # Transcribe using existing ASR service (run in thread pool)
            text, confidence, processing_time = await run_in_thread(
                asr_service.transcribe_array,
                audio,
                self.sample_rate,
                language
            )

            logger.info(
                f"Streaming transcription: '{text}' "
                f"(duration: {duration:.2f}s, time: {processing_time:.2f}s)"
            )

            # 1. 翻譯: 中文文字 -> 閩南語文字
            translated_text, trans_time = await run_in_thread(
                translation_service.translate,
                text,
                LanguageType.CHINESE,
                LanguageType.MIN_NAN
             )
            logger.info(f"Translated to Min Nan: '{translated_text}' in {trans_time:.2f}s")
            
            # 2. 語音合成: 閩南語文字 -> 閩南語語音
            #    tts_service 會回傳檔案路徑 (e.g., "output/tts_xyz.wav")
            audio_path, tts_time = await run_in_thread(
                tts_service.text_to_speech,
                translated_text
            )
            logger.info(f"Generated Min Nan audio: '{audio_path}' in {tts_time:.2f}s")
            
            
            self.reset_buffer()
            
            # 4. 回傳音檔的路徑 (URL)
            #    我們需要將 "output\tts_xyz.wav" 轉成 "output/tts_xyz.wav"
            filename = os.path.basename(audio_path)

            # 我們的主程式 app/main.py 掛載在 "/output"
            # 所以 URL 必須是 "output/tts_123.wav"
            audio_url = f"output/{filename}"
            return {
                'translated_text': translated_text,
                'audio_url': audio_url
            }

        except Exception as e:
            logger.error(f"Error in streaming transcription: {str(e)}")
            self.reset_buffer()
            raise

    async def transcribe_final(
        self,
        language: LanguageType
    ) -> Optional[str]:
        """
        Transcribe all buffered audio (final result)

        Args:
            language: Language type

        Returns:
            Final transcribed text
        """
        audio = self.get_buffered_audio()

        if len(audio) == 0:
            return None

        try:
            # Transcribe using existing ASR service (run in thread pool)
            text, confidence, processing_time = await run_in_thread(
                asr_service.transcribe_array,
                audio,
                self.sample_rate,
                language
            )

            logger.info(f"Final transcription: '{text}'")

            return text

        except Exception as e:
            logger.error(f"Error in final transcription: {str(e)}")
            raise


class StreamingSessionManager:
    """Manage multiple streaming sessions"""

    def __init__(self):
        self.sessions = {}
        logger.info("Streaming Session Manager initialized")

    def create_session(self, session_id: str, sample_rate: int = 16000) -> StreamingASRService:
        """Create a new streaming session"""
        session = StreamingASRService(sample_rate=sample_rate)
        self.sessions[session_id] = session
        logger.info(f"Created streaming session: {session_id} with sample rate: {sample_rate}")
        return session

    def get_session(self, session_id: str) -> Optional[StreamingASRService]:
        """Get existing session"""
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        """Remove session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Removed streaming session: {session_id}")

    def cleanup_inactive_sessions(self, max_age_seconds: int = 3600):
        """Cleanup inactive sessions older than max_age_seconds"""
        # TODO: Implement timestamp tracking and cleanup
        pass


# Global streaming session manager
streaming_session_manager = StreamingSessionManager()
