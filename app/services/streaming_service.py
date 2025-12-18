import torch
import numpy as np
import io
import logging
from typing import AsyncGenerator, Optional, Tuple
from collections import deque
import asyncio

from ..models.schemas import LanguageType
from .asr_service import asr_service
from ..utils.async_helpers import run_in_thread

logger = logging.getLogger(__name__)


class StreamingASRService:

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 1.0):

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        self.audio_buffer = deque(maxlen=2000)

        self.min_audio_duration = 0.5

        logger.info(
            f"Streaming ASR Service initialized: "
            f"sample_rate={sample_rate}, chunk_duration={chunk_duration}s"
        )

        self.silence_counter = 0

        self.last_final_text = ""

    def reset_buffer(self):

        self.audio_buffer.clear()
        logger.debug("Audio buffer reset")

    def add_audio_chunk(self, audio_chunk: bytes):

        try:
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

            audio_array = audio_array.astype(np.float32) / 32768.0
            self.audio_buffer.append(audio_array)
            logger.debug(f"Added audio chunk: {len(audio_array)} samples")
        except Exception as e:
            logger.error(f"Error adding audio chunk: {str(e)}")
            raise

    def get_buffered_audio(self) -> np.ndarray:

        if not self.audio_buffer:
            return np.array([], dtype=np.float32)

        combined_audio = np.concatenate(list(self.audio_buffer))
        return combined_audio

    async def transcribe_stream(
        self, audio_chunk: bytes, language: LanguageType, interim_results: bool = False
    ) -> Tuple[Optional[str], bool]:

        self.add_audio_chunk(audio_chunk)

        audio = self.get_buffered_audio()

        duration = len(audio) / self.sample_rate
        if duration < self.min_audio_duration:
            logger.debug(f"Not enough audio yet: {duration:.2f}s")
            return None, False

        rms = np.sqrt(np.mean(audio**2))
        SILENCE_THRESHOLD = 0.002

        is_final = False

        if rms < SILENCE_THRESHOLD:
            self.silence_counter += 1
            if self.silence_counter >= 6:
                if len(self.audio_buffer) > 0:
                    logger.info(
                        f"Silence detected (RMS < {SILENCE_THRESHOLD}), finalizing transcription"
                    )
                    is_final = True

                else:
                    self.silence_counter = 0
                    return None, False
            else:

                return None, False
        else:
            self.silence_counter = 0

            self.last_final_text = ""

        import tempfile
        import soundfile as sf
        import os

        tmp_path = None
        try:

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

                await run_in_thread(sf.write, tmp_path, audio, self.sample_rate)

            text, confidence, processing_time = await run_in_thread(
                asr_service.transcribe, tmp_path, language
            )

            logger.info(
                f"Streaming transcription result: '{text}' "
                f"(duration: {duration:.2f}s, time: {processing_time:.2f}s, chunks: {len(self.audio_buffer)})"
                f"{' [FINAL]' if is_final else ''}"
            )

            if is_final:

                if text == self.last_final_text:
                    logger.info(
                        f"Duplicate final result detected ('{text}'), suppressing"
                    )
                    self.reset_buffer()
                    self.silence_counter = 0
                    return None, False

                self.last_final_text = text
                self.reset_buffer()
                self.silence_counter = 0

            return text, is_final

        except Exception as e:
            logger.error(f"Error in streaming transcription: {str(e)}")
            raise
        finally:

            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup temp file {tmp_path}: {cleanup_error}"
                    )

    async def transcribe_final(self, language: LanguageType) -> Optional[str]:

        audio = self.get_buffered_audio()

        if len(audio) == 0:
            return None

        import tempfile
        import soundfile as sf
        import os

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

                await run_in_thread(sf.write, tmp_path, audio, self.sample_rate)

            text, confidence, processing_time = await run_in_thread(
                asr_service.transcribe, tmp_path, language
            )

            logger.info(f"Final transcription: '{text}'")

            return text

        except Exception as e:
            logger.error(f"Error in final transcription: {str(e)}")
            raise
        finally:

            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup temp file {tmp_path}: {cleanup_error}"
                    )


class StreamingSessionManager:

    def __init__(self):
        self.sessions = {}
        logger.info("Streaming Session Manager initialized")

    def create_session(self, session_id: str) -> StreamingASRService:

        session = StreamingASRService()
        self.sessions[session_id] = session
        logger.info(f"Created streaming session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[StreamingASRService]:

        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):

        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Removed streaming session: {session_id}")

    def cleanup_inactive_sessions(self, max_age_seconds: int = 3600):

        pass


streaming_session_manager = StreamingSessionManager()
