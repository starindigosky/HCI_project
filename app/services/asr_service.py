import torch
import librosa
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
import time
import logging
import os
from typing import Tuple, Optional
from ..config import settings
from ..models.schemas import LanguageType
import warnings
from opencc import OpenCC

logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ASRService:


    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ASR Service initializing on device: {self.device}")


        self.chinese_processor = None
        self.chinese_model = None
        self.min_nan_processor = None
        self.min_nan_model = None

        self.models_loaded = False


        self.cc = OpenCC("s2t")

    def load_models(self):

        try:
            logger.info("Loading ASR models...")


            logger.info(
                f"Loading Chinese ASR model: {settings.CHINESE_ASR_MODEL} (Cache: {settings.MODEL_CACHE_DIR})"
            )
            if not settings.CHINESE_ASR_MODEL:
                raise ValueError("CHINESE_ASR_MODEL is not set")

            self.chinese_processor = WhisperProcessor.from_pretrained(
                settings.CHINESE_ASR_MODEL, cache_dir=settings.MODEL_CACHE_DIR
            )
            self.chinese_model = WhisperForConditionalGeneration.from_pretrained(
                settings.CHINESE_ASR_MODEL, cache_dir=settings.MODEL_CACHE_DIR
            )
            self.chinese_model.to(self.device)
            self.chinese_model.eval()



            logger.info(f"Loading Min Nan ASR model: {settings.MIN_NAN_ASR_MODEL}")
            logger.debug(f"Min Nan ASR model path: {settings.MIN_NAN_ASR_MODEL}")
            logger.debug(f"Model cache directory: {settings.MODEL_CACHE_DIR}")
            logger.debug(f"HF_HOME environment variable: {os.environ.get('HF_HOME')}")
            self.min_nan_processor = Wav2Vec2Processor.from_pretrained(
                settings.MIN_NAN_ASR_MODEL, cache_dir=settings.MODEL_CACHE_DIR
            )
            self.min_nan_model = Wav2Vec2ForCTC.from_pretrained(
                settings.MIN_NAN_ASR_MODEL, cache_dir=settings.MODEL_CACHE_DIR
            )
            self.min_nan_model.to(self.device)
            self.min_nan_model.eval()

            self.models_loaded = True
            logger.info("ASR models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading ASR models: {str(e)}")
            raise

    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:

        try:

            audio, sample_rate = librosa.load(
                audio_path, sr=settings.SAMPLE_RATE, mono=True
            )
            return audio, sample_rate
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise

    def transcribe_chinese(self, audio_path: str) -> Tuple[str, Optional[float], float]:

        start_time = time.time()

        try:
            if not self.models_loaded:
                self.load_models()


            audio, sample_rate = self.preprocess_audio(audio_path)


            input_features = self.chinese_processor(
                audio, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features

            input_features = input_features.to(self.device)


            with torch.no_grad():

                forced_decoder_ids = self.chinese_processor.get_decoder_prompt_ids(
                    language="zh", task="transcribe"
                )
                predicted_ids = self.chinese_model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    no_repeat_ngram_size=3,
                    compression_ratio_threshold=1.35,
                )


            transcription = self.chinese_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]


            transcription = self.cc.convert(transcription)

            processing_time = time.time() - start_time

            logger.info(f"Chinese transcription completed in {processing_time:.2f}s")
            return transcription, None, processing_time

        except Exception as e:
            logger.error(f"Error in Chinese transcription: {str(e)}")
            raise

    def transcribe_min_nan(self, audio_path: str) -> Tuple[str, Optional[float], float]:

        start_time = time.time()

        try:
            if not self.models_loaded:
                self.load_models()


            audio, sample_rate = self.preprocess_audio(audio_path)


            inputs = self.min_nan_processor(
                audio, sampling_rate=sample_rate, return_tensors="pt", padding=True
            )

            input_values = inputs.input_values.to(self.device)


            with torch.no_grad():
                logits = self.min_nan_model(input_values).logits


            predicted_ids = torch.argmax(logits, dim=-1)


            transcription = self.min_nan_processor.batch_decode(predicted_ids)[0]

            processing_time = time.time() - start_time

            logger.info(f"Min Nan transcription completed in {processing_time:.2f}s")
            return transcription, None, processing_time

        except Exception as e:
            logger.error(f"Error in Min Nan transcription: {str(e)}")
            raise

    def transcribe(
        self, audio_path: str, language: LanguageType
    ) -> Tuple[str, Optional[float], float]:

        logger.info(f"Transcribing audio with language: {language}")

        if language == LanguageType.CHINESE or language == LanguageType.ZH_TW:
            return self.transcribe_chinese(audio_path)
        elif language == LanguageType.MIN_NAN:
            return self.transcribe_min_nan(audio_path)
        else:
            raise ValueError(f"Unsupported language: {language}")



asr_service = ASRService()