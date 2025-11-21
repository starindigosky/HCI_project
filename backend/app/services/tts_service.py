import torch
import numpy as np
from transformers import (
    VitsModel, VitsTokenizer, 
    AutoProcessor, BarkModel # 新增 Bark 支援
)
import time
import logging
import os
import uuid
import gc
from scipy.io import wavfile
from typing import Tuple
from ..config import settings
from ..models.schemas import LanguageType

logger = logging.getLogger(__name__)

class TTSService:
    """Service for Text-to-Speech (TTS)"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TTS Service initializing on device: {self.device}")

        # Min Nan (MMS/VITS)
        self.min_nan_tokenizer = None
        self.min_nan_model = None
        
        # Chinese (Bark)
        self.chinese_processor = None
        self.chinese_model = None
        
        self.models_loaded = False

    def _clear_gpu_memory(self):
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def load_min_nan_model(self):
        """載入閩南語模型 (VITS 架構)"""
        if self.min_nan_model is not None:
            return
        
        # 卸載中文模型以節省記憶體
        if self.chinese_model is not None:
            del self.chinese_model
            del self.chinese_processor
            self.chinese_model = None
            self.chinese_processor = None
            self._clear_gpu_memory()

        logger.info(f"Loading Min Nan TTS: {settings.MIN_NAN_TTS_MODEL}")
        self.min_nan_tokenizer = VitsTokenizer.from_pretrained(settings.MIN_NAN_TTS_MODEL, cache_dir=settings.MODEL_CACHE_DIR)
        self.min_nan_model = VitsModel.from_pretrained(settings.MIN_NAN_TTS_MODEL, cache_dir=settings.MODEL_CACHE_DIR)
        self.min_nan_model.to(self.device)
        self.min_nan_model.eval()

    def load_chinese_model(self):
        """載入中文模型 (Bark 架構)"""
        if self.chinese_model is not None:
            return

        # 卸載閩南語模型
        if self.min_nan_model is not None:
            del self.min_nan_model
            del self.min_nan_tokenizer
            self.min_nan_model = None
            self.min_nan_tokenizer = None
            self._clear_gpu_memory()

        logger.info(f"Loading Chinese TTS: {settings.CHINESE_TTS_MODEL}")
        # Bark 使用 AutoProcessor
        self.chinese_processor = AutoProcessor.from_pretrained(settings.CHINESE_TTS_MODEL, cache_dir=settings.MODEL_CACHE_DIR)
        self.chinese_model = BarkModel.from_pretrained(settings.CHINESE_TTS_MODEL, cache_dir=settings.MODEL_CACHE_DIR)
        
        # Bark 在 4GB 顯卡上建議使用 float16 (如果支援)
        if self.device == "cuda":
            self.chinese_model = self.chinese_model.to(dtype=torch.float16)
        
        self.chinese_model.to(self.device)
        self.chinese_model.eval()

    def text_to_speech(
        self,
        text: str,
        language: LanguageType = LanguageType.MIN_NAN,
        output_filename: str = None
    ) -> Tuple[str, float]:
        
        start_time = time.time()

        try:
            waveform = None
            sample_rate = 16000 # 預設

            # --- 閩南語 (MMS/VITS) ---
            if language == LanguageType.MIN_NAN:
                self.load_min_nan_model()
                logger.info(f"TTS (Min Nan): {text[:30]}...")

                inputs = self.min_nan_tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if 'input_ids' in inputs:
                    inputs['input_ids'] = inputs['input_ids'].to(torch.long)

                with torch.no_grad():
                    outputs = self.min_nan_model(**inputs)
                    waveform = outputs.waveform[0].cpu().numpy()
                
                sample_rate = 16000 # MMS 預設

            # --- 中文 (Bark) ---
            elif language in [LanguageType.CHINESE, LanguageType.ZH_TW]:
                self.load_chinese_model()
                logger.info(f"TTS (Chinese Bark): {text[:30]}...")
                
                # Bark 需要特殊的 prompt 來指定語言，這裡讓它自動或使用預設
                # 為了讓 Bark 講中文，我們可以使用 "v2/zh_speaker_0" 等預設 (如果有的話)，
                # 但最簡單的是直接丟中文進去，Bark 會嘗試生成
                
                inputs = self.chinese_processor(text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    # Bark 生成音訊
                    audio_array = self.chinese_model.generate(**inputs)
                    waveform = audio_array.cpu().numpy().squeeze()
                
                sample_rate = self.chinese_model.generation_config.sample_rate # Bark 通常是 24000

            else:
                raise ValueError(f"Unsupported TTS language: {language}")

            # 儲存音檔
            if output_filename is None:
                output_filename = f"tts_{uuid.uuid4().hex}.wav"

            output_path = os.path.join(settings.OUTPUT_DIR, output_filename)
            
            # 正規化並轉存 int16
            if waveform is not None:
                # 簡單的正規化防止爆音
                waveform = waveform / np.max(np.abs(waveform))
                waveform_int16 = np.int16(waveform * 32767)
                wavfile.write(output_path, sample_rate, waveform_int16)
            
            return output_path, time.time() - start_time

        except Exception as e:
            logger.error(f"Error in TTS: {str(e)}", exc_info=True)
            raise

    def translate_and_speak(
        self,
        text: str,
        source_language: LanguageType,
        target_language: LanguageType,
        output_filename: str = None,
        use_neural_translation: bool = True
    ) -> Tuple[str, str, float]:
        
        start_time = time.time()
        from .translation_service import translation_service

        # 1. 翻譯
        if source_language != target_language:
            translated_text, _ = translation_service.translate(
                text, source_language, target_language, use_neural=use_neural_translation
            )
        else:
            translated_text = text

        # 2. 合成
        output_path, tts_time = self.text_to_speech(
            translated_text,
            language=target_language,
            output_filename=output_filename
        )

        return translated_text, output_path, time.time() - start_time

    def load_models(self):
        pass

tts_service = TTSService()