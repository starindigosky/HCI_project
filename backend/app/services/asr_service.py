import torch
import librosa
import numpy as np
import webrtcvad
import gc
import os
from transformers import pipeline
import time
import logging
from typing import Tuple, Optional
from ..config import settings
from ..models.schemas import LanguageType

logger = logging.getLogger(__name__)

class ASRService:
    """Service for Automatic Speech Recognition (ASR) for Chinese and Min Nan"""

    def __init__(self):
        # 檢測 GPU，但我們會策略性地只讓部分模型使用它
        self.has_gpu = torch.cuda.is_available()
        self.device_str = "cuda" if self.has_gpu else "cpu"
        logger.info(f"ASR Service initializing. System has GPU: {self.has_gpu}")

        # Pipelines
        self.chinese_asr_pipeline = None
        self.min_nan_pipeline = None # 改用 pipeline

        # VAD
        self.vad = webrtcvad.Vad(1)
        self.models_loaded = False

    def _clear_gpu_memory(self):
        """Utility to clear GPU memory"""
        if self.has_gpu:
            gc.collect()
            torch.cuda.empty_cache()

    def _contains_speech(self, audio_array: np.ndarray, sample_rate: int) -> bool:
        if sample_rate not in [8000, 16000, 32000, 48000]:
            return True
        
        # 簡易 VAD 檢查
        try:
            frame_duration_ms = 30
            frame_size = int(sample_rate * frame_duration_ms / 1000)
            pcm_audio = (audio_array * 32767).astype(np.int16)
            pcm_bytes = pcm_audio.tobytes()
            num_frames = len(pcm_bytes) // (2 * frame_size)
            
            for i in range(num_frames):
                start = i * 2 * frame_size
                end = start + 2 * frame_size
                if self.vad.is_speech(pcm_bytes[start:end], sample_rate):
                    return True
        except Exception:
            return True # 如果 VAD 出錯，預設為有聲音
            
        return False

    def load_chinese_model(self):
        """Load Chinese model (Whisper) on GPU if available"""
        if self.chinese_asr_pipeline is not None:
            return

        logger.info(f"Loading Chinese ASR: {settings.CHINESE_ASR_MODEL}")
        
        # 釋放閩南語模型 (如果是 GPU 模式)
        if self.min_nan_pipeline is not None:
            del self.min_nan_pipeline
            self.min_nan_pipeline = None
            self._clear_gpu_memory()

        # Whisper Small 在 4GB 顯卡上可以跑 GPU，使用 float16 加速
        self.chinese_asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=settings.CHINESE_ASR_MODEL,
            chunk_length_s=30,
            device=0 if self.has_gpu else -1, # 0 = GPU, -1 = CPU
            torch_dtype=torch.float16 if self.has_gpu else torch.float32
        )

    def load_min_nan_model(self):
        """Load Min Nan model - FORCE CPU to avoid OOM"""
        if self.min_nan_pipeline is not None:
            return

        logger.info(f"Loading Min Nan ASR: {settings.MIN_NAN_ASR_MODEL}")
        
        # 釋放 Whisper 記憶體
        if self.chinese_asr_pipeline is not None:
            del self.chinese_asr_pipeline
            self.chinese_asr_pipeline = None
            self._clear_gpu_memory()

        try:
            # 關鍵修改：
            # 1. 使用 pipeline 簡化載入
            # 2. device=-1 強制使用 CPU。因為 Large 模型在 4GB VRAM 跑長音訊必死無疑。
            # 3. chunk_length_s=10 自動將長音訊切成 10 秒片段處理，防止記憶體暴增。
            self.min_nan_pipeline = pipeline(
                "automatic-speech-recognition",
                model=settings.MIN_NAN_ASR_MODEL,
                chunk_length_s=10,
                device=-1, # 強制 CPU
            )
            logger.info("Min Nan model loaded successfully on CPU.")
            
        except Exception as e:
            logger.error(f"Failed to load Min Nan model: {e}")
            raise

    def transcribe_chinese(self, audio_path: str) -> Tuple[str, Optional[float], float]:
        start_time = time.time()
        self.load_chinese_model()
        
        try:
            # Whisper pipeline handles loading automatically
            result = self.chinese_asr_pipeline(
                audio_path,
                generate_kwargs={"language": "zh", "task": "transcribe"}
            )
            text = result["text"] if isinstance(result, dict) else result[0]["text"]
            return text, None, time.time() - start_time
        except Exception as e:
            logger.error(f"Error in Chinese ASR: {e}")
            raise

    def transcribe_min_nan(self, audio_path: str) -> Tuple[str, Optional[float], float]:
        start_time = time.time()
        self.load_min_nan_model()

        try:
            # 直接使用 pipeline 進行辨識，它會自動處理取樣率和切分
            result = self.min_nan_pipeline(audio_path)
            
            # pipeline 回傳格式通常是 {'text': '...'}
            text = result["text"] if isinstance(result, dict) else result[0]["text"]
            
            # 簡單的後處理：如果是繁體模型，通常不需要轉換，但有時候會輸出空格
            text = text.replace(" ", "")
            
            logger.info(f"Min Nan transcription: {text[:50]}...")
            return text, None, time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error in Min Nan ASR: {e}")
            # 發生錯誤時嘗試回傳空字串而不是讓整個 Server 崩潰
            return "", 0.0, 0.0

    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int, language: LanguageType):
        """Transcribe from numpy array (for streaming)"""
        start_time = time.time()
        
        # 對於閩南語，因為我們現在用 pipeline 且強制 CPU，
        # 為了避免格式問題，最穩健的方法是存成暫存檔
        if language == LanguageType.MIN_NAN:
            import tempfile
            import soundfile as sf
            
            # 轉存暫存檔
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                sf.write(tmp_path, audio_array, sample_rate)
                text, conf, _ = self.transcribe_min_nan(tmp_path)
                return text, conf, time.time() - start_time
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # 中文 (Whisper) 繼續使用原本的高效方式
        self.load_chinese_model()
        
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

        if not self._contains_speech(audio_array, 16000):
            return "", None, time.time() - start_time

        result = self.chinese_asr_pipeline(
            {"array": audio_array, "sampling_rate": 16000},
            generate_kwargs={"language": "zh", "task": "transcribe"}
        )
        text = result["text"] if isinstance(result, dict) else result[0]["text"]
        return text, None, time.time() - start_time

    def transcribe(self, audio_path: str, language: LanguageType):
        if language == LanguageType.MIN_NAN:
            return self.transcribe_min_nan(audio_path)
        return self.transcribe_chinese(audio_path)

    def load_models(self):
        pass 

asr_service = ASRService()