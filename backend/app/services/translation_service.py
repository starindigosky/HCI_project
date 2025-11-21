import torch
import pypinyin
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import time
import logging
import gc
from typing import Tuple, Optional
from ..config import settings
from ..models.schemas import LanguageType

logger = logging.getLogger(__name__)

class TranslationService:
    """
    Service for translating between Chinese and Min Nan languages
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Translation Service initializing on device: {self.device}")

        self.tokenizer = None
        self.model = None
        self.models_loaded = False

        # 簡單的字典對應
        self.chinese_to_minnan_dict = {
            "你好": "汝好", "早安": "早", "謝謝": "多謝", "再見": "再會",
            "是": "是", "不是": "毋是", "好": "好", "不好": "無好",
            "吃飯": "食飯", "喝水": "飲水", "什麼": "啥物", "哪裡": "佗位",
            "怎麼": "按怎", "誰": "啥人", "為什麼": "為啥物", "多少": "偌濟",
            "今天": "今仔日", "明天": "明仔載", "昨天": "昨昏", "現在": "這馬",
            "這個": "這个", "那個": "彼个", "我": "我", "你": "汝",
            "他": "伊", "她": "伊", "我們": "阮", "你們": "恁",
            "他們": "𪜶", "愛": "愛", "喜歡": "佮意", "想": "想欲",
            "知道": "知影", "不知道": "毋知", "很": "真", "非常": "誠",
            "一點": "一屑仔", "太": "傷", "請": "請", "對不起": "歹勢",
            "沒關係": "無要緊", "歡迎": "歡迎", "加油": "拍拚",
            "恭喜": "恭喜", "新年快樂": "新年恭喜", "生日快樂": "生日快樂",
            "平安": "平安", "健康": "健康",
        }
        self.minnan_to_chinese_dict = {v: k for k, v in self.chinese_to_minnan_dict.items()}

    def _clear_gpu_memory(self):
        """釋放 GPU 記憶體"""
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def load_models(self):
        """Load NMT models with FP16 optimization"""
        try:
            if self.models_loaded:
                return

            logger.info("Loading Translation models...")
            model_name = "facebook/m2m100_418M"

            logger.info(f"Loading translation model: {model_name} (FP16 enabled)")
            
            self.tokenizer = M2M100Tokenizer.from_pretrained(
                model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            
            self.model = M2M100ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=settings.MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.model.to(self.device)
            self.model.eval()

            self.models_loaded = True
            logger.info("Translation models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading translation models: {str(e)}")
            self.models_loaded = False

    def dictionary_translate(self, text, source_language, target_language):
        if source_language == LanguageType.CHINESE and target_language == LanguageType.MIN_NAN:
            # 直接查字典，如果沒有就回傳原字
            translated_chars = []
            # 簡單的長詞優先匹配邏輯可以更複雜，這裡先做簡單的字元/詞彙替換
            # 為了效率，這裡簡化處理，若要精確需要最大正向匹配法
            for k, v in self.chinese_to_minnan_dict.items():
                if k in text:
                    text = text.replace(k, v)
            return text
        elif source_language == LanguageType.MIN_NAN and target_language == LanguageType.CHINESE:
            for k, v in self.minnan_to_chinese_dict.items():
                if k in text:
                    text = text.replace(k, v)
            return text
        return text

    def neural_translate(self, text: str, source_language: LanguageType, target_language: LanguageType) -> str:
        if not self.models_loaded:
            self.load_models()

        try:
            lang_code_map = {
                LanguageType.CHINESE: "zh",
                LanguageType.ZH_TW: "zh",
                LanguageType.MIN_NAN: "zh", 
            }

            src_lang = lang_code_map.get(source_language, "zh")
            tgt_lang = lang_code_map.get(target_language, "zh")

            self.tokenizer.src_lang = src_lang

            encoded = self.tokenizer(text, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
                    max_length=512
                )

            translated = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translated

        except Exception as e:
            logger.error(f"Neural translation failed: {e}")
            return self.dictionary_translate(text, source_language, target_language)

    def translate(self, text: str, source_language: LanguageType, target_language: LanguageType, use_neural: bool = True) -> Tuple[str, float]:
        start_time = time.time()
        
        # 1. 決定翻譯後的文字 (Hanzi)
        dict_trans = self.dictionary_translate(text, source_language, target_language)
        
        if dict_trans != text or not use_neural:
             # 如果字典有對應到，或者不強制用 AI，就用字典結果
             final_text = dict_trans
        else:
             # 否則用 AI 翻譯
             final_text = self.neural_translate(text, source_language, target_language)
        
        # 2. [關鍵修正] 針對閩南語 TTS 的後處理 (MUST DO)
        # 這裡一定要執行，不能放在 else 裡面
        if target_language == LanguageType.MIN_NAN:
             # 再次確保字典詞彙替換 (針對 AI 翻譯結果)
             for zh, mn in self.chinese_to_minnan_dict.items():
                 if zh in final_text:
                     final_text = final_text.replace(zh, mn)
             
             # [最重要] 強制轉拼音 (Romanization)
             # 這是為了防止 TTS 模型因為看不懂漢字而崩潰
             # 注意：這裡是用普通話拼音模擬閩南語發音，雖然不完美，但能讓模型發出聲音且不崩潰
             final_text = " ".join(pypinyin.lazy_pinyin(final_text))
             
             logger.info(f"Converted to Pinyin for Min Nan TTS: {final_text}")

        return final_text, time.time() - start_time

# Global instance
translation_service = TranslationService()