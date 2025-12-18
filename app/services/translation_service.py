import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import time
import logging
import json
import os
from typing import Tuple, Optional
from ..config import settings
from ..models.schemas import LanguageType

logger = logging.getLogger(__name__)


class TranslationService:


    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Translation Service initializing on device: {self.device}")


        self.tokenizer = None
        self.model = None
        self.models_loaded = False



        self.chinese_to_minnan_dict = {



            "你好": "汝好",
            "早安": "早",
            "謝謝": "多謝",
            "再見": "再會",
            "是": "是",
            "不是": "毋是",
            "好": "好",
            "不好": "無好",
            "什麼": "啥物",
            "哪裡": "佗位",
            "怎麼": "按怎",
            "誰": "啥人",
            "為什麼": "為啥物",
            "多少": "偌濟",
            "這個": "這个",
            "那個": "彼个",
            "我": "我",
            "你": "汝",
            "他": "伊",
            "她": "伊",
            "我們": "阮",
            "你們": "恁",
            "他們": "𪜶",
            "愛": "愛",
            "喜歡": "佮意",
            "想": "想欲",
            "知道": "知影",
            "不知道": "毋知影",
            "很": "真",
            "非常": "誠",
            "一點": "一屑仔",
            "太": "傷",
            "請": "請",
            "對不起": "歹勢",
            "沒關係": "無要緊",
            "歡迎": "歡迎",
            "加油": "拍拚",
            "恭喜": "恭喜",
            "新年快樂": "新年恭喜",
            "生日快樂": "生日快樂",
            "平安": "平安",
            "健康": "健康",
            "沒問題": "無問題",
            "謝謝你": "感謝你",
            "請坐": "請坐",
            "好吃": "好食",
            "喝茶": "啉茶",
            "很高興認識你": "誠歡喜捌你",
            "不好意思": "歹勢",
            "怎麼辦": "按怎辦",
            "可以嗎": "會使無",
            "整天的": "歸剛誒",



            "爸爸": "老爸",
            "媽媽": "老母",
            "爺爺": "阿公",
            "奶奶": "阿媽",
            "哥哥": "阿兄",
            "姊姊": "阿姊",
            "弟弟": "小弟",
            "妹妹": "小妹",
            "先生": "先生",
            "太太": "太太",
            "兒子": "後生",
            "女兒": "查某囝",
            "孫子": "孫",
            "鄰居": "隔壁的",
            "老闆": "頭家",
            "老闆娘": "頭家娘",
            "客人": "人客",



            "捷運": "捷運",
            "火車": "火車",
            "公車": "公車",
            "計程車": "計程車",
            "飛機": "飛行機",
            "機車": "機車",
            "腳踏車": "孔明車",
            "車站": "車頭",
            "票": "票",
            "悠遊卡": "悠遊卡",
            "地圖": "地圖",
            "紅綠燈": "紅綠燈",
            "塞車": "塞車",
            "左邊": "倒手爿",
            "右邊": "正手爿",
            "前面": "頭前",
            "後面": "後壁",
            "裡面": "內底",
            "外面": "外口",
            "學校": "學校",
            "公司": "公司",
            "公園": "公園",
            "夜市": "夜市",



            "漂亮": "媠",
            "醜": "䆀",
            "香": "芳",
            "臭": "臭",
            "乾淨": "清氣",
            "髒": "垃圾",
            "高興": "歡喜",
            "生氣": "受氣",
            "怕": "驚",
            "熱鬧": "熱鬧",
            "安靜": "恬",
            "快": "緊",
            "慢": "慢",
            "新": "新",
            "舊": "舊",
            "真的": "有的",
            "假的": "假的",
            "簡單": "簡單",
            "困難": "困難",
            "聰明": "巧",
            "笨": "戇",
            "可愛": "古錐",
            "可憐": "可憐",
            "無聊": "無聊",



            "幫忙": "鬥相共",
            "使用": "用",
            "開始": "開始",
            "結束": "結束",
            "等": "等",
            "找": "揣",
            "拿": "提",
            "給": "予",
            "放": "放",
            "問": "問",
            "說": "講",
            "看見": "看見",
            "可以": "會使",
            "不可以": "袂使",
            "會有": "會",
            "沒有": "無",
            "要": "欲",
            "不要": "莫",
            "正在": "咧",



            "雨傘": "雨傘",
            "颱風": "風颱",
            "地震": "地動",
            "垃圾桶": "垃圾桶",
            "眼鏡": "目鏡",
            "錢包": "錢包",
            "鑰匙": "鎖匙",
            "報紙": "報紙",
            "電視": "電視",
            "冷氣": "冷氣",
            "電燈": "電火",



            "睡覺": "睏",
            "起床": "起來",
            "洗澡": "洗身軀",
            "洗臉": "洗面",
            "刷牙": "洗喙",
            "衣服": "衫",
            "褲子": "褲",
            "鞋子": "鞋仔",
            "穿衣服": "穿衫",
            "脫衣服": "脫衫",
            "講話": "講話",
            "聽": "聽",
            "看": "看",
            "走": "行",
            "跑": "走",
            "去": "去",
            "來": "來",
            "買東西": "買物件",
            "買": "買",
            "太貴": "傷貴",
            "便宜": "俗",
            "下雨": "落雨",
            "工作": "食頭路",
            "上學": "讀冊",
            "回家": "轉去",
            "房子": "厝",
            "廁所": "便所",
            "廚房": "灶跤",
            "打電話": "拍電話",



            "醫生": "醫生",
            "護士": "護士",
            "醫院": "病院",
            "生病": "破病",
            "感冒": "感冒",
            "發燒": "發燒",
            "頭痛": "頭殼痛",
            "肚子痛": "腹肚痛",
            "咳嗽": "嗽",
            "藥": "藥仔",
            "吃藥": "食藥",
            "打針": "拍針",
            "痛": "痛",
            "癢": "癢",
            "累": "忝",
            "身體": "身軀",
            "手": "手",
            "腳": "跤",
            "眼睛": "目睭",
            "耳朵": "耳仔",
            "嘴巴": "喙",
            "脖子": "頷頸",
            "背": "尻脊骿",
            "流血": "流血",
            "血壓": "血壓",
            "慢慢來": "寬寬仔來",



            "水": "水",
            "茶": "茶",
            "飯": "飯",
            "麵": "麵",
            "菜": "菜",
            "肉": "肉",
            "魚": "魚",
            "蛋": "卵",
            "水果": "果子",
            "冰": "冰",
            "熱": "燒",
            "冷": "冷",
            "甜": "甜",
            "鹹": "鹹",
            "酸": "酸",
            "苦": "苦",
            "辣": "辣",
            "筷子": "箸",
            "湯匙": "湯匙",
            "碗": "碗",
            "飽": "飽",
            "餓": "枵",
            "吃飯": "食飯",



            "今天": "今仔日",
            "明天": "明仔載",
            "昨天": "昨昏",
            "現在": "這馬",
            "早上": "早起",
            "中午": "中晝",
            "下午": "下晡",
            "晚上": "暗時",
            "以前": "進前",
            "以後": "以後",
            "剛才": "頭先",
            "等一下": "等咧",
            "大": "大",
            "小": "細",
            "男": "查埔",
            "女": "查某",
            "小孩": "囡仔",
            "大人": "大人",
            "老人": "老歲仔",
            "老師": "老師",
            "學生": "學生",
            "朋友": "朋友",
            "錢": "錢",
            "車": "車",
            "書": "冊",
            "多少錢": "偌濟錢",
        }


        self.minnan_to_chinese_dict = {
            v: k for k, v in self.chinese_to_minnan_dict.items()
        }


        self.slang_dict = {}
        try:
            slang_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "slang_dict.json"
            )
            if os.path.exists(slang_path):
                with open(slang_path, "r", encoding="utf-8") as f:
                    self.slang_dict = json.load(f)
                logger.info(
                    f"Loaded slang dictionary with {len(self.slang_dict)} entries"
                )
            else:
                logger.warning(f"Slang dictionary not found at {slang_path}")
        except Exception as e:
            logger.error(f"Error loading slang dictionary: {e}")

    def load_models(self):

        try:
            logger.info("Loading Translation models...")




            model_name = "facebook/m2m100_418M"

            logger.info(f"Loading translation model: {model_name}")
            self.tokenizer = M2M100Tokenizer.from_pretrained(
                model_name, cache_dir=settings.MODEL_CACHE_DIR
            )
            self.model = M2M100ForConditionalGeneration.from_pretrained(
                model_name, cache_dir=settings.MODEL_CACHE_DIR
            )
            self.model.to(self.device)
            self.model.eval()

            self.models_loaded = True
            logger.info("Translation models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading translation models: {str(e)}")
            logger.warning("Will use dictionary-based translation as fallback")

            self.models_loaded = False

    def dictionary_translate(
        self, text: str, source_language: LanguageType, target_language: LanguageType
    ) -> str:

        if (
            source_language == LanguageType.CHINESE
            and target_language == LanguageType.MIN_NAN
        ):

            if text in self.chinese_to_minnan_dict:
                return self.chinese_to_minnan_dict[text]


            translated_words = []
            words = list(text)
            for word in words:
                if word in self.chinese_to_minnan_dict:
                    translated_words.append(self.chinese_to_minnan_dict[word])
                else:
                    translated_words.append(word)
            return "".join(translated_words)

        elif (
            source_language == LanguageType.MIN_NAN
            and target_language == LanguageType.CHINESE
        ):

            if text in self.minnan_to_chinese_dict:
                return self.minnan_to_chinese_dict[text]


            translated_words = []
            words = list(text)
            for word in words:
                if word in self.minnan_to_chinese_dict:
                    translated_words.append(self.minnan_to_chinese_dict[word])
                else:
                    translated_words.append(word)
            return "".join(translated_words)


        return text

    def neural_translate(
        self, text: str, source_language: LanguageType, target_language: LanguageType
    ) -> str:

        if not self.models_loaded:
            logger.warning("NMT models not loaded, using dictionary translation")
            return self.dictionary_translate(text, source_language, target_language)

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
                    max_length=512,
                )


            translated = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]

            return translated

        except Exception as e:
            logger.error(f"Error in neural translation: {str(e)}")
            logger.warning("Falling back to dictionary translation")
            return self.dictionary_translate(text, source_language, target_language)

    def translate(
        self,
        text: str,
        source_language: LanguageType,
        target_language: LanguageType,
        use_neural: bool = True,
    ) -> Tuple[str, float]:

        start_time = time.time()

        logger.info(f"Translating from {source_language} to {target_language}")


        if source_language == target_language:
            return text, 0.0


        if (
            source_language == LanguageType.CHINESE
            and target_language == LanguageType.MIN_NAN
            and text in self.slang_dict
        ):
            return self.slang_dict[text], 0.0


        dict_translation = self.dictionary_translate(
            text, source_language, target_language
        )


        if dict_translation != text and not use_neural:
            processing_time = time.time() - start_time
            return dict_translation, processing_time


        if use_neural and self.models_loaded:
            translated_text = self.neural_translate(
                text, source_language, target_language
            )
        else:
            translated_text = dict_translation



        if (
            source_language == LanguageType.CHINESE
            and target_language == LanguageType.MIN_NAN
        ):

            for chinese, minnan in self.chinese_to_minnan_dict.items():
                if chinese in translated_text:
                    translated_text = translated_text.replace(chinese, minnan)

        processing_time = time.time() - start_time

        logger.info(
            f"Translation completed in {processing_time:.2f}s: "
            f"'{text}' -> '{translated_text}'"
        )

        return translated_text, processing_time

    def translate_with_details(
        self,
        text: str,
        source_language: LanguageType,
        target_language: LanguageType,
        use_neural: bool = True,
    ) -> dict:

        start_time = time.time()


        if (
            source_language == LanguageType.CHINESE
            and target_language == LanguageType.MIN_NAN
            and text in self.slang_dict
        ):
            processing_time = time.time() - start_time
            return {
                "translated_text": self.slang_dict[text],
                "method": "semantic_slang",
                "confidence": "high",
                "processing_time": processing_time,
            }


        dict_translation = self.dictionary_translate(
            text, source_language, target_language
        )


        if dict_translation != text:
            processing_time = time.time() - start_time
            return {
                "translated_text": dict_translation,
                "method": "dictionary",
                "confidence": "high",
                "processing_time": processing_time,
            }

        processing_time = time.time() - start_time
        return {
            "translated_text": text,
            "method": "fallback",
            "confidence": "low",
            "processing_time": processing_time,
        }



translation_service = TranslationService()