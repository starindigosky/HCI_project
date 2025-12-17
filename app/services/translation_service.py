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
    """
    Service for translating between Chinese and Min Nan languages
    Uses Neural Machine Translation (NMT) with Facebook's M2M-100 model
    Falls back to dictionary-based translation for common phrases
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Translation Service initializing on device: {self.device}")

        # Initialize models as None - lazy loading
        self.tokenizer = None
        self.model = None
        self.models_loaded = False

        # Dictionary-based translation fallback for common phrases
        # Min Nan romanization (POJ - Pe̍h-ōe-jī) to Chinese mapping
        self.chinese_to_minnan_dict = {
            # ==========================================
            # 0. 基本詞彙 (Basic)
            # ==========================================
            "你好": "汝好",  # Hello
            "早安": "早",  # Good morning
            "謝謝": "多謝",  # Thank you
            "再見": "再會",  # Goodbye
            "是": "是",  # Yes
            "不是": "毋是",  # No
            "好": "好",  # Good/OK
            "不好": "無好",  # Not good
            "什麼": "啥物",  # What
            "哪裡": "佗位",  # Where
            "怎麼": "按怎",  # How
            "誰": "啥人",  # Who
            "為什麼": "為啥物",  # Why
            "多少": "偌濟",  # How much/many
            "這個": "這个",  # This
            "那個": "彼个",  # That
            "我": "我",  # I/me
            "你": "汝",  # You
            "他": "伊",  # He/him
            "她": "伊",  # She/her
            "我們": "阮",  # We/us
            "你們": "恁",  # You (plural)
            "他們": "𪜶",  # They/them
            "愛": "愛",  # Love
            "喜歡": "佮意",  # Like
            "想": "想欲",  # Want/think
            "知道": "知影",  # Know
            "不知道": "毋知影",  # Don't know
            "很": "真",  # Very
            "非常": "誠",  # Very/extremely
            "一點": "一屑仔",  # A little
            "太": "傷",  # Too (much)
            "請": "請",  # Please
            "對不起": "歹勢",  # Sorry
            "沒關係": "無要緊",  # It's okay
            "歡迎": "歡迎",  # Welcome
            "加油": "拍拚",  # Keep it up/add oil
            "恭喜": "恭喜",  # Congratulations
            "新年快樂": "新年恭喜",  # Happy New Year
            "生日快樂": "生日快樂",  # Happy Birthday
            "平安": "平安",  # Peace/safe
            "健康": "健康",  # Health/healthy
            "沒問題": "無問題",  # No problem
            "謝謝你": "感謝你",  # Thank you (more formal)
            "請坐": "請坐",  # Please sit
            "好吃": "好食",  # Delicious
            "喝茶": "啉茶",  # Drink tea
            "很高興認識你": "誠歡喜捌你",  # Nice to meet you
            "不好意思": "歹勢",  # Excuse me/Sorry
            "怎麼辦": "按怎辦",  # What to do
            "可以嗎": "會使無",  # Is it okay?
            "整天的": "歸剛誒",  # Every day
            # ==========================================
            # 1. 家庭與稱謂 (Family & People)
            # ==========================================
            "爸爸": "老爸",  # Father
            "媽媽": "老母",  # Mother
            "爺爺": "阿公",  # Grandfather
            "奶奶": "阿媽",  # Grandmother
            "哥哥": "阿兄",  # Older brother
            "姊姊": "阿姊",  # Older sister
            "弟弟": "小弟",  # Younger brother
            "妹妹": "小妹",  # Younger sister
            "先生": "先生",  # Mr. / Husband
            "太太": "太太",  # Mrs. / Wife
            "兒子": "後生",  # Son
            "女兒": "查某囝",  # Daughter
            "孫子": "孫",  # Grandchild
            "鄰居": "隔壁的",  # Neighbor
            "老闆": "頭家",  # Boss
            "老闆娘": "頭家娘",  # Boss's wife
            "客人": "人客",  # Guest/Customer
            # ==========================================
            # 2. 交通與位置 (Transport & Location)
            # ==========================================
            "捷運": "捷運",  # MRT
            "火車": "火車",  # Train
            "公車": "公車",  # Bus
            "計程車": "計程車",  # Taxi
            "飛機": "飛行機",  # Airplane
            "機車": "機車",  # Scooter
            "腳踏車": "孔明車",  # Bicycle
            "車站": "車頭",  # Station
            "票": "票",  # Ticket
            "悠遊卡": "悠遊卡",  # EasyCard
            "地圖": "地圖",  # Map
            "紅綠燈": "紅綠燈",  # Traffic light
            "塞車": "塞車",  # Traffic jam
            "左邊": "倒手爿",  # Left side
            "右邊": "正手爿",  # Right side
            "前面": "頭前",  # Front
            "後面": "後壁",  # Back/Behind
            "裡面": "內底",  # Inside
            "外面": "外口",  # Outside
            "學校": "學校",  # School
            "公司": "公司",  # Company
            "公園": "公園",  # Park
            "夜市": "夜市",  # Night market
            # ==========================================
            # 3. 日常形容詞與感覺 (Adjectives & Feelings)
            # ==========================================
            "漂亮": "媠",  # Beautiful
            "醜": "䆀",  # Ugly
            "香": "芳",  # Fragrant
            "臭": "臭",  # Stinky
            "乾淨": "清氣",  # Clean
            "髒": "垃圾",  # Dirty
            "高興": "歡喜",  # Happy
            "生氣": "受氣",  # Angry
            "怕": "驚",  # Scared
            "熱鬧": "熱鬧",  # Bustling
            "安靜": "恬",  # Quiet
            "快": "緊",  # Fast
            "慢": "慢",  # Slow
            "新": "新",  # New
            "舊": "舊",  # Old
            "真的": "有的",  # Real/True
            "假的": "假的",  # Fake
            "簡單": "簡單",  # Simple
            "困難": "困難",  # Difficult
            "聰明": "巧",  # Smart
            "笨": "戇",  # Stupid
            "可愛": "古錐",  # Cute
            "可憐": "可憐",  # Pitiful
            "無聊": "無聊",  # Boring
            # ==========================================
            # 4. 常見動詞與助詞 (Verbs & Particles)
            # ==========================================
            "幫忙": "鬥相共",  # Help
            "使用": "用",  # Use
            "開始": "開始",  # Start
            "結束": "結束",  # End
            "等": "等",  # Wait
            "找": "揣",  # Find
            "拿": "提",  # Take
            "給": "予",  # Give
            "放": "放",  # Put
            "問": "問",  # Ask
            "說": "講",  # Say
            "看見": "看見",  # See
            "可以": "會使",  # Can
            "不可以": "袂使",  # Cannot
            "會有": "會",  # Will have
            "沒有": "無",  # Don't have
            "要": "欲",  # Want
            "不要": "莫",  # Don't
            "正在": "咧",  # -ing
            # ==========================================
            # 5. 北部天氣與生活雜項 (Weather & Misc)
            # ==========================================
            "雨傘": "雨傘",  # Umbrella
            "颱風": "風颱",  # Typhoon
            "地震": "地動",  # Earthquake
            "垃圾桶": "垃圾桶",  # Trash can
            "眼鏡": "目鏡",  # Glasses
            "錢包": "錢包",  # Wallet
            "鑰匙": "鎖匙",  # Key
            "報紙": "報紙",  # Newspaper
            "電視": "電視",  # TV
            "冷氣": "冷氣",  # Air conditioner
            "電燈": "電火",  # Light
            # ==========================================
            # 6. 日常生活與動作 (Daily Life & Actions)
            # ==========================================
            "睡覺": "睏",  # Sleep
            "起床": "起來",  # Wake up
            "洗澡": "洗身軀",  # Shower
            "洗臉": "洗面",  # Wash face
            "刷牙": "洗喙",  # Brush teeth
            "衣服": "衫",  # Clothes
            "褲子": "褲",  # Pants
            "鞋子": "鞋仔",  # Shoes
            "穿衣服": "穿衫",  # Dress up
            "脫衣服": "脫衫",  # Undress
            "講話": "講話",  # Speak
            "聽": "聽",  # Listen
            "看": "看",  # Look
            "走": "行",  # Walk
            "跑": "走",  # Run
            "去": "去",  # Go
            "來": "來",  # Come
            "買東西": "買物件",  # Shopping
            "買": "買",  # Buy
            "太貴": "傷貴",  # Too expensive
            "便宜": "俗",  # Cheap
            "下雨": "落雨",  # Rain
            "工作": "食頭路",  # Work
            "上學": "讀冊",  # School
            "回家": "轉去",  # Go home
            "房子": "厝",  # House
            "廁所": "便所",  # Toilet
            "廚房": "灶跤",  # Kitchen
            "打電話": "拍電話",  # Phone call
            # ==========================================
            # 7. 醫療與身體 (Medical & Body)
            # ==========================================
            "醫生": "醫生",  # Doctor
            "護士": "護士",  # Nurse
            "醫院": "病院",  # Hospital
            "生病": "破病",  # Sick
            "感冒": "感冒",  # Cold
            "發燒": "發燒",  # Fever
            "頭痛": "頭殼痛",  # Headache
            "肚子痛": "腹肚痛",  # Stomachache
            "咳嗽": "嗽",  # Cough
            "藥": "藥仔",  # Medicine
            "吃藥": "食藥",  # Take medicine
            "打針": "拍針",  # Injection
            "痛": "痛",  # Pain
            "癢": "癢",  # Itchy
            "累": "忝",  # Tired
            "身體": "身軀",  # Body
            "手": "手",  # Hand
            "腳": "跤",  # Foot/Leg
            "眼睛": "目睭",  # Eye
            "耳朵": "耳仔",  # Ear
            "嘴巴": "喙",  # Mouth
            "脖子": "頷頸",  # Neck
            "背": "尻脊骿",  # Back
            "流血": "流血",  # Bleeding
            "血壓": "血壓",  # Blood pressure
            "慢慢來": "寬寬仔來",  # Take it easy
            # ==========================================
            # 8. 食物與味道 (Food & Taste)
            # ==========================================
            "水": "水",  # Water
            "茶": "茶",  # Tea
            "飯": "飯",  # Rice
            "麵": "麵",  # Noodles
            "菜": "菜",  # Vegetables
            "肉": "肉",  # Meat
            "魚": "魚",  # Fish
            "蛋": "卵",  # Egg
            "水果": "果子",  # Fruit
            "冰": "冰",  # Ice
            "熱": "燒",  # Hot
            "冷": "冷",  # Cold
            "甜": "甜",  # Sweet
            "鹹": "鹹",  # Salty
            "酸": "酸",  # Sour
            "苦": "苦",  # Bitter
            "辣": "辣",  # Spicy
            "筷子": "箸",  # Chopsticks
            "湯匙": "湯匙",  # Spoon
            "碗": "碗",  # Bowl
            "飽": "飽",  # Full
            "餓": "枵",  # Hungry
            "吃飯": "食飯",  # Eat
            # ==========================================
            # 9. 時間與其他 (Time & Others)
            # ==========================================
            "今天": "今仔日",  # Today
            "明天": "明仔載",  # Tomorrow
            "昨天": "昨昏",  # Yesterday
            "現在": "這馬",  # Now
            "早上": "早起",  # Morning
            "中午": "中晝",  # Noon
            "下午": "下晡",  # Afternoon
            "晚上": "暗時",  # Night
            "以前": "進前",  # Before
            "以後": "以後",  # After
            "剛才": "頭先",  # Just now
            "等一下": "等咧",  # Wait a moment
            "大": "大",  # Big
            "小": "細",  # Small
            "男": "查埔",  # Male
            "女": "查某",  # Female
            "小孩": "囡仔",  # Child
            "大人": "大人",  # Adult
            "老人": "老歲仔",  # Elderly
            "老師": "老師",  # Teacher
            "學生": "學生",  # Student
            "朋友": "朋友",  # Friend
            "錢": "錢",  # Money
            "車": "車",  # Car
            "書": "冊",  # Book
            "多少錢": "偌濟錢",  # How much money
        }

        # Reverse dictionary for Min Nan to Chinese
        self.minnan_to_chinese_dict = {
            v: k for k, v in self.chinese_to_minnan_dict.items()
        }

        # Load Slang Dictionary (Priority 1)
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
        """Load NMT models (lazy loading)"""
        try:
            logger.info("Loading Translation models...")

            # Load M2M-100 model for multilingual translation
            # M2M-100 supports 100 languages including Chinese
            # We'll use it as a base and fine-tune or adapt for Min Nan
            model_name = "facebook/m2m100_418M"  # Smaller model for faster inference

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
            # Don't raise - allow dictionary-based translation
            self.models_loaded = False

    def dictionary_translate(
        self, text: str, source_language: LanguageType, target_language: LanguageType
    ) -> str:
        """
        Dictionary-based translation for common phrases

        Args:
            text: Input text
            source_language: Source language
            target_language: Target language

        Returns:
            Translated text
        """
        if (
            source_language == LanguageType.CHINESE
            and target_language == LanguageType.MIN_NAN
        ):
            # Try exact match first
            if text in self.chinese_to_minnan_dict:
                return self.chinese_to_minnan_dict[text]

            # Try word-by-word translation
            translated_words = []
            words = list(text)  # Split into characters for Chinese
            for word in words:
                if word in self.chinese_to_minnan_dict:
                    translated_words.append(self.chinese_to_minnan_dict[word])
                else:
                    translated_words.append(word)  # Keep original if no translation
            return "".join(translated_words)

        elif (
            source_language == LanguageType.MIN_NAN
            and target_language == LanguageType.CHINESE
        ):
            # Try exact match first
            if text in self.minnan_to_chinese_dict:
                return self.minnan_to_chinese_dict[text]

            # Try word-by-word translation
            translated_words = []
            words = list(text)
            for word in words:
                if word in self.minnan_to_chinese_dict:
                    translated_words.append(self.minnan_to_chinese_dict[word])
                else:
                    translated_words.append(word)
            return "".join(translated_words)

        # No translation needed if same language
        return text

    def neural_translate(
        self, text: str, source_language: LanguageType, target_language: LanguageType
    ) -> str:
        """
        Neural Machine Translation using M2M-100

        Args:
            text: Input text
            source_language: Source language
            target_language: Target language

        Returns:
            Translated text
        """
        if not self.models_loaded:
            logger.warning("NMT models not loaded, using dictionary translation")
            return self.dictionary_translate(text, source_language, target_language)

        try:
            # Map our language types to M2M-100 language codes
            lang_code_map = {
                LanguageType.CHINESE: "zh",
                LanguageType.ZH_TW: "zh",
                # Note: M2M-100 doesn't have Min Nan, we'll use Chinese as approximation
                # In production, you would fine-tune the model for Min Nan
                LanguageType.MIN_NAN: "zh",
            }

            src_lang = lang_code_map.get(source_language, "zh")
            tgt_lang = lang_code_map.get(target_language, "zh")

            # Set source language
            self.tokenizer.src_lang = src_lang

            # Tokenize input
            encoded = self.tokenizer(text, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
                    max_length=512,
                )

            # Decode translation
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
        """
        Translate text between languages

        Args:
            text: Input text
            source_language: Source language
            target_language: Target language
            use_neural: Whether to use neural translation (default: True)

        Returns:
            Tuple of (translated_text, processing_time)
        """
        start_time = time.time()

        logger.info(f"Translating from {source_language} to {target_language}")

        # If same language, no translation needed
        if source_language == target_language:
            return text, 0.0

        # Priority 1: Semantic/Slang Translation (High Context)
        if (
            source_language == LanguageType.CHINESE
            and target_language == LanguageType.MIN_NAN
            and text in self.slang_dict
        ):
            return self.slang_dict[text], 0.0

        # Priority 2: Dictionary translation first for common phrases (faster)
        dict_translation = self.dictionary_translate(
            text, source_language, target_language
        )

        # If dictionary translation changed the text significantly, use it
        if dict_translation != text and not use_neural:
            processing_time = time.time() - start_time
            return dict_translation, processing_time

        # Use neural translation if available and requested
        if use_neural and self.models_loaded:
            translated_text = self.neural_translate(
                text, source_language, target_language
            )
        else:
            translated_text = dict_translation

        # For Chinese to Min Nan, apply dictionary post-processing
        # This helps with domain-specific terms
        if (
            source_language == LanguageType.CHINESE
            and target_language == LanguageType.MIN_NAN
        ):
            # Replace known phrases in the translation
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
        """
        Translate text and return detailed metadata (Hybrid Strategy)

        Returns:
            dict: {
                "translated_text": str,
                "method": "dictionary" | "neural" | "fallback",
                "confidence": "high" | "low",
                "processing_time": float
            }
        """
        start_time = time.time()

        # 0. Check Slang Dictionary First (Priority 1)
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

        # 1. Dictionary Translation (High Confidence)
        dict_translation = self.dictionary_translate(
            text, source_language, target_language
        )

        # Check if dictionary actually changed anything
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
            "translated_text": text,  # Fallback to original text
            "method": "fallback",
            "confidence": "low",
            "processing_time": processing_time,
        }


# Global translation service instance
translation_service = TranslationService()
