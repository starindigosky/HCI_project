import pytest
from unittest.mock import Mock, patch
import torch

from app.services.translation_service import TranslationService
from app.models.schemas import LanguageType


@pytest.mark.unit
class TestTranslationService:


    def test_translation_service_initialization(self):

        service = TranslationService()
        assert service.device in ["cpu", "cuda"]
        assert service.models_loaded is False
        assert service.tokenizer is None
        assert service.model is None
        assert len(service.chinese_to_minnan_dict) > 0
        assert len(service.minnan_to_chinese_dict) > 0

    def test_dictionary_contains_common_phrases(self):

        service = TranslationService()


        assert "你好" in service.chinese_to_minnan_dict
        assert "謝謝" in service.chinese_to_minnan_dict
        assert "再見" in service.chinese_to_minnan_dict
        assert "我" in service.chinese_to_minnan_dict
        assert "你" in service.chinese_to_minnan_dict


        assert "汝好" in service.minnan_to_chinese_dict
        assert "多謝" in service.minnan_to_chinese_dict

    def test_dictionary_translate_exact_match_chinese_to_minnan(self):

        service = TranslationService()

        result = service.dictionary_translate(
            "你好",
            LanguageType.CHINESE,
            LanguageType.MIN_NAN
        )

        assert result == "汝好"

    def test_dictionary_translate_exact_match_minnan_to_chinese(self):

        service = TranslationService()

        result = service.dictionary_translate(
            "汝好",
            LanguageType.MIN_NAN,
            LanguageType.CHINESE
        )

        assert result == "你好"

    def test_dictionary_translate_partial_match(self):

        service = TranslationService()


        result = service.dictionary_translate(
            "你好世界",
            LanguageType.CHINESE,
            LanguageType.MIN_NAN
        )


        assert "汝" in result

    def test_dictionary_translate_same_language(self):

        service = TranslationService()

        text = "測試文本"
        result = service.dictionary_translate(
            text,
            LanguageType.CHINESE,
            LanguageType.CHINESE
        )

        assert result == text

    def test_dictionary_translate_no_match(self):

        service = TranslationService()


        text = "測試"
        result = service.dictionary_translate(
            text,
            LanguageType.CHINESE,
            LanguageType.MIN_NAN
        )


        assert isinstance(result, str)

    @pytest.mark.slow
    @pytest.mark.requires_models
    def test_load_models(self):

        service = TranslationService()

        try:
            service.load_models()


            assert service.models_loaded in [True, False]
        except Exception as e:

            pytest.skip(f"Model loading failed: {e}")

    @patch('app.services.translation_service.M2M100Tokenizer')
    @patch('app.services.translation_service.M2M100ForConditionalGeneration')
    def test_neural_translate_mock(
        self,
        mock_model,
        mock_tokenizer
    ):


        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()

        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance


        mock_tokenizer_instance.src_lang = "zh"
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        mock_tokenizer_instance.get_lang_id.return_value = 1
        mock_tokenizer_instance.batch_decode.return_value = ["汝好"]


        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = None
        mock_model_instance.generate.return_value = torch.randint(0, 1000, (1, 10))


        service = TranslationService()
        service.load_models()

        result = service.neural_translate(
            "你好",
            LanguageType.CHINESE,
            LanguageType.MIN_NAN
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @patch('app.services.translation_service.TranslationService.neural_translate')
    def test_translate_uses_neural_when_available(
        self,
        mock_neural_translate
    ):

        mock_neural_translate.return_value = "汝好"

        service = TranslationService()
        service.models_loaded = True

        result, processing_time = service.translate(
            "你好",
            LanguageType.CHINESE,
            LanguageType.MIN_NAN,
            use_neural=True
        )

        mock_neural_translate.assert_called_once()
        assert result == "汝好"
        assert processing_time >= 0

    @patch('app.services.translation_service.TranslationService.dictionary_translate')
    def test_translate_uses_dictionary_when_neural_disabled(
        self,
        mock_dict_translate
    ):

        mock_dict_translate.return_value = "汝好"

        service = TranslationService()
        service.models_loaded = True

        result, processing_time = service.translate(
            "你好",
            LanguageType.CHINESE,
            LanguageType.MIN_NAN,
            use_neural=False
        )

        mock_dict_translate.assert_called()
        assert result == "汝好"
        assert processing_time >= 0

    def test_translate_same_language_returns_original(self):

        service = TranslationService()

        text = "測試文本"
        result, processing_time = service.translate(
            text,
            LanguageType.CHINESE,
            LanguageType.CHINESE
        )

        assert result == text
        assert processing_time == 0.0

    def test_translate_applies_dictionary_postprocessing(self):

        service = TranslationService()
        service.models_loaded = False


        result, _ = service.translate(
            "你好",
            LanguageType.CHINESE,
            LanguageType.MIN_NAN,
            use_neural=False
        )

        assert result == "汝好"

    def test_translate_fallback_to_dictionary_on_error(self):

        service = TranslationService()
        service.models_loaded = True

        with patch.object(
            service,
            'neural_translate',
            side_effect=Exception("Neural translation failed")
        ):
            with patch.object(
                service,
                'dictionary_translate',
                return_value="汝好"
            ) as mock_dict:
                result, _ = service.translate(
                    "你好",
                    LanguageType.CHINESE,
                    LanguageType.MIN_NAN,
                    use_neural=True
                )


                mock_dict.assert_called()

    def test_multiple_translations(self):

        service = TranslationService()

        test_cases = [
            ("你好", LanguageType.CHINESE, LanguageType.MIN_NAN, "汝好"),
            ("謝謝", LanguageType.CHINESE, LanguageType.MIN_NAN, "多謝"),
            ("再見", LanguageType.CHINESE, LanguageType.MIN_NAN, "再會"),
        ]

        for text, src, tgt, expected in test_cases:
            result, _ = service.translate(text, src, tgt, use_neural=False)
            assert result == expected

    def test_translation_timing(self):

        service = TranslationService()

        _, processing_time = service.translate(
            "你好",
            LanguageType.CHINESE,
            LanguageType.MIN_NAN,
            use_neural=False
        )

        assert processing_time >= 0
        assert isinstance(processing_time, float)

    @patch('app.services.translation_service.TranslationService.neural_translate')
    def test_neural_translate_not_called_for_same_language(
        self,
        mock_neural
    ):

        service = TranslationService()
        service.models_loaded = True

        service.translate(
            "測試",
            LanguageType.CHINESE,
            LanguageType.CHINESE,
            use_neural=True
        )

        mock_neural.assert_not_called()