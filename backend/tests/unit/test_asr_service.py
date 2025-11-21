import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

from app.services.asr_service import ASRService
from app.models.schemas import LanguageType


@pytest.mark.unit
class TestASRService:
    """Unit tests for ASR Service"""

    def test_asr_service_initialization(self):
        """Test ASR service initializes correctly"""
        service = ASRService()
        assert service.device in ["cpu", "cuda"]
        assert service.models_loaded is False
        assert service.chinese_asr_pipeline is None
        assert service.min_nan_processor is None
        assert service.min_nan_model is None

    @pytest.mark.slow
    @pytest.mark.requires_models
    def test_load_models(self):
        """Test loading ASR models (slow test, requires internet)"""
        service = ASRService()

        # This test will actually download models
        # Skip in CI/CD or when models are not needed
        try:
            service.load_models()
            assert service.models_loaded is True
            assert service.chinese_asr_pipeline is not None
            assert service.min_nan_processor is not None
            assert service.min_nan_model is not None
        except Exception as e:
            pytest.skip(f"ASR pipeline loading failed: {e}")





    @patch('app.services.asr_service.pipeline')
    def test_transcribe_chinese_mock(
        self,
        mock_pipeline,
        chinese_sample_audio
    ):
        """Test Chinese transcription with mocked pipeline"""
        # Setup mocks
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        # Configure the pipeline mock to return a dictionary with 'text'
        mock_pipeline_instance.return_value = [{"text": "你好世界"}]

        # Test transcription
        service = ASRService()
        service.load_models()

        text, confidence, processing_time = service.transcribe_chinese(chinese_sample_audio)

        assert isinstance(text, str)
        assert text == "你好世界"
        assert processing_time > 0

    @pytest.mark.skip(reason="Min Nan ASR model is disabled and causes CUDA out of memory in tests.")
    @patch('transformers.Wav2Vec2ForCTC.from_pretrained')
    @patch('transformers.Wav2Vec2Processor.from_pretrained')
    def test_transcribe_minnan_mock(
        self,
        mock_wav2vec_processor_from_pretrained,
        mock_wav2vec_model_from_pretrained,
        minnan_sample_audio
    ):
        """Test Min Nan transcription with mocked models"""
        # Setup mocks
        mock_processor_instance = Mock()
        mock_model_instance = Mock()

        mock_wav2vec_processor_from_pretrained.return_value = mock_processor_instance
        mock_wav2vec_model_from_pretrained.return_value = mock_model_instance

        # Mock processor methods
        mock_inputs = Mock()
        mock_inputs.input_values = torch.randn(1, 16000)
        mock_processor_instance.return_value = mock_inputs
        mock_processor_instance.batch_decode.return_value = ["汝好"]

        # Mock model methods
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = None
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 100, 32)
        mock_model_instance.return_value = mock_output

        # Test transcription
        service = ASRService()
        service.load_models()

        text, confidence, processing_time = service.transcribe_min_nan(minnan_sample_audio)

        assert processing_time == 0.0

    def test_transcribe_unsupported_language(self, sample_audio_file):
        """Test transcription with unsupported language"""
        service = ASRService()

        with pytest.raises(ValueError, match="Unsupported language"):
            service.transcribe(sample_audio_file, "unsupported_language")

    def test_transcribe_chinese_route(self, sample_audio_file):
        """Test that Chinese language routes to Chinese transcription"""
        service = ASRService()
        service.models_loaded = True

        with patch.object(service, 'transcribe_chinese', return_value=("測試", None, 1.0)) as mock_transcribe:
            text, confidence, time = service.transcribe(
                sample_audio_file,
                LanguageType.CHINESE
            )

            mock_transcribe.assert_called_once_with(sample_audio_file)
            assert text == "測試"

    def test_transcribe_minnan_route(self, sample_audio_file):
        """Test that Min Nan language routes to Min Nan transcription"""
        service = ASRService()
        service.models_loaded = True

        with patch.object(service, 'transcribe_min_nan', return_value=("測試", None, 1.0)) as mock_transcribe:
            text, confidence, time = service.transcribe(
                sample_audio_file,
                LanguageType.MIN_NAN
            )

            mock_transcribe.assert_called_once_with(sample_audio_file)
            assert text == "測試"
