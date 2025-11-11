import pytest
from unittest.mock import patch, Mock, AsyncMock
from fastapi import UploadFile
from io import BytesIO
import json
from app.models.schemas import LanguageType # Import LanguageType for cleaner config


@pytest.mark.integration
class TestHealthEndpoint:
    """Integration tests for health endpoint"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


@pytest.mark.integration
class TestASREndpoint:
    """Integration tests for ASR endpoint"""

    @patch('app.api.routes.asr_service')
    def test_transcribe_audio_chinese(
        self,
        mock_asr_service,
        client,
        chinese_sample_audio
    ):
        """Test ASR endpoint with Chinese audio"""
        # Mock the service
        mock_asr_service.transcribe.return_value = ("你好世界", None, 1.5)

        # Read audio file
        with open(chinese_sample_audio, 'rb') as f:
            files = {'audio_file': ('chinese.wav', f, 'audio/wav')}
            data = {'language': 'chinese'}

            response = client.post(
                "/api/v1/asr/transcribe",
                files=files,
                data=data
            )

        assert response.status_code == 200
        result = response.json()
        assert "text" in result
        assert "language" in result
        assert "processing_time" in result
        assert result["text"] == "你好世界"
        assert result["language"] == "chinese"

    @patch('app.api.routes.asr_service')
    def test_transcribe_audio_minnan(
        self,
        mock_asr_service,
        client,
        minnan_sample_audio
    ):
        """Test ASR endpoint with Min Nan audio"""
        mock_asr_service.transcribe.return_value = ("汝好", None, 1.5)

        with open(minnan_sample_audio, 'rb') as f:
            files = {'audio_file': ('minnan.wav', f, 'audio/wav')}
            data = {'language': 'min_nan'}

            response = client.post(
                "/api/v1/asr/transcribe",
                files=files,
                data=data
            )

        assert response.status_code == 200
        result = response.json()
        assert result["text"] == "汝好"
        assert result["language"] == "min_nan"

    def test_transcribe_audio_invalid_format(self, client):
        """Test ASR with invalid audio format"""
        # Create a fake text file
        files = {'audio_file': ('test.txt', BytesIO(b'not an audio file'), 'text/plain')}
        data = {'language': 'chinese'}

        response = client.post(
            "/api/v1/asr/transcribe",
            files=files,
            data=data
        )

        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_transcribe_audio_missing_file(self, client):
        """Test ASR without audio file"""
        data = {'language': 'chinese'}

        response = client.post(
            "/api/v1/asr/transcribe",
            data=data
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.integration
class TestTTSEndpoint:
    """Integration tests for TTS endpoint"""

    @patch('app.api.routes.tts_service')
    def test_synthesize_speech(
        self,
        mock_tts_service,
        client,
        sample_chinese_text,
        tmp_path
    ):
        """Test TTS endpoint"""
        output_file = str(tmp_path / "output.wav")
        mock_tts_service.translate_and_speak.return_value = (
            sample_chinese_text,
            output_file,
            2.0
        )

        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": sample_chinese_text,
                "source_language": "chinese",
                "target_language": "min_nan"
            }
        )

        assert response.status_code == 200
        result = response.json()
        assert "audio_url" in result
        assert "text" in result
        assert "processing_time" in result
        assert result["text"] == sample_chinese_text
        assert "/api/v1/audio/" in result["audio_url"]

    @patch('app.api.routes.tts_service')
    def test_synthesize_speech_default_languages(
        self,
        mock_tts_service,
        client,
        tmp_path
    ):
        """Test TTS with default language settings"""
        output_file = str(tmp_path / "output.wav")
        mock_tts_service.translate_and_speak.return_value = (
            "測試",
            output_file,
            1.5
        )

        response = client.post(
            "/api/v1/tts/synthesize",
            json={"text": "測試"}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["source_language"] == "chinese"
        assert result["target_language"] == "min_nan"

    def test_synthesize_speech_empty_text(self, client):
        """Test TTS with empty text"""
        response = client.post(
            "/api/v1/tts/synthesize",
            json={"text": ""}
        )

        # Should get validation error
        assert response.status_code == 422

    def test_synthesize_speech_invalid_json(self, client):
        """Test TTS with invalid JSON"""
        response = client.post(
            "/api/v1/tts/synthesize",
            data="not a json"
        )

        assert response.status_code == 422


@pytest.mark.integration
class TestVoiceConversionEndpoint:
    """Integration tests for voice conversion endpoint"""

    @patch('app.api.routes.asr_service')
    @patch('app.api.routes.tts_service')
    def test_voice_to_voice_conversion(
        self,
        mock_tts_service,
        mock_asr_service,
        client,
        chinese_sample_audio,
        tmp_path
    ):
        """Test voice-to-voice conversion"""
        # Mock ASR
        mock_asr_service.transcribe.return_value = ("你好世界", None, 1.5)

        # Mock TTS
        output_file = str(tmp_path / "converted.wav")
        mock_tts_service.translate_and_speak.return_value = (
            "你好世界",
            output_file,
            2.0
        )

        with open(chinese_sample_audio, 'rb') as f:
            files = {'audio_file': ('chinese.wav', f, 'audio/wav')}
            data = {
                'source_language': 'chinese',
                'target_language': 'min_nan'
            }

            response = client.post(
                "/api/v1/voice-conversion",
                files=files,
                data=data
            )

        assert response.status_code == 200
        result = response.json()
        assert "transcribed_text" in result
        assert "audio_url" in result
        assert "processing_time" in result
        assert result["transcribed_text"] == "你好世界"
        assert result["source_language"] == "chinese"
        assert result["target_language"] == "min_nan"

    @patch('app.api.routes.asr_service')
    @patch('app.api.routes.tts_service')
    def test_voice_conversion_default_languages(
        self,
        mock_tts_service,
        mock_asr_service,
        client,
        chinese_sample_audio,
        tmp_path
    ):
        """Test voice conversion with default languages"""
        mock_asr_service.transcribe.return_value = ("測試", None, 1.0)

        output_file = str(tmp_path / "output.wav")
        mock_tts_service.translate_and_speak.return_value = (
            "測試",
            output_file,
            1.0
        )

        with open(chinese_sample_audio, 'rb') as f:
            files = {'audio_file': ('audio.wav', f, 'audio/wav')}

            response = client.post(
                "/api/v1/voice-conversion",
                files=files
            )

        assert response.status_code == 200
        result = response.json()
        assert result["source_language"] == "chinese"
        assert result["target_language"] == "min_nan"

    def test_voice_conversion_invalid_format(self, client):
        """Test voice conversion with invalid file format"""
        files = {'audio_file': ('test.pdf', BytesIO(b'fake pdf'), 'application/pdf')}

        response = client.post(
            "/api/v1/voice-conversion",
            files=files
        )

        assert response.status_code == 400


@pytest.mark.integration
class TestAudioDownloadEndpoint:
    """Integration tests for audio download endpoint"""

    def test_download_existing_audio(self, client, tmp_path, monkeypatch):
        """Test downloading an existing audio file"""
        # Create a fake audio file in outputs directory
        import os
        from app.config import settings

        # Temporarily set output dir
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(exist_ok=True)
        monkeypatch.setattr('app.config.settings.OUTPUT_DIR', str(output_dir))

        audio_file = output_dir / "test_audio.wav"
        audio_file.write_bytes(b"fake audio content")

        response = client.get("/api/v1/audio/test_audio.wav")

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_download_nonexistent_audio(self, client):
        """Test downloading a non-existent audio file"""
        response = client.get("/api/v1/audio/nonexistent.wav")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.integration
class TestModelLoadingEndpoint:
    """Integration tests for model loading endpoint"""

    @patch('app.api.routes.asr_service')
    @patch('app.api.routes.tts_service')
    def test_load_models_success(
        self,
        mock_tts_service,
        mock_asr_service,
        client
    ):
        """Test successful model loading"""
        mock_asr_service.models_loaded = False
        mock_tts_service.models_loaded = False

        mock_asr_service.load_models.return_value = None
        mock_tts_service.load_models.return_value = None

        # After loading, set to True
        def load_asr():
            mock_asr_service.models_loaded = True

        def load_tts():
            mock_tts_service.models_loaded = True

        mock_asr_service.load_models.side_effect = load_asr
        mock_tts_service.load_models.side_effect = load_tts

        response = client.post("/api/v1/models/load")

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert "Models loaded successfully" in result["message"]

    @patch('app.api.routes.asr_service')
    def test_load_models_already_loaded(
        self,
        mock_asr_service,
        client
    ):
        """Test loading models when already loaded"""
        mock_asr_service.models_loaded = True

        response = client.post("/api/v1/models/load")

        assert response.status_code == 200


@pytest.mark.integration
class TestWebSocketEndpoints:
    """Integration tests for WebSocket endpoints"""

    @patch('app.api.websocket_routes.streaming_session_manager')
    @patch('app.api.websocket_routes.tts_service')
    @patch('app.api.websocket_routes.translation_service')
    # Patch the StreamingASRService class itself
    @patch('app.services.streaming_service.StreamingASRService')
    def test_websocket_asr_endpoint(
        self,
        mock_streaming_asr_service_class, # This is the mock of the class
        mock_translation_service,
        mock_tts_service,
        mock_streaming_session_manager,
        client
    ):
        """Test WebSocket ASR endpoint"""
        # Configure the mock StreamingASRService class to return a Mock instance
        mock_session_instance = Mock() # Changed to Mock
        mock_streaming_asr_service_class.return_value = mock_session_instance

        # Make async methods of mock_session_instance be AsyncMocks
        mock_session_instance.transcribe_stream = AsyncMock()
        mock_session_instance.transcribe_final = AsyncMock()

        # Configure streaming_session_manager to return our mock instance
        mock_streaming_session_manager.create_session.return_value = mock_session_instance

        # Mock ASR service transcribe method within the mock session instance
        # Use side_effect to simulate the internal call to add_audio_chunk
        async def transcribe_stream_side_effect(audio_chunk, language, interim_results):
            # REMOVED await from here
            mock_session_instance.add_audio_chunk(audio_chunk) # Simulate internal call
            return "你好"

        mock_session_instance.transcribe_stream.side_effect = transcribe_stream_side_effect
        mock_session_instance.transcribe_final.return_value = "你好世界"

        with client.websocket_connect("/api/v1/ws/asr") as websocket:
            # 1. Send config message
            websocket.send_json({
                "type": "config",
                "language": LanguageType.CHINESE.value, # Use .value for enum
                "interim_results": True
            })
            response = websocket.receive_json()
            assert response == {
                "type": "config_updated",
                "language": LanguageType.CHINESE.value,
                "interim_results": True
            }

            # 2. Send audio chunk
            websocket.send_bytes(b"fake_audio_chunk")
            response = websocket.receive_json()
            assert response == {
                "type": "transcription",
                "text": "你好",
                "is_final": False
            }
            # Now add_audio_chunk should be called on the mock_session_instance
            mock_session_instance.add_audio_chunk.assert_called_once_with(b"fake_audio_chunk")
            mock_session_instance.transcribe_stream.assert_called_once_with(
                b"fake_audio_chunk", LanguageType.CHINESE, interim_results=True
            )

            # Reset mock for next call
            mock_session_instance.add_audio_chunk.reset_mock()
            mock_session_instance.transcribe_stream.reset_mock()

            # 3. Send another audio chunk
            websocket.send_bytes(b"another_fake_audio_chunk")
            response = websocket.receive_json()
            assert response == {
                "type": "transcription",
                "text": "你好",
                "is_final": False
            }
            mock_session_instance.add_audio_chunk.assert_called_once_with(b"another_fake_audio_chunk")
            mock_session_instance.transcribe_stream.assert_called_once_with(
                b"another_fake_audio_chunk", LanguageType.CHINESE, interim_results=True
            )

            # 4. Send stop message
            websocket.send_json({"type": "stop"})
            response = websocket.receive_json()
            assert response == {
                "type": "transcription",
                "text": "你好世界",
                "is_final": True
            }
            response = websocket.receive_json()
            assert response == {"type": "stopped"}

            mock_session_instance.transcribe_final.assert_called_once_with(LanguageType.CHINESE)
            mock_session_instance.reset_buffer.assert_called_once()

        mock_streaming_session_manager.remove_session.assert_called_once()
