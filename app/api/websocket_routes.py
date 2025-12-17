from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import json
import logging
import uuid
from typing import Dict, Any

from ..models.schemas import LanguageType
from ..services.streaming_service import streaming_session_manager
from ..services.tts_service import tts_service
from ..services.translation_service import translation_service
from ..utils.async_helpers import run_in_thread

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/asr")
async def websocket_asr_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ASR (Automatic Speech Recognition)

    The client sends audio chunks and receives transcription results in real-time.

    Message format from client:
    {
        "type": "config",
        "language": "chinese" | "min_nan",
        "interim_results": true | false
    }
    OR
    {
        "type": "audio",
        "data": <base64 encoded audio bytes>
    }
    OR
    {
        "type": "stop"
    }

    Message format to client:
    {
        "type": "transcription",
        "text": "transcribed text",
        "is_final": true | false
    }
    OR
    {
        "type": "error",
        "message": "error message"
    }
    """
    await websocket.accept()

    # Create session
    session_id = str(uuid.uuid4())
    session = streaming_session_manager.create_session(session_id)

    # Session configuration
    # Session configuration
    config = {
        "language": LanguageType.CHINESE,
        "target_language": LanguageType.MIN_NAN,
        "interim_results": False,
    }

    logger.info(f"WebSocket ASR connection established: session={session_id}")

    try:
        while True:
            # Receive message
            data = await websocket.receive()

            if "text" in data:
                # Handle JSON messages
                try:
                    message = json.loads(data["text"])
                    message_type = message.get("type")

                    if message_type == "config":
                        # Update configuration
                        lang = message.get("language", "chinese")
                        target_lang_str = message.get("target_language", "min_nan")
                        try:
                            config["language"] = LanguageType(lang)
                            config["target_language"] = LanguageType(target_lang_str)
                            config["interim_results"] = message.get(
                                "interim_results", False
                            )

                            await websocket.send_json(
                                {
                                    "type": "config_updated",
                                    "language": lang,
                                    "target_language": target_lang_str,
                                    "interim_results": config["interim_results"],
                                }
                            )
                        except ValueError:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": f"Invalid language: {lang}. Must be 'chinese' or 'min_nan'",
                                }
                            )

                    elif message_type == "switch_speaker":
                        # Force finalization of current buffer
                        final_text = await session.transcribe_final(config["language"])

                        if final_text:
                            # Hallucination Filtering
                            # Common hallucinations for empty audio: "你", "你好", "謝謝", "嗯"
                            hallucinations = {
                                "你",
                                "你好",
                                "謝謝",
                                "嗯",
                                "你說",
                                "我",
                                "阿",
                                "蛤",
                            }
                            is_short = len(final_text.strip()) <= 1
                            is_hallucination = final_text.strip() in hallucinations

                            if not (is_short or is_hallucination):
                                # 1. Get detailed translation
                                from app.services.translation_service import (
                                    translation_service,
                                )

                                target_lang = config["target_language"]
                                source_lang = config["language"]

                                trans_result = (
                                    translation_service.translate_with_details(
                                        final_text, source_lang, target_lang
                                    )
                                )

                                logger.info(
                                    f"Forced finalization result: '{final_text}' -> '{trans_result['translated_text']}'"
                                )

                                await websocket.send_json(
                                    {
                                        "type": "transcription",
                                        "text": final_text,
                                        "translated_text": trans_result[
                                            "translated_text"
                                        ],
                                        "method": trans_result["method"],
                                        "confidence": trans_result["confidence"],
                                        "is_final": True,
                                        "cause": "switch_speaker_forced",
                                    }
                                )
                            else:
                                logger.info(
                                    f"Filtered hallucination/short text: '{final_text}'"
                                )

                        # Reset buffer immediately for the new speaker
                        session.reset_buffer()
                        session.silence_counter = 0

                    elif message_type == "stop":
                        # Get final transcription
                        final_text = await session.transcribe_final(config["language"])

                        if final_text:
                            await websocket.send_json(
                                {
                                    "type": "transcription",
                                    "text": final_text,
                                    "is_final": True,
                                }
                            )

                        # Reset buffer for next session
                        session.reset_buffer()

                        await websocket.send_json({"type": "stopped"})

                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": f"Unknown message type: {message_type}",
                            }
                        )

                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid JSON"}
                    )

            elif "bytes" in data:
                # Handle audio bytes
                try:
                    audio_chunk = data["bytes"]

                    # Transcribe stream
                    text, is_final = await session.transcribe_stream(
                        audio_chunk,
                        config["language"],
                        interim_results=config["interim_results"],
                    )

                    if text:
                        # Hallucination Filtering for streaming results
                        # Common hallucinations for empty/noise audio
                        hallucinations = {
                            "你",
                            "你好",
                            "謝謝",
                            "嗯",
                            "你說",
                            "我",
                            "阿",
                            "蛤",
                            "。",
                            "，",
                            "？",
                            "！",
                            " ",
                            "",
                        }

                        # Only filter if it's a "Final" result to avoid blocking legitimate sentence starts
                        if is_final:
                            clean_text = text.strip()
                            if clean_text in hallucinations or len(clean_text) == 0:
                                logger.info(
                                    f"Filtered streaming hallucination: '{text}'"
                                )
                                session.reset_buffer()
                                session.silence_counter = 0
                                continue

                        # 1. Get detailed translation
                        from app.services.translation_service import translation_service

                        # Use configured languages
                        target_lang = config["target_language"]
                        source_lang = config["language"]

                        trans_result = translation_service.translate_with_details(
                            text, source_lang, target_lang
                        )

                        logger.info(
                            f"Sending hybrid result: '{text}' -> '{trans_result['translated_text']}' "
                            f"({trans_result['method']}, Final: {is_final})"
                        )

                        await websocket.send_json(
                            {
                                "type": "transcription",
                                "text": text,  # Source Text (Mandarin)
                                "translated_text": trans_result[
                                    "translated_text"
                                ],  # Target Text (Min Nan)
                                "method": trans_result[
                                    "method"
                                ],  # dictionary vs fallback
                                "confidence": trans_result["confidence"],  # high vs low
                                "is_final": is_final,
                            }
                        )
                    else:
                        logger.debug("Transcription was empty/None")

                except Exception as e:
                    import traceback

                    error_trace = traceback.format_exc()
                    logger.error(f"Error processing audio: {error_trace}")
                    await websocket.send_json(
                        {"type": "error", "message": f"Internal Error: {str(e)}"}
                    )

    except WebSocketDisconnect:
        logger.info(f"WebSocket ASR disconnected: session={session_id}")
    except Exception as e:
        # Suppress "Cannot call send once a close message has been sent" error
        if isinstance(e, RuntimeError) and 'Cannot call "send"' in str(e):
            logger.info(f"WebSocket ASR client disconnected during send: {session_id}")
        else:
            logger.error(f"WebSocket ASR error: {str(e)}")

        # Only try to send error if connection appears open, but catch RuntimeError anyway
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except RuntimeError:
                pass  # Connection already closed
    finally:
        # Cleanup session
        streaming_session_manager.remove_session(session_id)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                pass  # Connection already closed


@router.websocket("/ws/voice-chat")
async def websocket_voice_chat_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice chat with translation

    The client sends audio in one language and receives audio in another language.

    Message format from client:
    {
        "type": "config",
        "source_language": "chinese" | "min_nan",
        "target_language": "chinese" | "min_nan"
    }
    OR
    {
        "type": "audio",
        "data": <base64 encoded audio bytes>
    }
    OR
    {
        "type": "stop"
    }

    Message format to client:
    {
        "type": "transcription",
        "text": "transcribed text"
    }
    OR
    {
        "type": "translation",
        "text": "translated text"
    }
    OR
    {
        "type": "audio_ready",
        "audio_url": "/api/v1/audio/filename.wav"
    }
    OR
    {
        "type": "error",
        "message": "error message"
    }
    """
    await websocket.accept()

    # Create session
    session_id = str(uuid.uuid4())
    session = streaming_session_manager.create_session(session_id)

    # Session configuration
    config = {
        "source_language": LanguageType.CHINESE,
        "target_language": LanguageType.MIN_NAN,
    }

    logger.info(f"WebSocket Voice Chat connection established: session={session_id}")

    try:
        while True:
            data = await websocket.receive()

            if "text" in data:
                try:
                    message = json.loads(data["text"])
                    message_type = message.get("type")

                    if message_type == "config":
                        # Update configuration
                        src_lang = message.get("source_language", "chinese")
                        tgt_lang = message.get("target_language", "min_nan")

                        try:
                            config["source_language"] = LanguageType(src_lang)
                            config["target_language"] = LanguageType(tgt_lang)

                            await websocket.send_json(
                                {
                                    "type": "config_updated",
                                    "source_language": src_lang,
                                    "target_language": tgt_lang,
                                }
                            )
                        except ValueError as e:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": f"Invalid language. Must be 'chinese' or 'min_nan'",
                                }
                            )

                    elif message_type == "stop":
                        # Process final audio
                        # 1. Transcribe
                        transcribed_text = await session.transcribe_final(
                            config["source_language"]
                        )

                        if transcribed_text:
                            await websocket.send_json(
                                {"type": "transcription", "text": transcribed_text}
                            )

                            # 2. Translate (run in thread pool to avoid blocking)
                            translated_text, translation_time = await run_in_thread(
                                translation_service.translate,
                                transcribed_text,
                                config["source_language"],
                                config["target_language"],
                            )

                            await websocket.send_json(
                                {"type": "translation", "text": translated_text}
                            )

                            # 3. Generate speech (run in thread pool to avoid blocking)
                            output_path, tts_time = await run_in_thread(
                                tts_service.text_to_speech,
                                translated_text,
                                f"voice_chat_{session_id}.wav",
                            )

                            import os

                            if output_path:
                                filename = os.path.basename(output_path)

                                await websocket.send_json(
                                    {
                                        "type": "audio_ready",
                                        "audio_url": f"/api/v1/audio/{filename}",
                                        "text": translated_text,
                                    }
                                )
                            else:
                                logger.warning(
                                    "TTS generated no audio (likely empty text), skipping audio response"
                                )
                                await websocket.send_json(
                                    {
                                        "type": "audio_skipped",
                                        "text": translated_text,
                                        "message": "No audio generated",
                                    }
                                )

                        # Reset buffer
                        session.reset_buffer()

                        await websocket.send_json({"type": "stopped"})

                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid JSON"}
                    )

            elif "bytes" in data:
                # Just buffer audio for now
                try:
                    audio_chunk = data["bytes"]
                    session.add_audio_chunk(audio_chunk)

                    await websocket.send_json(
                        {"type": "audio_received", "size": len(audio_chunk)}
                    )

                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}")
                    await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info(f"WebSocket Voice Chat disconnected: session={session_id}")
    except Exception as e:
        # Suppress "Cannot call send once a close message has been sent" error
        if isinstance(e, RuntimeError) and 'Cannot call "send"' in str(e):
            logger.info(
                f"WebSocket Voice Chat client disconnected during send: {session_id}"
            )
        else:
            logger.error(f"WebSocket Voice Chat error: {str(e)}")

        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except RuntimeError:
                pass
    finally:
        # Cleanup session
        streaming_session_manager.remove_session(session_id)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                pass
