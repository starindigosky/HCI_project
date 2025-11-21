from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
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
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = None
    
    logger.info(f"WebSocket ASR connection established: session={session_id}")

    try:
        # Step 1: Wait for the mandatory 'config' message
        first_message = await websocket.receive()
        
        if "text" in first_message:
            try:
                config_data = json.loads(first_message["text"])
                if config_data.get("type") != "config":
                    raise ValueError("First message must be of type 'config'.")

                # Configure and create the session
                lang = config_data.get("language", "chinese")
                sample_rate = config_data.get("sample_rate", 16000)
                interim_results = config_data.get("interim_results", False)

                session = streaming_session_manager.create_session(session_id, sample_rate=sample_rate)
                
                config = {
                    "language": LanguageType(lang),
                    "interim_results": interim_results,
                    "sample_rate": sample_rate
                }
                
                logger.info(f"Session {session_id} configured with language: {lang}, sample_rate: {sample_rate}")
                await websocket.send_json({"type": "config_updated", **config_data})

            except (json.JSONDecodeError, ValueError) as e:
                error_message = f"Invalid or missing config message: {e}"
                logger.error(error_message)
                await websocket.send_json({"type": "error", "message": error_message})
                await websocket.close()
                return
        else: # This means we received 'bytes' first, which is an error
            error_message = "Connection must start with a 'config' text message, but received binary data first."
            logger.error(error_message)
            await websocket.send_json({"type": "error", "message": error_message})
            await websocket.close()
            return

        # Step 2: Process subsequent messages (audio, stop)
        while True:
            data = await websocket.receive()

            if "text" in data:
                message = json.loads(data["text"])
                message_type = message.get("type")

                if message_type == "stop":
                    final_text = await session.transcribe_final(config["language"])
                    if final_text:
                        await websocket.send_json({
                            "type": "transcription", "text": final_text, "is_final": True
                        })
                    session.reset_buffer()
                    await websocket.send_json({"type": "stopped"})
                
                elif message_type == "config":
                    logger.warning(f"Received unexpected 'config' message during stream for session {session_id}")

            elif "bytes" in data:
                audio_chunk = data["bytes"]
                text = await session.transcribe_stream(
                    audio_chunk, config["language"], interim_results=config["interim_results"]
                )
                if text:
                    await websocket.send_json({
                        "type": "transcription", "text": text, "is_final": not config["interim_results"]
                    })

    except WebSocketDisconnect:
        logger.info(f"WebSocket ASR disconnected: session={session_id}")
    except ConnectionClosedOK:
        logger.info(f"Client closed connection gracefully: session={session_id}")
    except ConnectionClosedError as e:
        logger.warning(f"Client connection closed with error: {e} for session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket ASR error for session {session_id}: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if session_id in streaming_session_manager.sessions:
            streaming_session_manager.remove_session(session_id)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


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
        "target_language": LanguageType.MIN_NAN
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

                            try:
                                await websocket.send_json({
                                    "type": "config_updated",
                                    "source_language": src_lang,
                                    "target_language": tgt_lang
                                })
                            except (ConnectionClosedOK, ConnectionClosedError):
                                logger.warning(f"Could not send config update to disconnected client: session={session_id}")
                                break
                        except ValueError as e:
                            try:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Invalid language. Must be 'chinese' or 'min_nan'"
                                })
                            except (ConnectionClosedOK, ConnectionClosedError):
                                logger.warning(f"Could not send error to disconnected client: session={session_id}")
                                break

                    elif message_type == "stop":
                        # Process final audio
                        # 1. Transcribe
                        transcribed_text = await session.transcribe_final(
                            config["source_language"]
                        )

                        if transcribed_text:
                            try:
                                await websocket.send_json({
                                    "type": "transcription",
                                    "text": transcribed_text
                                })
                            except (ConnectionClosedOK, ConnectionClosedError):
                                logger.warning(f"Could not send transcription to disconnected client: session={session_id}")
                                break

                            # 2. Translate (run in thread pool to avoid blocking)
                            translated_text, translation_time = await run_in_thread(
                                translation_service.translate,
                                transcribed_text,
                                config["source_language"],
                                config["target_language"]
                            )

                            try:
                                await websocket.send_json({
                                    "type": "translation",
                                    "text": translated_text
                                })
                            except (ConnectionClosedOK, ConnectionClosedError):
                                logger.warning(f"Could not send translation to disconnected client: session={session_id}")
                                break

                            # 3. Generate speech (run in thread pool to avoid blocking)
                            output_path, tts_time = await run_in_thread(
                                tts_service.text_to_speech,
                                translated_text,
                                f"voice_chat_{session_id}.wav"
                            )

                            import os
                            filename = os.path.basename(output_path)

                            try:
                                await websocket.send_json({
                                    "type": "audio_ready",
                                    "audio_url": f"/api/v1/audio/{filename}",
                                    "text": translated_text
                                })
                            except (ConnectionClosedOK, ConnectionClosedError):
                                logger.warning(f"Could not send audio ready message to disconnected client: session={session_id}")
                                break

                        # Reset buffer
                        session.reset_buffer()

                        try:
                            await websocket.send_json({
                                "type": "stopped"
                            })
                        except (ConnectionClosedOK, ConnectionClosedError):
                            logger.warning(f"Could not send stop confirmation to disconnected client: session={session_id}")
                            break

                except json.JSONDecodeError:
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON"
                        })
                    except (ConnectionClosedOK, ConnectionClosedError):
                        logger.warning(f"Could not send error to disconnected client: session={session_id}")
                        break

            elif "bytes" in data:
                # Just buffer audio for now
                try:
                    audio_chunk = data["bytes"]
                    session.add_audio_chunk(audio_chunk)

                    try:
                        await websocket.send_json({
                            "type": "audio_received",
                            "size": len(audio_chunk)
                        })
                    except (ConnectionClosedOK, ConnectionClosedError):
                        logger.warning(f"Could not send audio received confirmation to disconnected client: session={session_id}")
                        break

                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}")
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
                    except (ConnectionClosedOK, ConnectionClosedError):
                        logger.warning(f"Could not send error to disconnected client: session={session_id}")
                        break

    except WebSocketDisconnect:
        logger.info(f"WebSocket Voice Chat disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket Voice Chat error: {str(e)}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    finally:
        # Cleanup session
        streaming_session_manager.remove_session(session_id)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
