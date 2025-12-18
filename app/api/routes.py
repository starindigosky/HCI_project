from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import uuid
import logging
import aiofiles


from ..models.schemas import (
    ASRResponse,
    TTSRequest,
    TTSResponse,
    VoiceConversionResponse,
    HealthResponse,
    LanguageType,
)
from ..services.asr_service import asr_service
from ..services.tts_service import tts_service
from ..services.translation_service import translation_service
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


def cleanup_file(file_path: str):

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():

    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        models_loaded=(
            asr_service.models_loaded
            and tts_service.models_loaded

        ),
    )


@router.post("/asr/transcribe", response_model=ASRResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: LanguageType = Form(
        ..., description="Language of the audio (chinese or min_nan)"
    ),
):

    try:

        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed formats: {settings.ALLOWED_AUDIO_FORMATS}",
            )


        file_id = uuid.uuid4().hex
        file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{file_ext}")

        async with aiofiles.open(file_path, "wb") as f:
            content = await audio_file.read()
            await f.write(content)

        logger.info(f"Saved uploaded file to: {file_path}")


        text, confidence, processing_time = asr_service.transcribe(file_path, language)


        background_tasks.add_task(cleanup_file, file_path)

        return ASRResponse(
            text=text,
            language=language,
            confidence=confidence,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")

        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tts/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest, background_tasks: BackgroundTasks):

    try:
        logger.info(f"TTS request: {request.text[:50]}...")


        translated_text, output_path, processing_time = tts_service.translate_and_speak(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
        )

        if output_path:

            filename = os.path.basename(output_path)
            audio_url = f"/api/v1/audio/{filename}"

            return TTSResponse(
                audio_url=audio_url,
                text=translated_text,
                source_language=request.source_language,
                target_language=request.target_language,
                processing_time=processing_time,
            )
        else:

            raise HTTPException(
                status_code=400,
                detail="Could not generate audio (text might be empty or invalid)",
            )

    except Exception as e:
        logger.error(f"Error in synthesize_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice-conversion", response_model=VoiceConversionResponse)
async def voice_to_voice_conversion(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to convert"),
    source_language: LanguageType = Form(
        default=LanguageType.CHINESE, description="Source language of the audio"
    ),
    target_language: LanguageType = Form(
        default=LanguageType.MIN_NAN, description="Target language for output audio"
    ),
):

    try:

        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed formats: {settings.ALLOWED_AUDIO_FORMATS}",
            )


        file_id = uuid.uuid4().hex
        input_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{file_ext}")

        async with aiofiles.open(input_path, "wb") as f:
            content = await audio_file.read()
            await f.write(content)

        logger.info(
            f"Processing voice conversion: {source_language} -> {target_language}"
        )


        transcribed_text, _, asr_time = asr_service.transcribe(
            input_path, source_language
        )


        translated_text, output_path, tts_time = tts_service.translate_and_speak(
            text=transcribed_text,
            source_language=source_language,
            target_language=target_language,
        )


        filename = os.path.basename(output_path)
        audio_url = f"/api/v1/audio/{filename}"

        total_time = asr_time + tts_time


        background_tasks.add_task(cleanup_file, input_path)

        return VoiceConversionResponse(
            transcribed_text=transcribed_text,
            audio_url=audio_url,
            source_language=source_language,
            target_language=target_language,
            processing_time=total_time,
        )

    except Exception as e:
        logger.error(f"Error in voice_to_voice_conversion: {str(e)}")

        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audio/{filename}")
async def get_audio_file(filename: str):

    file_path = os.path.join(settings.OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(path=file_path, media_type="audio/wav", filename=filename)


@router.delete("/audio/{filename}")
async def delete_audio_file(filename: str):

    try:

        if ".." in filename or "/" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = os.path.join(settings.OUTPUT_DIR, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        os.remove(file_path)
        logger.info(f"Deleted audio file: {file_path}")

        return {"status": "success", "message": f"Deleted {filename}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tts/history")
async def get_audio_history():

    try:
        files = []
        if os.path.exists(settings.OUTPUT_DIR):
            for filename in os.listdir(settings.OUTPUT_DIR):
                if any(
                    filename.lower().endswith(ext)
                    for ext in settings.ALLOWED_AUDIO_FORMATS
                ):
                    file_path = os.path.join(settings.OUTPUT_DIR, filename)
                    stats = os.stat(file_path)
                    files.append(
                        {
                            "filename": filename,
                            "url": f"/api/v1/audio/{filename}",
                            "created_at": stats.st_mtime,
                            "size": stats.st_size,
                        }
                    )


        files.sort(key=lambda x: x["created_at"], reverse=True)
        return files[:7]
    except Exception as e:
        logger.error(f"Error listing audio history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/load")
async def load_models():

    try:
        if not asr_service.models_loaded:
            asr_service.load_models()

        if not tts_service.models_loaded:
            tts_service.load_models()

        if not translation_service.models_loaded:
            translation_service.load_models()

        return {
            "status": "success",
            "message": "Models loaded successfully",
            "asr_loaded": asr_service.models_loaded,
            "tts_loaded": tts_service.models_loaded,
            "translation_loaded": translation_service.models_loaded,
        }
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))