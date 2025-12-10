import logging
import base64
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import (
    VoucherAnalysisRequest,
    VoucherAnalysisResponse,
    HealthResponse
)
from app.services.vision_service import vision_service

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VERSION = "2.0.0"
MODEL_NAME = "InternVL2-1B"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - NO carga el modelo al iniciar (lazy loading)"""
    logger.info(f"Iniciando servicio OCR con {MODEL_NAME} (modo lazy loading)...")
    logger.info("El modelo se cargará automáticamente en la primera solicitud")
    logger.info(f"El modelo se descargará después de 5 minutos de inactividad para liberar RAM")
    yield
    logger.info("Cerrando servicio...")
    vision_service.unload_model()


app = FastAPI(
    title="Vision OCR Service",
    description=f"Servicio de analisis de vouchers/boletas usando {MODEL_NAME}",
    version=VERSION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check del servicio"""
    is_loaded = vision_service.is_loaded()
    is_lazy = vision_service.is_lazy_mode()

    # En modo lazy, el servicio está healthy aunque el modelo no esté cargado
    if is_lazy:
        status = "healthy (lazy mode - model loads on demand)"
    else:
        status = "healthy" if is_loaded else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=is_loaded,
        model_name=MODEL_NAME,
        version=VERSION
    )


@app.post("/api/analyze-voucher", response_model=VoucherAnalysisResponse)
async def analyze_voucher_base64(request: VoucherAnalysisRequest):
    """
    Analiza un voucher/boleta enviado en base64.

    - **image_base64**: Imagen del voucher en formato base64
    - **expected_amount**: (Opcional) Monto esperado para validar
    - **expected_document**: (Opcional) Documento esperado para validar
    - **custom_prompt**: (Opcional) Prompt personalizado

    Nota: El modelo se carga automáticamente en la primera solicitud (lazy loading).
    La primera solicitud puede tardar ~30 segundos mientras se carga el modelo.
    """
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="Se requiere image_base64")

    try:
        data, validacion_monto, validacion_documento, processing_time = vision_service.analyze_voucher(
            request.image_base64,
            request.expected_amount,
            request.expected_document,
            request.custom_prompt
        )

        return VoucherAnalysisResponse(
            success=True,
            data=data,
            validacion_monto=validacion_monto,
            validacion_documento=validacion_documento,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error analizando voucher: {str(e)}")
        return VoucherAnalysisResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/analyze-voucher/upload", response_model=VoucherAnalysisResponse)
async def analyze_voucher_upload(
    file: UploadFile = File(..., description="Imagen del voucher"),
    expected_amount: Optional[float] = Form(None, description="Monto esperado para validar"),
    expected_document: Optional[str] = Form(None, description="Documento esperado para validar"),
    custom_prompt: Optional[str] = Form(None, description="Prompt personalizado")
):
    """
    Analiza un voucher/boleta subido como archivo.

    - **file**: Archivo de imagen (JPG, PNG, etc.)
    - **expected_amount**: (Opcional) Monto esperado para validar
    - **expected_document**: (Opcional) Documento esperado para validar

    Nota: El modelo se carga automáticamente en la primera solicitud (lazy loading).
    """
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido: {file.content_type}. Permitidos: {allowed_types}"
        )

    try:
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")

        data, validacion_monto, validacion_documento, processing_time = vision_service.analyze_voucher(
            image_base64,
            expected_amount,
            expected_document,
            custom_prompt
        )

        return VoucherAnalysisResponse(
            success=True,
            data=data,
            validacion_monto=validacion_monto,
            validacion_documento=validacion_documento,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error analizando voucher: {str(e)}")
        return VoucherAnalysisResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/ask")
async def ask_about_image(
    file: UploadFile = File(..., description="Imagen"),
    question: str = Form(..., description="Pregunta sobre la imagen")
):
    """
    Hace una pregunta libre sobre una imagen.

    - **file**: Archivo de imagen
    - **question**: Pregunta en español o inglés

    Nota: El modelo se carga automáticamente si no está en memoria.
    """
    # Asegurar que el modelo esté cargado (lazy loading)
    if not vision_service._ensure_loaded():
        raise HTTPException(status_code=503, detail="No se pudo cargar el modelo.")

    try:
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")
        image = vision_service._decode_base64_image(image_base64)

        response = vision_service._chat(image, question)

        return {
            "success": True,
            "question": question,
            "answer": response
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
