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
    """Lifecycle manager - carga el modelo al iniciar"""
    logger.info(f"Iniciando servicio OCR con {MODEL_NAME}...")
    success = vision_service.load_model()
    if not success:
        logger.error(f"No se pudo cargar el modelo {MODEL_NAME}")
    yield
    logger.info("Cerrando servicio...")


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
    return HealthResponse(
        status="healthy" if vision_service.is_loaded() else "degraded",
        model_loaded=vision_service.is_loaded(),
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
    """
    if not vision_service.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo no cargado. Intente mas tarde.")

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
    """
    if not vision_service.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo no cargado. Intente mas tarde.")

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
    """
    if not vision_service.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

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
