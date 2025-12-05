import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
from typing import Optional

from app.models.schemas import (
    VoucherAnalysisRequest,
    VoucherAnalysisResponse,
    HealthResponse,
    ExtractedVoucherData
)
from app.services.moondream_service import moondream_service

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - carga el modelo al iniciar"""
    logger.info("Iniciando servicio Moondream2 OCR...")
    success = moondream_service.load_model()
    if not success:
        logger.error("No se pudo cargar el modelo Moondream2")
    yield
    logger.info("Cerrando servicio...")


app = FastAPI(
    title="Moondream2 OCR Service",
    description="Servicio de analisis de vouchers/boletas de pago usando Moondream2",
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
        status="healthy" if moondream_service.is_loaded() else "degraded",
        model_loaded=moondream_service.is_loaded(),
        version=VERSION
    )


@app.post("/api/analyze-voucher", response_model=VoucherAnalysisResponse)
async def analyze_voucher_base64(request: VoucherAnalysisRequest):
    """
    Analiza un voucher/boleta enviado en base64.

    - **image_base64**: Imagen del voucher en formato base64
    - **expected_amount**: (Opcional) Monto esperado para validar
    """
    if not moondream_service.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo no cargado. Intente mas tarde.")

    if not request.image_base64:
        raise HTTPException(status_code=400, detail="Se requiere image_base64")

    try:
        # Analizar voucher
        data, processing_time = moondream_service.analyze_voucher(request.image_base64)

        # Validar contra monto esperado si se proporciono
        validation = None
        if request.expected_amount is not None and data.monto is not None:
            tolerance = 0.01  # 1 centavo de tolerancia
            matches = abs(data.monto - request.expected_amount) <= tolerance
            validation = {
                "expected_amount": request.expected_amount,
                "extracted_amount": data.monto,
                "matches": matches,
                "difference": round(data.monto - request.expected_amount, 2)
            }

        return VoucherAnalysisResponse(
            success=True,
            data=data,
            validation=validation,
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
    expected_amount: Optional[float] = Form(None, description="Monto esperado para validar")
):
    """
    Analiza un voucher/boleta subido como archivo.

    - **file**: Archivo de imagen (JPG, PNG, etc.)
    - **expected_amount**: (Opcional) Monto esperado para validar
    """
    if not moondream_service.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo no cargado. Intente mas tarde.")

    # Validar tipo de archivo
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido: {file.content_type}. Permitidos: {allowed_types}"
        )

    try:
        # Leer archivo y convertir a base64
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")

        # Analizar voucher
        data, processing_time = moondream_service.analyze_voucher(image_base64)

        # Validar contra monto esperado si se proporciono
        validation = None
        if expected_amount is not None and data.monto is not None:
            tolerance = 0.01
            matches = abs(data.monto - expected_amount) <= tolerance
            validation = {
                "expected_amount": expected_amount,
                "extracted_amount": data.monto,
                "matches": matches,
                "difference": round(data.monto - expected_amount, 2)
            }

        return VoucherAnalysisResponse(
            success=True,
            data=data,
            validation=validation,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error analizando voucher: {str(e)}")
        return VoucherAnalysisResponse(
            success=False,
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
