from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class VoucherAnalysisRequest(BaseModel):
    """Request para analizar un voucher/boleta"""
    image_base64: Optional[str] = Field(None, description="Imagen en base64")
    expected_amount: Optional[float] = Field(None, description="Monto esperado para validar")


class ExtractedVoucherData(BaseModel):
    """Datos extraidos del voucher"""
    monto: Optional[float] = Field(None, description="Monto encontrado en el voucher")
    moneda: Optional[str] = Field(None, description="Moneda (PEN, USD, etc)")
    fecha: Optional[str] = Field(None, description="Fecha de la operacion")
    numero_operacion: Optional[str] = Field(None, description="Numero de operacion/referencia")
    banco: Optional[str] = Field(None, description="Banco o entidad financiera")
    texto_completo: Optional[str] = Field(None, description="Texto completo extraido")


class VoucherAnalysisResponse(BaseModel):
    """Response del analisis de voucher"""
    success: bool = Field(..., description="Si el analisis fue exitoso")
    data: Optional[ExtractedVoucherData] = Field(None, description="Datos extraidos")
    validation: Optional[dict] = Field(None, description="Resultado de validacion contra monto esperado")
    error: Optional[str] = Field(None, description="Mensaje de error si fallo")
    processing_time_ms: Optional[int] = Field(None, description="Tiempo de procesamiento en ms")


class HealthResponse(BaseModel):
    """Response del health check"""
    status: str
    model_loaded: bool
    version: str
