from pydantic import BaseModel, Field
from typing import Optional


class VoucherAnalysisRequest(BaseModel):
    """Request para analizar un voucher/boleta"""
    image_base64: Optional[str] = Field(None, description="Imagen en base64")
    expected_amount: Optional[float] = Field(None, description="Monto esperado para validar")
    expected_document: Optional[str] = Field(None, description="Documento esperado para validar")
    expected_name: Optional[str] = Field(None, description="Nombre esperado para validar")
    custom_prompt: Optional[str] = Field(None, description="Prompt personalizado (opcional)")


class ExtractedVoucherData(BaseModel):
    """Datos extraidos del voucher"""
    monto: Optional[float] = Field(None, description="Monto encontrado en el voucher")
    moneda: Optional[str] = Field(None, description="Moneda (PEN, USD, etc)")
    fecha: Optional[str] = Field(None, description="Fecha de la operacion")
    numero_operacion: Optional[str] = Field(None, description="Numero de operacion/referencia")
    banco: Optional[str] = Field(None, description="Banco o entidad financiera")
    documento: Optional[str] = Field(None, description="Documento/DNI del pagador")
    nombre: Optional[str] = Field(None, description="Nombre del beneficiario/destinatario")
    texto_completo: Optional[str] = Field(None, description="Respuesta completa del modelo")


class ValidationResult(BaseModel):
    """Resultado de validacion"""
    coincide: bool = Field(..., description="Si el valor coincide")
    valor_esperado: Optional[str] = Field(None, description="Valor esperado")
    valor_extraido: Optional[str] = Field(None, description="Valor extraido")
    diferencia: Optional[float] = Field(None, description="Diferencia (solo para montos)")
    mensaje: Optional[str] = Field(None, description="Mensaje descriptivo")


class VoucherAnalysisResponse(BaseModel):
    """Response del analisis de voucher"""
    success: bool = Field(..., description="Si el analisis fue exitoso")
    data: Optional[ExtractedVoucherData] = Field(None, description="Datos extraidos")
    validacion_monto: Optional[ValidationResult] = Field(None, description="Resultado de validacion de monto")
    validacion_documento: Optional[ValidationResult] = Field(None, description="Resultado de validacion de documento")
    validacion_nombre: Optional[ValidationResult] = Field(None, description="Resultado de validacion de nombre")
    error: Optional[str] = Field(None, description="Mensaje de error si fallo")
    processing_time_ms: Optional[int] = Field(None, description="Tiempo de procesamiento en ms")


class HealthResponse(BaseModel):
    """Response del health check"""
    status: str
    model_loaded: bool
    model_name: str
    version: str
