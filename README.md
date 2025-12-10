# Vision OCR Service - InternVL2-1B

Servicio de análisis de vouchers/boletas de pago usando el modelo de visión InternVL2-1B.

## Características

- Extracción automática de datos de vouchers:
  - Monto y moneda
  - Fecha de operación
  - Número de operación/referencia
  - Banco o entidad financiera
  - Documento/DNI del pagador
- Validación de monto contra valor esperado
- Validación de documento contra valor esperado
- API REST con FastAPI
- Soporte para imágenes en base64 y upload directo
- Endpoint para preguntas libres sobre imágenes

## Requisitos del Sistema

- Python 3.11+
- ~2-3GB RAM disponible
- ~2GB espacio en disco (modelo + dependencias)

## Instalación Local

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o en Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servicio
uvicorn app.main:app --host 0.0.0.0 --port 5002
```

## Instalación con Docker

```bash
# Construir imagen
docker-compose build

# Ejecutar servicio
docker-compose up -d

# Ver logs
docker-compose logs -f vision-ocr
```

## Endpoints

### Health Check
```
GET /health
```
Respuesta:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "InternVL2-1B",
  "version": "2.0.0"
}
```

### Analizar Voucher (Base64)
```
POST /api/analyze-voucher
Content-Type: application/json

{
  "image_base64": "base64_encoded_image",
  "expected_amount": 150.00,
  "expected_document": "12345678"
}
```

### Analizar Voucher (Upload)
```
POST /api/analyze-voucher/upload
Content-Type: multipart/form-data

file: [imagen]
expected_amount: 150.00 (opcional)
expected_document: 12345678 (opcional)
```

### Pregunta Libre sobre Imagen
```
POST /api/ask
Content-Type: multipart/form-data

file: [imagen]
question: "¿Cuál es el monto total?"
```

## Respuesta de Análisis

```json
{
  "success": true,
  "data": {
    "monto": 150.00,
    "moneda": "PEN",
    "fecha": "10/12/2024",
    "numero_operacion": "12345678",
    "banco": "BCP",
    "documento": "12345678",
    "texto_completo": "Respuesta completa del modelo..."
  },
  "validacion_monto": {
    "coincide": true,
    "valor_esperado": "150.0",
    "valor_extraido": "150.0",
    "diferencia": 0.0,
    "mensaje": "El monto coincide"
  },
  "validacion_documento": {
    "coincide": true,
    "valor_esperado": "12345678",
    "valor_extraido": "12345678",
    "mensaje": "El documento coincide"
  },
  "processing_time_ms": 1234
}
```

## Modelo Utilizado

**InternVL2-1B** de OpenGVLab
- Modelo de visión multimodal ligero
- ~1GB de RAM en uso activo
- Capacidad de responder preguntas sobre imágenes
- Buena extracción de texto y números

## Configuración

Variables de entorno:
- `PYTHONUNBUFFERED`: Activar logs en tiempo real
- `TRANSFORMERS_CACHE`: Directorio de caché del modelo

## Notas de Despliegue

1. La primera ejecución descargará el modelo (~2GB)
2. El modelo se cachea en `/root/.cache/huggingface`
3. El healthcheck tiene `start_period: 180s` para dar tiempo a la carga inicial
4. Límite de memoria: 3GB (configurable en docker-compose.yml)
