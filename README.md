# Vision OCR Service - Florence-2-base

Servicio de análisis de vouchers/boletas de pago usando el modelo Florence-2-base de Microsoft.

## Características

- **Bajo consumo de RAM:** ~1GB (vs 4GB de otros modelos)
- Extracción automática de datos de vouchers:
  - Monto y moneda
  - Fecha de operación
  - Número de operación/referencia
  - Banco o entidad financiera (Yape, Plin, BCP, etc.)
  - Documento/DNI del pagador
- Validación de monto contra valor esperado
- Validación de documento contra valor esperado
- **Lazy loading:** El modelo se carga solo cuando se necesita y se descarga después de 5 minutos de inactividad

## Requisitos del Sistema

- Python 3.10+
- ~1-2GB RAM disponible (cuando procesa)
- ~1GB espacio en disco (modelo)

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/Contacto-Total/Moondream2-OCR.git
cd Moondream2-OCR

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servicio
uvicorn app.main:app --host 0.0.0.0 --port 5002
```

## Endpoints

### Health Check
```
GET /health
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

## Modelo

**Florence-2-base** de Microsoft
- ~230M parámetros
- ~1GB RAM en uso
- OCR + comprensión de imágenes
- Rápido en CPU (~5-10 segundos)

## Lazy Loading

El modelo NO se carga al iniciar el servicio para ahorrar RAM:
- Se carga automáticamente en la primera solicitud (~10-15 segundos)
- Se descarga después de 5 minutos sin uso
- Libera ~1GB de RAM cuando está inactivo
