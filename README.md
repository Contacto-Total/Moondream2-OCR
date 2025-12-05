# Moondream2 OCR Service

Microservicio de analisis de vouchers/boletas de pago usando el modelo de vision Moondream2.

## Caracteristicas

- Analisis de imagenes de vouchers/boletas
- Extraccion automatica de: monto, fecha, numero de operacion, banco
- Validacion contra monto esperado
- API REST con FastAPI
- Funciona en CPU (no requiere GPU)

## Requisitos

- Python 3.11+
- 4GB RAM minimo (recomendado 8GB)
- Docker (opcional, para despliegue)

## Instalacion Local

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python -m uvicorn app.main:app --host 0.0.0.0 --port 5001
```

## Despliegue con Docker

```bash
# Construir imagen
docker-compose build

# Ejecutar
docker-compose up -d

# Ver logs
docker-compose logs -f
```

## API Endpoints

### Health Check
```
GET /health
```
Respuesta:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Analizar Voucher (Base64)
```
POST /api/analyze-voucher
Content-Type: application/json

{
  "image_base64": "...",
  "expected_amount": 150.00  // opcional
}
```

### Analizar Voucher (Upload)
```
POST /api/analyze-voucher/upload
Content-Type: multipart/form-data

file: <archivo de imagen>
expected_amount: 150.00  // opcional
```

### Respuesta
```json
{
  "success": true,
  "data": {
    "monto": 150.00,
    "moneda": "PEN",
    "fecha": "05/12/2024",
    "numero_operacion": "123456789",
    "banco": "BCP",
    "texto_completo": "..."
  },
  "validation": {
    "expected_amount": 150.00,
    "extracted_amount": 150.00,
    "matches": true,
    "difference": 0.0
  },
  "processing_time_ms": 15234
}
```

## Integracion con Backend Java

Ejemplo de llamada desde Spring Boot:

```java
@Service
public class VoucherValidationService {

    private final RestTemplate restTemplate;
    private final String ocrServiceUrl = "http://localhost:5001";

    public VoucherAnalysisResponse analyzeVoucher(String imageBase64, Double expectedAmount) {
        String url = ocrServiceUrl + "/api/analyze-voucher";

        Map<String, Object> request = new HashMap<>();
        request.put("image_base64", imageBase64);
        if (expectedAmount != null) {
            request.put("expected_amount", expectedAmount);
        }

        return restTemplate.postForObject(url, request, VoucherAnalysisResponse.class);
    }
}
```

## Notas

- El modelo se descarga automaticamente la primera vez (~2GB)
- El primer request puede tardar mas mientras el modelo se inicializa
- Tiempo de procesamiento tipico: 10-20 segundos por imagen en CPU
- Soporta formatos: JPG, PNG, WebP, GIF
