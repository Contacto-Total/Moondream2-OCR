"""
Servicio de análisis de texto usando Gemma (Google AI)
Recibe texto extraído por OCR y lo analiza para extraer datos estructurados
"""

import os
import json
import logging
from typing import Optional, Dict, Any

import google.generativeai as genai

logger = logging.getLogger(__name__)

# Configurar API key de Google AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCLm8OTco_WLaKuVYiOtO7EI5xc4xteJD4")
genai.configure(api_key=GEMINI_API_KEY)


class GemmaAnalyzerService:
    """
    Servicio que usa Gemma para analizar texto de comprobantes.
    Gemma es un modelo de texto que "entiende" contexto.
    Límites gratis: 14,400 requests/día, 30 RPM
    """

    def __init__(self):
        # Usar gemma-3-27b para mejor comprensión (o gemma-3-12b si es muy lento)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
        self.model_name = "Gemma 3 27B"
        logger.info(f"Servicio {self.model_name} inicializado")

    def analyze_voucher_text(self, ocr_text: str) -> Dict[str, Any]:
        """
        Analiza el texto extraído de un comprobante y extrae datos estructurados.

        Args:
            ocr_text: Texto crudo extraído por OCR

        Returns:
            Dict con: monto, moneda, fecha, numero_operacion, banco, documento
        """

        prompt = f"""Analiza el siguiente texto extraído de un comprobante o voucher de pago peruano.

TEXTO DEL COMPROBANTE:
{ocr_text}

Extrae la siguiente información y responde ÚNICAMENTE con un JSON válido (sin markdown, sin explicaciones):

{{
    "monto": <número decimal del monto pagado, o null si no se encuentra>,
    "moneda": "PEN" o "USD",
    "fecha": "<fecha en el formato que aparezca, o null>",
    "numero_operacion": "<número de operación/transacción/referencia, o null>",
    "banco": "<banco o app: Yape, Plin, BCP, BBVA, Interbank, Scotiabank, etc., o null>",
    "documento": "<DNI o RUC SOLO si aparece con etiqueta explícita como 'DNI:', 'DOC:', 'RUC:', o null>"
}}

REGLAS IMPORTANTES:
1. El monto puede estar oculto con asteriscos (ej: S/****500.00), extrae solo la parte numérica visible
2. El número de operación suele ser un código de 8-15 dígitos
3. NO confundas números de cuenta o teléfonos con el número de operación
4. Solo extrae "documento" si dice EXPLÍCITAMENTE "DNI", "DOC", "Documento" o "RUC" seguido del número
5. Si no estás seguro de un campo, pon null
6. Responde SOLO el JSON, nada más"""

        try:
            logger.info(f"Enviando texto a Gemma para análisis ({len(ocr_text)} chars)...")

            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            logger.info(f"Respuesta de Gemma: {response_text[:300]}...")

            # Limpiar respuesta (quitar markdown si lo hay)
            if "```" in response_text:
                # Extraer contenido entre ```
                parts = response_text.split("```")
                for part in parts:
                    if part.strip().startswith("json"):
                        response_text = part.strip()[4:].strip()
                        break
                    elif part.strip().startswith("{"):
                        response_text = part.strip()
                        break

            # Parsear JSON
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Error parseando JSON: {e}")
                logger.error(f"Respuesta raw: {response_text}")

                # Intentar extraer JSON de la respuesta
                import re
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    # Retornar valores vacíos si no se puede parsear
                    logger.error("No se pudo extraer JSON de la respuesta")
                    return {
                        "monto": None,
                        "moneda": "PEN",
                        "fecha": None,
                        "numero_operacion": None,
                        "banco": None,
                        "documento": None,
                        "error": "No se pudo parsear respuesta de Gemma"
                    }

            # Asegurar que monto sea float
            if data.get("monto") is not None:
                try:
                    data["monto"] = float(data["monto"])
                except (ValueError, TypeError):
                    data["monto"] = None

            logger.info(f"Datos extraídos por Gemma: monto={data.get('monto')}, banco={data.get('banco')}, op={data.get('numero_operacion')}")

            return data

        except Exception as e:
            logger.error(f"Error en Gemma: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "monto": None,
                "moneda": "PEN",
                "fecha": None,
                "numero_operacion": None,
                "banco": None,
                "documento": None,
                "error": str(e)
            }


# Singleton instance
gemma_service = GemmaAnalyzerService()
