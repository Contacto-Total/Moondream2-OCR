import re
import time
import base64
import logging
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.models.schemas import ExtractedVoucherData

logger = logging.getLogger(__name__)


class MoondreamService:
    """Servicio para analizar imagenes usando Moondream2"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "vikhyatk/moondream2"
        self.revision = "2024-08-26"  # Revision estable

    def load_model(self) -> bool:
        """Carga el modelo Moondream2 en memoria"""
        try:
            logger.info(f"Cargando modelo Moondream2 en {self.device}...")
            start_time = time.time()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=True,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                device_map={"": self.device}
            )

            load_time = time.time() - start_time
            logger.info(f"Modelo cargado exitosamente en {load_time:.2f} segundos")
            return True

        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            return False

    def is_loaded(self) -> bool:
        """Verifica si el modelo esta cargado"""
        return self.model is not None and self.tokenizer is not None

    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decodifica una imagen en base64 a PIL Image"""
        # Remover el prefijo data:image/... si existe
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))

        # Convertir a RGB si es necesario
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _ask_model(self, image: Image.Image, question: str) -> str:
        """Hace una pregunta al modelo sobre la imagen"""
        enc_image = self.model.encode_image(image)
        answer = self.model.answer_question(enc_image, question, self.tokenizer)
        return answer.strip()

    def _extract_amount(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extrae monto y moneda del texto"""
        # Patrones para diferentes formatos de monto
        patterns = [
            # S/ 150.00, S/. 150.00, S/.150.00
            r'S/\.?\s*(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
            # PEN 150.00
            r'PEN\s*(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
            # $ 150.00, USD 150.00
            r'(?:\$|USD)\s*(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
            # 150.00 soles
            r'(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)\s*(?:soles|SOLES)',
            # Monto: 150.00
            r'[Mm]onto[:\s]+(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
            # Importe: 150.00
            r'[Ii]mporte[:\s]+(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
            # Total: 150.00
            r'[Tt]otal[:\s]+(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
        ]

        moneda = "PEN"  # Default

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(",", "").replace(" ", "")
                try:
                    amount = float(amount_str)
                    # Detectar moneda
                    if "$" in text or "USD" in text.upper():
                        moneda = "USD"
                    return amount, moneda
                except ValueError:
                    continue

        return None, None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extrae fecha del texto"""
        patterns = [
            # DD/MM/YYYY o DD-MM-YYYY
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            # YYYY-MM-DD
            r'(\d{4}-\d{2}-\d{2})',
            # DD de Mes de YYYY
            r'(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_operation_number(self, text: str) -> Optional[str]:
        """Extrae numero de operacion del texto"""
        patterns = [
            # N° Operacion: 123456
            r'[Nn][°o]?\s*(?:[Oo]peraci[oó]n|[Tt]ransacci[oó]n)[:\s]*(\d+)',
            # Referencia: 123456
            r'[Rr]eferencia[:\s]*(\d+)',
            # Codigo: 123456
            r'[Cc][oó]digo[:\s]*(\d+)',
            # Op: 123456
            r'Op[:\s]*(\d+)',
            # Nro: 123456
            r'Nro[:\s]*(\d+)',
            # Secuencia larga de numeros (posible numero de operacion)
            r'\b(\d{6,20})\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    def _detect_bank(self, text: str) -> Optional[str]:
        """Detecta el banco del texto"""
        banks = [
            "BCP", "BBVA", "Interbank", "Scotiabank", "BanBif",
            "Banco de la Nacion", "Banco de Credito", "Yape", "Plin",
            "Banco Continental", "Caja", "Financiera"
        ]

        text_upper = text.upper()
        for bank in banks:
            if bank.upper() in text_upper:
                return bank

        return None

    def analyze_voucher(self, image_base64: str) -> Tuple[ExtractedVoucherData, int]:
        """
        Analiza un voucher/boleta de pago

        Returns:
            Tuple[ExtractedVoucherData, int]: Datos extraidos y tiempo de procesamiento en ms
        """
        start_time = time.time()

        try:
            # Decodificar imagen
            image = self._decode_base64_image(image_base64)

            # Pregunta principal para extraer informacion del voucher
            prompt = """Analyze this payment voucher/receipt image. Extract and list:
1. Total amount (monto)
2. Date of transaction
3. Operation/transaction number
4. Bank or payment method name

Be specific and include the exact numbers and text you see."""

            # Obtener respuesta del modelo
            response = self._ask_model(image, prompt)
            logger.info(f"Respuesta del modelo: {response}")

            # Segunda pregunta especifica para el monto si no se encontro
            amount_response = self._ask_model(image, "What is the exact total amount shown in this receipt? Include currency symbol.")
            logger.info(f"Respuesta de monto: {amount_response}")

            # Combinar respuestas para extraccion
            full_text = f"{response}\n{amount_response}"

            # Extraer datos estructurados
            monto, moneda = self._extract_amount(full_text)
            fecha = self._extract_date(full_text)
            numero_operacion = self._extract_operation_number(full_text)
            banco = self._detect_bank(full_text)

            processing_time = int((time.time() - start_time) * 1000)

            return ExtractedVoucherData(
                monto=monto,
                moneda=moneda,
                fecha=fecha,
                numero_operacion=numero_operacion,
                banco=banco,
                texto_completo=full_text
            ), processing_time

        except Exception as e:
            logger.error(f"Error analizando voucher: {str(e)}")
            processing_time = int((time.time() - start_time) * 1000)
            return ExtractedVoucherData(texto_completo=f"Error: {str(e)}"), processing_time


# Singleton instance
moondream_service = MoondreamService()
