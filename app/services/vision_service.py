import re
import time
import base64
import logging
import threading
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from app.models.schemas import ExtractedVoucherData, ValidationResult

logger = logging.getLogger(__name__)

# Tiempo en segundos antes de descargar el modelo por inactividad
UNLOAD_TIMEOUT = 300  # 5 minutos


class Florence2Service:
    """Servicio para analizar imagenes usando Florence-2-base (~1GB RAM)"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"  # Florence-2 funciona bien en CPU
        self.model_id = "microsoft/Florence-2-base"
        self.model_name = "Florence-2-base"
        self._lock = threading.Lock()
        self._last_used = None
        self._unload_timer = None
        self._lazy_mode = True

    def load_model(self) -> bool:
        """Carga el modelo Florence-2-base en memoria"""
        with self._lock:
            if self.model is not None:
                return True

            try:
                logger.info(f"Cargando modelo {self.model_name}...")
                start_time = time.time()

                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=True
                )

                # Desactivar flash attention para CPU
                import os
                os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    _attn_implementation="eager"
                ).eval()

                load_time = time.time() - start_time
                logger.info(f"Modelo cargado exitosamente en {load_time:.2f} segundos")
                self._last_used = time.time()
                self._schedule_unload()
                return True

            except Exception as e:
                logger.error(f"Error cargando modelo: {str(e)}")
                return False

    def unload_model(self) -> None:
        """Descarga el modelo de memoria para liberar RAM"""
        with self._lock:
            if self.model is None:
                return

            logger.info("Descargando modelo para liberar memoria...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None

            import gc
            gc.collect()

            logger.info("Modelo descargado. RAM liberada.")

    def _schedule_unload(self) -> None:
        """Programa la descarga del modelo después del timeout"""
        if self._unload_timer:
            self._unload_timer.cancel()

        self._unload_timer = threading.Timer(UNLOAD_TIMEOUT, self._check_and_unload)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def _check_and_unload(self) -> None:
        """Verifica si debe descargar el modelo por inactividad"""
        if self._last_used and (time.time() - self._last_used) >= UNLOAD_TIMEOUT:
            self.unload_model()

    def _ensure_loaded(self) -> bool:
        """Asegura que el modelo esté cargado (carga perezosa)"""
        if not self.is_loaded():
            return self.load_model()
        self._last_used = time.time()
        self._schedule_unload()
        return True

    def is_loaded(self) -> bool:
        """Verifica si el modelo esta cargado"""
        return self.model is not None and self.processor is not None

    def is_lazy_mode(self) -> bool:
        """Verifica si está en modo lazy loading"""
        return self._lazy_mode

    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decodifica una imagen en base64 a PIL Image"""
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _run_florence(self, image: Image.Image, task: str, text_input: str = "") -> str:
        """Ejecuta Florence-2 con una tarea específica"""
        try:
            prompt = f"<{task}>" if not text_input else f"<{task}>{text_input}"

            # Asegurar que la imagen esté en RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )

            # Debug: ver qué keys tiene inputs
            logger.info(f"Inputs keys: {inputs.keys()}")

            # Manejar diferentes nombres de keys según la versión
            pixel_key = "pixel_values" if "pixel_values" in inputs else "flattened_patches"

            if pixel_key not in inputs or inputs[pixel_key] is None:
                logger.error(f"No se encontró pixel_values en inputs. Keys disponibles: {list(inputs.keys())}")
                return ""

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs[pixel_key],
                    max_new_tokens=1024,
                    num_beams=1,  # Florence-2 tiene bug con num_beams > 1
                    do_sample=False
                )

            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]

            # Parsear respuesta de Florence
            parsed = self.processor.post_process_generation(
                generated_text,
                task=f"<{task}>",
                image_size=(image.width, image.height)
            )

            result = parsed.get(f"<{task}>", str(parsed))
            logger.info(f"Florence result for {task}: {result[:200] if result else 'empty'}")
            return result

        except Exception as e:
            logger.error(f"Error en Florence-2: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def _extract_text_ocr(self, image: Image.Image) -> str:
        """Extrae todo el texto de la imagen usando OCR"""
        return self._run_florence(image, "OCR")

    def _get_detailed_caption(self, image: Image.Image) -> str:
        """Obtiene una descripción detallada de la imagen"""
        return self._run_florence(image, "MORE_DETAILED_CAPTION")

    def _extract_amount(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extrae monto y moneda del texto"""
        patterns = [
            r'S/\.?\s*([\d,]+\.?\d*)',
            r'PEN\s*([\d,]+\.?\d*)',
            r'(?:\$|USD)\s*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*(?:soles|SOLES)',
            r'[Mm]onto[:\s]+S?/?\\.?\s*([\d,]+\.?\d*)',
            r'[Tt]otal[:\s]+S?/?\\.?\s*([\d,]+\.?\d*)',
            r'[Ii]mporte[:\s]+S?/?\\.?\s*([\d,]+\.?\d*)',
            r'[Pp]agaste\s+S/\s*([\d,]+\.?\d*)',
            r'[Ee]nviaste\s+S/\s*([\d,]+\.?\d*)',
        ]

        moneda = "PEN"

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(",", "").replace(" ", "")
                try:
                    amount = float(amount_str)
                    if "$" in text or "USD" in text.upper():
                        moneda = "USD"
                    return amount, moneda
                except ValueError:
                    continue

        return None, None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extrae fecha del texto"""
        patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})',
            r'(\d{1,2}\s+\w+\s+\d{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_operation_number(self, text: str) -> Optional[str]:
        """Extrae numero de operacion del texto"""
        patterns = [
            r'[Nn][°o]?\s*(?:[Oo]peraci[oó]n|[Tt]ransacci[oó]n)[:\s]*(\d+)',
            r'[Rr]eferencia[:\s]*(\d+)',
            r'[Cc][oó]digo[:\s]*(\d+)',
            r'Op[:\s]*(\d+)',
            r'Nro[:\s]*(\d+)',
            r'N[°o]\s*(\d{6,})',
            r'\b(\d{8,20})\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    def _extract_document(self, text: str) -> Optional[str]:
        """Extrae documento/DNI del texto"""
        patterns = [
            r'(?:DNI|DOC|DOCUMENTO|RUC)[:\s]*(\d{8,11})',
            r'\b(\d{8})\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc = match.group(1)
                if len(doc) >= 8:
                    return doc

        return None

    def _detect_bank(self, text: str) -> Optional[str]:
        """Detecta el banco del texto"""
        banks = [
            ("YAPE", "Yape"),
            ("PLIN", "Plin"),
            ("BCP", "BCP"),
            ("BBVA", "BBVA"),
            ("INTERBANK", "Interbank"),
            ("SCOTIABANK", "Scotiabank"),
            ("BANBIF", "BanBif"),
            ("BANCO DE LA NACION", "Banco de la Nacion"),
            ("BANCO DE CREDITO", "BCP"),
            ("NEQUI", "Nequi"),
            ("TUNKI", "Tunki"),
        ]

        text_upper = text.upper()
        for keyword, bank_name in banks:
            if keyword in text_upper:
                return bank_name

        return None

    def analyze_voucher(
        self,
        image_base64: str,
        expected_amount: Optional[float] = None,
        expected_document: Optional[str] = None,
        custom_prompt: Optional[str] = None
    ) -> Tuple[ExtractedVoucherData, Optional[ValidationResult], Optional[ValidationResult], int]:
        """
        Analiza un voucher/boleta de pago

        Returns:
            Tuple con: datos extraidos, validacion monto, validacion documento, tiempo en ms
        """
        start_time = time.time()

        if not self._ensure_loaded():
            raise Exception("No se pudo cargar el modelo")

        try:
            image = self._decode_base64_image(image_base64)

            # Extraer texto con OCR
            ocr_text = self._extract_text_ocr(image)
            logger.info(f"OCR extraído: {ocr_text}")

            # También obtener descripción para más contexto
            caption = self._get_detailed_caption(image)
            logger.info(f"Caption: {caption}")

            # Combinar textos para análisis
            combined_text = f"{ocr_text} {caption}"

            # Extraer datos estructurados
            monto, moneda = self._extract_amount(combined_text)
            fecha = self._extract_date(combined_text)
            numero_operacion = self._extract_operation_number(combined_text)
            banco = self._detect_bank(combined_text)
            documento = self._extract_document(combined_text)

            processing_time = int((time.time() - start_time) * 1000)

            data = ExtractedVoucherData(
                monto=monto,
                moneda=moneda,
                fecha=fecha,
                numero_operacion=numero_operacion,
                banco=banco,
                documento=documento,
                texto_completo=ocr_text
            )

            # Validar monto si se proporcionó
            validacion_monto = None
            if expected_amount is not None and monto is not None:
                validacion_monto = self._validate_amount(expected_amount, monto)

            # Validar documento si se proporcionó
            validacion_documento = None
            if expected_document is not None and documento is not None:
                validacion_documento = self._validate_document(expected_document, documento)

            return data, validacion_monto, validacion_documento, processing_time

        except Exception as e:
            logger.error(f"Error analizando voucher: {str(e)}")
            processing_time = int((time.time() - start_time) * 1000)
            return ExtractedVoucherData(texto_completo=f"Error: {str(e)}"), None, None, processing_time

    def _validate_amount(self, expected: float, extracted: float) -> ValidationResult:
        """Valida si el monto extraído coincide con el esperado"""
        tolerance = 1.00
        difference = abs(extracted - expected)
        matches = difference <= tolerance

        if matches:
            mensaje = "El monto coincide"
        elif extracted > expected:
            mensaje = "El monto del comprobante es MAYOR al esperado"
        else:
            mensaje = "El monto del comprobante es MENOR al esperado"

        return ValidationResult(
            coincide=matches,
            valor_esperado=str(expected),
            valor_extraido=str(extracted),
            diferencia=round(difference, 2),
            mensaje=mensaje
        )

    def _validate_document(self, expected: str, extracted: str) -> ValidationResult:
        """Valida si el documento extraído coincide con el esperado"""
        expected_clean = re.sub(r'[^0-9]', '', expected)
        extracted_clean = re.sub(r'[^0-9]', '', extracted)

        matches = expected_clean == extracted_clean

        return ValidationResult(
            coincide=matches,
            valor_esperado=expected,
            valor_extraido=extracted,
            mensaje="El documento coincide" if matches else "El documento NO coincide"
        )


# Singleton instance
vision_service = Florence2Service()
