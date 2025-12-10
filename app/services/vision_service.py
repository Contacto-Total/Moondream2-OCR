import re
import sys
import time
import base64
import logging
import threading
from io import BytesIO
from typing import Optional, Tuple

# Crear módulo fake de flash_attn ANTES de importar transformers
# Florence-2 verifica si flash_attn existe pero no lo necesitamos en CPU
import types
from importlib.machinery import ModuleSpec

# Crear módulo principal
flash_attn_fake = types.ModuleType("flash_attn")
flash_attn_fake.__spec__ = ModuleSpec("flash_attn", None)
flash_attn_fake.__version__ = "2.5.0"
flash_attn_fake.flash_attn_func = lambda *args, **kwargs: None
flash_attn_fake.flash_attn_varlen_func = lambda *args, **kwargs: None

# Crear submódulo flash_attn_interface
flash_attn_interface = types.ModuleType("flash_attn.flash_attn_interface")
flash_attn_interface.__spec__ = ModuleSpec("flash_attn.flash_attn_interface", None)
flash_attn_interface.flash_attn_func = lambda *args, **kwargs: None
flash_attn_interface.flash_attn_varlen_func = lambda *args, **kwargs: None

sys.modules["flash_attn"] = flash_attn_fake
sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface

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
        """Extrae monto y moneda del texto - busca el monto más relevante"""

        # Patrones prioritarios (apps de pago y totales)
        priority_patterns = [
            r'[Pp]agaste\s+S/\.?\s*([\d,]+\.?\d*)',
            r'[Ee]nviaste\s+S/\.?\s*([\d,]+\.?\d*)',
            r'[Rr]ecibiste\s+S/\.?\s*([\d,]+\.?\d*)',
            r'[Mm]onto\s+S/\.?\s*([\d,]+\.?\d*)',
            r'[Tt]otal\s*:?\s*S/\.?\s*([\d,]+\.?\d*)',
            r'[Ii]mporte\s*:?\s*S/\.?\s*([\d,]+\.?\d*)',
            r'[Vv]alor\s*:?\s*S/\.?\s*([\d,]+\.?\d*)',
        ]

        # Patrones generales
        general_patterns = [
            r'S/\.?\s*([\d,]+\.?\d+)',
            r'([\d,]+\.?\d+)\s*[Ss]oles',
            r'PEN\s*([\d,]+\.?\d+)',
        ]

        moneda = "PEN"
        if "$" in text or "USD" in text.upper():
            moneda = "USD"

        # Primero buscar patrones prioritarios
        for pattern in priority_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(",", "").replace(" ", "")
                try:
                    amount = float(amount_str)
                    if amount > 0:
                        return amount, moneda
                except ValueError:
                    continue

        # Si no hay prioritarios, buscar todos los montos y elegir el más grande
        all_amounts = []
        for pattern in general_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match_str in matches:
                try:
                    amount_str = match_str.replace(",", "").replace(" ", "")
                    amount = float(amount_str)
                    if amount >= 1.0:  # Ignorar decimales sueltos
                        all_amounts.append(amount)
                except ValueError:
                    continue

        if all_amounts:
            # Elegir el monto más grande (típicamente el total)
            return max(all_amounts), moneda

        return None, None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extrae fecha del texto - soporta múltiples formatos"""

        # Meses en español para detectar fechas escritas
        meses = r'(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)'

        patterns = [
            # Formatos con separadores: 10/12/2025, 10-12-2025
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2})',
            # Formato ISO: 2025-12-10
            r'(\d{4}-\d{2}-\d{2})',
            # Formato con mes escrito: 10 de diciembre de 2025, 10 diciembre 2025
            r'(\d{1,2}\s+de\s+' + meses + r'\s+de\s+\d{4})',
            r'(\d{1,2}\s+de\s+' + meses + r'\s+\d{4})',
            r'(\d{1,2}\s+' + meses + r'\s+\d{4})',
            # Formato: diciembre 10, 2025
            r'(' + meses + r'\s+\d{1,2},?\s+\d{4})',
            # Fecha de emisión/operación específica
            r'[Ff]echa[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'[Ff]echa\s+de\s+[Oo]peraci[oó]n[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'[Ff]echa\s+de\s+[Ee]misi[oó]n[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_operation_number(self, text: str) -> Optional[str]:
        """Extrae numero de operacion del texto"""
        logger.info(f"Buscando número de operación en: {text[:500]}...")

        patterns = [
            # Patrones específicos con etiqueta (prioridad alta)
            r'[Nn][úu]mero\s+de\s+[Oo]peraci[oó]n[:\s]*(\d+)',  # "Número de operación: 123"
            r'[Nn][°º]?\s*[Oo]peraci[oó]n[:\s]*(\d+)',  # "N° Operación: 123"
            r'[Nn][°º]?\s*[Tt]ransacci[oó]n[:\s]*(\d+)',
            r'[Cc][oó]digo\s+de\s+[Oo]peraci[oó]n[:\s]*(\d+)',
            r'[Oo]peraci[oó]n[:\s]*[Nn][°º]?\s*(\d+)',  # "Operación N°: 123"
            r'[Oo]peraci[oó]n[:\s]+(\d+)',  # "Operación: 123" o "Operación 123"
            r'[Rr]eferencia[:\s]*(\d+)',
            r'[Cc][oó]digo[:\s]*(\d+)',
            r'[Cc]onstancia[:\s]*[Nn]?[°º]?\s*(\d+)',  # "Constancia N°: 123"
            r'[Cc]omprobante[:\s]*(\d+)',
            r'Op\.?[:\s]*(\d+)',
            r'Nro\.?[:\s]*(\d+)',
            r'N[°º]\s*(\d{6,})',
            r'ID[:\s]*(\d{6,})',
            # Patrones de Yape específicos
            r'[Oo]p[:\.\s]+(\d{8,})',  # "Op: 12345678" o "Op. 12345678"
            r'[Nn][\s°\.]*[Oo]p[\s\.]*(\d{6,})',  # "N Op 123456" variantes
            # Números que siguen a texto común de comprobantes
            r'(?:yape|plin|transferencia|pago)[^\d]*(\d{8,15})',
            # Números largos aislados (8-20 dígitos) - última prioridad
            r'\b(\d{8,20})\b',
        ]

        # Primero buscar patrones con etiqueta explícita (alta confianza)
        high_confidence_patterns = patterns[:14]  # Los primeros 14 son patrones con etiqueta
        for pattern in high_confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                num = match.group(1)
                logger.info(f"Patrón alta confianza '{pattern}' encontró: {num}")
                if len(num) >= 6:
                    logger.info(f"Número de operación encontrado (alta confianza): {num}")
                    return num

        # Buscar patrones de baja confianza (números largos sin etiqueta)
        low_confidence_patterns = patterns[14:]
        for pattern in low_confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                num = match.group(1)
                logger.info(f"Patrón baja confianza '{pattern}' encontró: {num}")
                # Para patrones sin etiqueta, ser más estricto
                # Ignorar números que parecen DNI SOLO si no hay contexto de operación
                if len(num) == 8 and num[0] in '01234567':
                    # Verificar si hay contexto de "operación" cerca del número
                    context_patterns = ['operaci', 'transac', 'comprobante', 'constancia', 'referencia', 'yape', 'plin']
                    text_lower = text.lower()
                    has_context = any(ctx in text_lower for ctx in context_patterns)
                    if not has_context:
                        logger.info(f"Ignorando {num} - parece DNI sin contexto de operación")
                        continue
                # Ignorar números muy cortos
                if len(num) < 6:
                    logger.info(f"Ignorando {num} - muy corto")
                    continue
                logger.info(f"Número de operación encontrado (baja confianza): {num}")
                return num

        logger.warning("No se encontró número de operación")
        return None

    def _extract_document(self, text: str) -> Optional[str]:
        """Extrae documento/DNI del texto"""
        patterns = [
            # Patrones específicos con etiqueta
            r'DNI[:\s]*(\d{8})',
            r'DOC(?:UMENTO)?[:\s]*(\d{8})',
            r'RUC[:\s]*(\d{11})',
            r'[Nn][°º]?\s*[Dd]ocumento[:\s]*(\d{8})',
            r'[Ii]dentificaci[oó]n[:\s]*(\d{8})',
            r'[Cc][eé]dula[:\s]*(\d{8,10})',
            # DNI peruano: 8 dígitos empezando con 0-7
            r'\b([0-7]\d{7})\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc = match.group(1)
                # Validar que sea un documento válido
                if len(doc) == 8 and doc[0] in '01234567':
                    return doc
                elif len(doc) == 11:  # RUC
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
