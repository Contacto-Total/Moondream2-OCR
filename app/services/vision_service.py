import re
import time
import base64
import logging
import threading
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel

from app.models.schemas import ExtractedVoucherData, ValidationResult

logger = logging.getLogger(__name__)

# Tiempo en segundos antes de descargar el modelo por inactividad
UNLOAD_TIMEOUT = 300  # 5 minutos


class InternVL2Service:
    """Servicio para analizar imagenes usando InternVL2-1B con carga perezosa"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "OpenGVLab/InternVL2-1B"
        self.model_name = "InternVL2-1B"
        self._lock = threading.Lock()
        self._last_used = None
        self._unload_timer = None
        self._lazy_mode = True  # Activar carga perezosa

    def load_model(self) -> bool:
        """Carga el modelo InternVL2-1B en memoria"""
        with self._lock:
            if self.model is not None:
                return True  # Ya está cargado

            try:
                logger.info(f"Cargando modelo {self.model_name} en {self.device}...")
                start_time = time.time()

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True
                )

                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).eval()

                if self.device == "cuda":
                    self.model = self.model.cuda()

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
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            # Limpiar memoria
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        return self.model is not None and self.tokenizer is not None

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

    def _chat(self, image: Image.Image, question: str) -> str:
        """Hace una pregunta al modelo sobre la imagen"""
        try:
            # Preparar imagen para el modelo
            pixel_values = self._process_image(image)

            # Generar respuesta usando el método chat del modelo
            generation_config = dict(max_new_tokens=512, do_sample=False)

            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config
            )

            return response.strip()
        except Exception as e:
            logger.error(f"Error en chat: {str(e)}")
            # Fallback: intentar método alternativo
            return self._chat_fallback(image, question)

    def _process_image(self, image: Image.Image):
        """Procesa la imagen para el modelo"""
        # Redimensionar si es muy grande
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        # Usar el procesador del modelo si está disponible
        if hasattr(self.model, 'process_images'):
            return self.model.process_images([image], self.model.config)

        # Fallback: procesar manualmente
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(image).unsqueeze(0)

        if self.device == "cuda":
            pixel_values = pixel_values.cuda()

        return pixel_values

    def _chat_fallback(self, image: Image.Image, question: str) -> str:
        """Método alternativo si el chat principal falla"""
        try:
            pixel_values = self._process_image(image)

            # Construir prompt
            prompt = f"<image>\n{question}"

            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    pixel_values=pixel_values,
                    max_new_tokens=512,
                    do_sample=False
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            logger.error(f"Error en fallback: {str(e)}")
            return f"Error: {str(e)}"

    def _extract_amount(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extrae monto y moneda del texto"""
        patterns = [
            r'S/\.?\s*(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
            r'PEN\s*(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
            r'(?:\$|USD)\s*(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)\s*(?:soles|SOLES)',
            r'[Mm]onto[:\s]+S?/?\.?\s*(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
            r'[Tt]otal[:\s]+S?/?\.?\s*(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
            r'[Ii]mporte[:\s]+S?/?\.?\s*(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)',
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
            r'\b(\d{8})\b',  # DNI peruano
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
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

        # Carga perezosa - cargar modelo solo cuando se necesita
        if not self._ensure_loaded():
            raise Exception("No se pudo cargar el modelo")

        try:
            image = self._decode_base64_image(image_base64)

            # Prompt para extraer informacion del voucher
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = """Analiza esta imagen de un comprobante/voucher de pago y extrae la siguiente información:
1. Monto total (incluye el símbolo de moneda)
2. Fecha de la transacción
3. Número de operación o referencia
4. Nombre del banco o servicio (Yape, Plin, BCP, etc.)
5. DNI o documento del pagador si aparece

Responde de forma estructurada con cada dato encontrado."""

            # Obtener respuesta del modelo
            response = self._chat(image, prompt)
            logger.info(f"Respuesta del modelo: {response}")

            # Extraer datos estructurados del texto
            monto, moneda = self._extract_amount(response)
            fecha = self._extract_date(response)
            numero_operacion = self._extract_operation_number(response)
            banco = self._detect_bank(response)
            documento = self._extract_document(response)

            # Si no se encontró monto, hacer pregunta específica
            if monto is None:
                amount_response = self._chat(image, "¿Cuál es el monto total exacto de este voucher? Responde solo con el número y la moneda.")
                logger.info(f"Respuesta de monto: {amount_response}")
                monto, moneda = self._extract_amount(amount_response)
                response += f"\n{amount_response}"

            processing_time = int((time.time() - start_time) * 1000)

            data = ExtractedVoucherData(
                monto=monto,
                moneda=moneda,
                fecha=fecha,
                numero_operacion=numero_operacion,
                banco=banco,
                documento=documento,
                texto_completo=response
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
        tolerance = 1.00  # 1 sol de tolerancia
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
vision_service = InternVL2Service()
