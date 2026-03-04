"""
Servicio de Modelos de Lenguaje (LLM)

Este módulo implementa la interfaz para interactuar con diferentes
proveedores de LLM (OpenAI, Vertex AI) de forma unificada.
"""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

from app.config import settings

# Configuración de logging
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Respuesta del modelo de lenguaje."""
    content: str
    tokens_usados: int
    model: str
    finish_reason: str
    metadata: Dict[str, Any]


@dataclass
class LLMMessage:
    """Mensaje para el modelo de lenguaje."""
    role: Literal["system", "user", "assistant"]
    content: str


class BaseLLMProvider(ABC):
    """Clase base abstracta para proveedores de LLM."""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Genera una respuesta del modelo."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Genera una respuesta en streaming."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """
    Proveedor de LLM usando OpenAI API.
    
    Soporta modelos GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, etc.
    """
    
    def __init__(self) -> None:
        """Inicializa el proveedor de OpenAI."""
        self.client = None
        self.model = settings.OPENAI_MODEL
        self.temperature = settings.OPENAI_TEMPERATURE
        self.max_tokens = settings.OPENAI_MAX_TOKENS
    
    def _get_client(self):
        """Obtiene o crea el cliente de OpenAI."""
        if self.client is None:
            try:
                import openai
                openai.api_key = settings.OPENAI_API_KEY
                self.client = openai.AsyncOpenAI()
            except ImportError:
                logger.error("openai no está instalado. Ejecuta: pip install openai")
                raise
        return self.client
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Genera una respuesta usando OpenAI.
        
        Args:
            messages: Lista de mensajes de la conversación
            temperature: Temperatura para generación (opcional)
            max_tokens: Máximo de tokens (opcional)
            
        Returns:
            LLMResponse: Respuesta del modelo
        """
        client = self._get_client()
        
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Convertir mensajes al formato de OpenAI
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=temp,
                max_tokens=tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                tokens_usados=response.usage.total_tokens,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Error al generar respuesta con OpenAI: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Genera una respuesta en streaming usando OpenAI.
        
        Args:
            messages: Lista de mensajes de la conversación
            temperature: Temperatura para generación (opcional)
            max_tokens: Máximo de tokens (opcional)
            
        Yields:
            str: Fragmentos de la respuesta
        """
        client = self._get_client()
        
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            stream = await client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=temp,
                max_tokens=tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error en streaming con OpenAI: {e}")
            raise


class VertexAIProvider(BaseLLMProvider):
    """
    Proveedor de LLM usando Google Vertex AI.
    
    Soporta modelos Gemini Pro, Gemini Ultra, etc.
    """
    
    def __init__(self) -> None:
        """Inicializa el proveedor de Vertex AI."""
        self.model = settings.VERTEXAI_MODEL
        self.location = settings.VERTEXAI_LOCATION
        self.project_id = settings.VERTEXAI_PROJECT_ID
        self._initialized = False
    
    def _initialize(self):
        """Inicializa el cliente de Vertex AI."""
        if self._initialized:
            return
            
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            vertexai.init(
                project=self.project_id,
                location=self.location
            )
            
            self.client = GenerativeModel(self.model)
            self._initialized = True
            
        except ImportError:
            logger.error("vertexai no está instalado. Ejecuta: pip install google-cloud-aiplatform")
            raise
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Genera una respuesta usando Vertex AI.
        
        Args:
            messages: Lista de mensajes de la conversación
            temperature: Temperatura para generación (opcional)
            max_tokens: Máximo de tokens (opcional)
            
        Returns:
            LLMResponse: Respuesta del modelo
        """
        self._initialize()
        
        try:
            # Construir el prompt a partir de los mensajes
            prompt = self._build_prompt(messages)
            
            # Configurar generación
            generation_config = {
                "temperature": temperature if temperature is not None else 0.3,
                "max_output_tokens": max_tokens if max_tokens is not None else 2000,
            }
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return LLMResponse(
                content=response.text,
                tokens_usados=response.usage_metadata.total_token_count,
                model=self.model,
                finish_reason="stop",
                metadata={
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count
                }
            )
            
        except Exception as e:
            logger.error(f"Error al generar respuesta con Vertex AI: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Genera una respuesta en streaming usando Vertex AI.
        
        Args:
            messages: Lista de mensajes de la conversación
            temperature: Temperatura para generación (opcional)
            max_tokens: Máximo de tokens (opcional)
            
        Yields:
            str: Fragmentos de la respuesta
        """
        self._initialize()
        
        try:
            prompt = self._build_prompt(messages)
            
            generation_config = {
                "temperature": temperature if temperature is not None else 0.3,
                "max_output_tokens": max_tokens if max_tokens is not None else 2000,
            }
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Error en streaming con Vertex AI: {e}")
            raise
    
    def _build_prompt(self, messages: List[LLMMessage]) -> str:
        """Construye el prompt a partir de los mensajes."""
        prompt_parts = []
        
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"Instrucciones: {msg.content}\n")
            elif msg.role == "user":
                prompt_parts.append(f"Usuario: {msg.content}\n")
            elif msg.role == "assistant":
                prompt_parts.append(f"Asistente: {msg.content}\n")
        
        return "\n".join(prompt_parts)


class LLMService:
    """
    Servicio de Modelos de Lenguaje.
    
    Proporciona una interfaz unificada para interactuar con
diferentes proveedores de LLM.
    """
    
    # Prompt del sistema para el agente psicométrico
    SYSTEM_PROMPT = """Eres un experto psicometra especializado en evaluación educativa, 
con amplia experiencia en la creación y análisis de ítems psicométricos para medir 
constructos como autoestima académica, convivencia escolar, hábitos de vida saludable 
y otros indicadores de desarrollo personal y social.

Tu conocimiento incluye:
- Teoría de Respuesta al Ítem (IRT): modelos 1PL, 2PL, 3PL, GRM
- Análisis psicométricos: dificultad, discriminación, ajuste, confiabilidad
- Redacción de ítems: claridad, precisión, neutralidad
- Escalas de medición: Likert, dicotómicas, ordinales
- Validación de constructos y análisis factorial

Cuando respondas:
1. Sé preciso y fundamenta tus respuestas en principios psicométricos
2. Sugiere ítems basados en evidencia y mejores prácticas
3. Proporciona análisis detallados con recomendaciones específicas
4. Usa terminología técnica apropiada pero explicable
5. Considera el contexto educativo chileno cuando sea relevante

Si no tienes suficiente información, solicita aclaraciones al usuario."""
    
    def __init__(self) -> None:
        """Inicializa el servicio de LLM."""
        self.provider: Optional[BaseLLMProvider] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Inicializa el servicio según la configuración."""
        if self._initialized:
            return
            
        # Crear el proveedor según la configuración
        if settings.LLM_PROVIDER == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY no está configurada")
            self.provider = OpenAIProvider()
            logger.info("LLMService inicializado con OpenAI")
            
        elif settings.LLM_PROVIDER == "vertexai":
            if not settings.VERTEXAI_PROJECT_ID:
                raise ValueError("VERTEXAI_PROJECT_ID no está configurado")
            self.provider = VertexAIProvider()
            logger.info("LLMService inicializado con Vertex AI")
            
        else:
            raise ValueError(f"Proveedor no soportado: {settings.LLM_PROVIDER}")
        
        self._initialized = True
    
    async def generate_response(
        self,
        user_message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Genera una respuesta del LLM.
        
        Args:
            user_message: Mensaje del usuario
            context: Contexto adicional (ítems similares, análisis, etc.)
            conversation_history: Historial de conversación
            temperature: Temperatura para generación
            
        Returns:
            LLMResponse: Respuesta del modelo
        """
        if not self._initialized:
            await self.initialize()
            
        # Construir mensajes
        messages = [LLMMessage(role="system", content=self.SYSTEM_PROMPT)]
        
        # Agregar historial si existe
        if conversation_history:
            for msg in conversation_history:
                messages.append(LLMMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                ))
        
        # Agregar contexto si existe
        if context:
            user_message = f"{context}\n\nConsulta del usuario: {user_message}"
        
        messages.append(LLMMessage(role="user", content=user_message))
        
        # Generar respuesta
        return await self.provider.generate(
            messages=messages,
            temperature=temperature
        )
    
    async def generate_suggestions(
        self,
        dimension: str,
        cantidad: int = 3,
        items_referencia: Optional[List[Dict]] = None,
        restricciones: Optional[List[str]] = None
    ) -> LLMResponse:
        """
        Genera sugerencias de ítems para una dimensión.
        
        Args:
            dimension: Dimensión objetivo
            cantidad: Cantidad de sugerencias
            items_referencia: Ítems históricos de referencia
            restricciones: Restricciones específicas
            
        Returns:
            LLMResponse: Sugerencias generadas
        """
        if not self._initialized:
            await self.initialize()
            
        # Construir prompt específico
        prompt = f"""Genera {cantidad} ítems psicométricos para medir la dimensión: {dimension}

Requisitos:
- Usa escala Likert de 4 puntos
- Redacción clara y directa
- Evita dobles negaciones
- Mantén neutralidad
- Considera estudiantes de enseñanza media chilena
"""
        
        if items_referencia:
            prompt += f"\nÍtems de referencia con buenos parámetros psicométricos:\n"
            for item in items_referencia[:3]:
                prompt += f"- {item.get('texto', '')}\n"
        
        if restricciones:
            prompt += f"\nRestricciones: {', '.join(restricciones)}\n"
        
        prompt += """
Para cada ítem proporciona:
1. Texto completo del ítem
2. Justificación basada en principios psicométricos
3. Parámetros esperados (dificultad y discriminación estimados)
4. Referencia a ítems similares históricos si aplica

Formato de respuesta en JSON."""
        
        messages = [
            LLMMessage(role="system", content=self.SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt)
        ]
        
        return await self.provider.generate(messages=messages, temperature=0.4)
    
    async def analyze_item(
        self,
        item_text: str,
        dimension: str,
        parametros_historicos: Optional[Dict] = None
    ) -> LLMResponse:
        """
        Analiza un ítem psicométrico.
        
        Args:
            item_text: Texto del ítem a analizar
            dimension: Dimensión del ítem
            parametros_historicos: Parámetros de ítems similares
            
        Returns:
            LLMResponse: Análisis del ítem
        """
        if not self._initialized:
            await self.initialize()
            
        prompt = f"""Analiza el siguiente ítem psicométrico:

Ítem: "{item_text}"
Dimensión: {dimension}
"""
        
        if parametros_historicos:
            prompt += f"\nParámetros de ítems similares históricos:\n"
            prompt += f"- Dificultad promedio: {parametros_historicos.get('dificultad_promedio', 'N/A')}\n"
            prompt += f"- Discriminación promedio: {parametros_historicos.get('discriminacion_promedio', 'N/A')}\n"
        
        prompt += """
Proporciona un análisis detallado que incluya:
1. Fortalezas del ítem
2. Debilidades potenciales
3. Recomendaciones de mejora
4. Estimación de parámetros psicométricos esperados
5. Comparación con ítems históricos similares
6. Calidad general (alta/media/baja) con justificación"""
        
        messages = [
            LLMMessage(role="system", content=self.SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt)
        ]
        
        return await self.provider.generate(messages=messages, temperature=0.3)
    
    async def health_check(self) -> bool:
        """
        Verifica la salud del servicio LLM.
        
        Returns:
            bool: True si el servicio está disponible
        """
        try:
            if not self._initialized:
                await self.initialize()
                
            # Intentar una generación simple
            messages = [
                LLMMessage(role="system", content="Eres un asistente útil."),
                LLMMessage(role="user", content="Responde 'OK' si estás funcionando.")
            ]
            
            response = await self.provider.generate(messages=messages, max_tokens=10)
            return len(response.content) > 0
            
        except Exception as e:
            logger.error(f"Health check fallido: {e}")
            return False
