"""
Schemas Pydantic para Respuestas del Agente de IA

Define los modelos de validación para las respuestas generadas
por el agente de IA psicométrico.
"""

from datetime import datetime
from typing import Optional, List, Any, Literal
from pydantic import BaseModel, Field


class SugerenciaItem(BaseModel):
    """
    Schema para una sugerencia de ítem generada por el agente.
    
    Contiene el texto propuesto, opciones de respuesta sugeridas
    y justificación basada en ítems históricos.
    """
    texto_propuesto: str = Field(..., description="Texto del ítem sugerido")
    dimension_sugerida: Optional[str] = Field(None, description="Dimensión sugerida")
    formato_respuesta: str = Field(..., description="Formato de respuesta recomendado")
    opciones_respuesta: List[dict[str, Any]] = Field(
        default=[],
        description="Opciones de respuesta sugeridas"
    )
    justificacion: str = Field(..., description="Justificación de la sugerencia")
    items_referencia: List[int] = Field(
        default=[],
        description="IDs de ítems históricos usados como referencia"
    )
    parametros_esperados: Optional[dict[str, float]] = Field(
        None,
        description="Parámetros psicométricos esperados (dificultad, discriminación)"
    )
    
    class Config:
        """Configuración del schema."""
        json_schema_extra = {
            "example": {
                "texto_propuesto": "Me siento capaz de resolver los problemas de matemáticas",
                "dimension_sugerida": "Autoestima Académica",
                "formato_respuesta": "Likert 4 puntos",
                "opciones_respuesta": [
                    {"texto": "Muy de acuerdo", "valor": 4},
                    {"texto": "De acuerdo", "valor": 3},
                    {"texto": "En desacuerdo", "valor": 2},
                    {"texto": "Muy en desacuerdo", "valor": 1}
                ],
                "justificacion": "Basado en ítems históricos con buenos parámetros psicométricos...",
                "items_referencia": [101, 102, 103],
                "parametros_esperados": {
                    "dificultad": 0.2,
                    "discriminacion": 1.2
                }
            }
        }


class ResultadoBusqueda(BaseModel):
    """
    Schema para un resultado de búsqueda de ítems.
    
    Incluye información del ítem encontrado y metadatos
    de la búsqueda (score de similitud, relevancia).
    """
    item_id: int = Field(..., description="ID del ítem")
    texto_item: str = Field(..., description="Texto del ítem")
    dimension: str = Field(..., description="Nombre de la dimensión")
    año_creacion: int = Field(..., description="Año de creación")
    formato_respuesta: str = Field(..., description="Formato de respuesta")
    score_similitud: float = Field(
        ...,
        ge=0,
        le=1,
        description="Score de similitud semántica"
    )
    parametros_psicometricos: Optional[dict[str, Any]] = Field(
        None,
        description="Parámetros psicométricos del ítem"
    )
    activo: bool = Field(..., description="Si el ítem está activo")
    
    class Config:
        """Configuración del schema."""
        json_schema_extra = {
            "example": {
                "item_id": 101,
                "texto_item": "Me siento seguro al responder preguntas en clase",
                "dimension": "Autoestima Académica",
                "año_creacion": 2022,
                "formato_respuesta": "Likert 4 puntos",
                "score_similitud": 0.89,
                "parametros_psicometricos": {
                    "dificultad": 0.15,
                    "discriminacion": 1.1,
                    "ajuste": "bueno"
                },
                "activo": True
            }
        }


class AnalisisItem(BaseModel):
    """
    Schema para análisis de un ítem realizado por el agente.
    
    Contiene la evaluación psicométrica, recomendaciones
    y comparación con ítems similares.
    """
    item_id: int = Field(..., description="ID del ítem analizado")
    texto_item: str = Field(..., description="Texto del ítem")
    fortalezas: List[str] = Field(default=[], description="Fortalezas identificadas")
    debilidades: List[str] = Field(default=[], description="Debilidades identificadas")
    recomendaciones: List[str] = Field(default=[], description="Recomendaciones de mejora")
    comparacion_historica: Optional[str] = Field(
        None,
        description="Comparación con ítems históricos similares"
    )
    calidad_psicometrica: Optional[str] = Field(
        None,
        description="Evaluación general de calidad (alta, media, baja)"
    )
    sugerencias_redaccion: Optional[List[str]] = Field(
        None,
        description="Sugerencias para mejorar la redacción"
    )
    
    class Config:
        """Configuración del schema."""
        json_schema_extra = {
            "example": {
                "item_id": 101,
                "texto_item": "Me siento seguro al responder preguntas en clase",
                "fortalezas": [
                    "Redacción clara y directa",
                    "Buena discriminación histórica"
                ],
                "debilidades": [
                    "Dificultad ligeramente baja para el nivel objetivo"
                ],
                "recomendaciones": [
                    "Considerar aumentar la complejidad léxica"
                ],
                "comparacion_historica": "Similar a ítems con buen desempeño...",
                "calidad_psicometrica": "alta",
                "sugerencias_redaccion": [
                    "Podría especificarse el contexto académico"
                ]
            }
        }


class RespuestaAgente(BaseModel):
    """
    Schema principal para respuestas del agente de IA.
    
    Contiene el mensaje generado, metadatos de la respuesta
    y datos estructurados según el tipo de consulta.
    """
    # Información básica
    mensaje: str = Field(..., description="Respuesta textual del agente")
    tipo_respuesta: Literal[
        "busqueda",
        "generacion",
        "analisis",
        "informacion",
        "error"
    ] = Field(..., description="Tipo de respuesta generada")
    
    # Identificación
    session_id: Optional[str] = Field(None, description="ID de sesión")
    consulta_id: str = Field(..., description="ID único de la consulta")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    
    # Datos estructurados según tipo
    items_encontrados: Optional[List[ResultadoBusqueda]] = Field(
        None,
        description="Resultados de búsqueda (tipo=busqueda)"
    )
    sugerencias: Optional[List[SugerenciaItem]] = Field(
        None,
        description="Sugerencias de ítems (tipo=generacion)"
    )
    analisis: Optional[AnalisisItem] = Field(
        None,
        description="Análisis de ítem (tipo=analisis)"
    )
    
    # Metadatos
    tokens_usados: Optional[int] = Field(None, description="Tokens utilizados en la respuesta")
    tiempo_procesamiento_ms: Optional[int] = Field(
        None,
        description="Tiempo de procesamiento en milisegundos"
    )
    fuentes_consultadas: Optional[List[str]] = Field(
        None,
        description="Fuentes de datos consultadas"
    )
    confianza: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Nivel de confianza de la respuesta"
    )
    
    # Información adicional
    notas: Optional[str] = Field(None, description="Notas o advertencias")
    sugerencias_consulta: Optional[List[str]] = Field(
        None,
        description="Sugerencias de consultas relacionadas"
    )
    
    class Config:
        """Configuración del schema."""
        json_schema_extra = {
            "example": {
                "mensaje": "Encontré 5 ítems relevantes sobre autoestima académica...",
                "tipo_respuesta": "busqueda",
                "session_id": "sess_12345",
                "consulta_id": "query_67890",
                "timestamp": "2024-01-15T10:30:00Z",
                "items_encontrados": [],
                "tokens_usados": 450,
                "tiempo_procesamiento_ms": 1250,
                "fuentes_consultadas": ["base_datos_vectorial", "postgresql"],
                "confianza": 0.92
            }
        }


class EstadoAgente(BaseModel):
    """
    Schema para el estado del agente de IA.
    
    Proporciona información sobre la salud y disponibilidad
    de los componentes del agente.
    """
    estado: Literal["operativo", "degradado", "no_disponible"] = Field(
        ...,
        description="Estado general del agente"
    )
    version: str = Field(..., description="Versión del agente")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    
    # Estado de componentes
    base_datos: Literal["conectada", "desconectada", "error"] = Field(
        ...,
        description="Estado de la conexión a PostgreSQL"
    )
    base_datos_vectorial: Literal["conectada", "desconectada", "error"] = Field(
        ...,
        description="Estado de la base de datos vectorial"
    )
    servicio_llm: Literal["disponible", "no_disponible", "error"] = Field(
        ...,
        description="Estado del servicio de LLM"
    )
    
    # Métricas
    total_items_indexados: Optional[int] = Field(
        None,
        description="Total de ítems en la base de datos"
    )
    total_dimensiones: Optional[int] = Field(
        None,
        description="Total de dimensiones"
    )
    consultas_hoy: Optional[int] = Field(
        None,
        description="Consultas procesadas hoy"
    )
    
    # Información adicional
    mensaje: Optional[str] = Field(None, description="Mensaje informativo")
    
    class Config:
        """Configuración del schema."""
        json_schema_extra = {
            "example": {
                "estado": "operativo",
                "version": "0.1.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "base_datos": "conectada",
                "base_datos_vectorial": "conectada",
                "servicio_llm": "disponible",
                "total_items_indexados": 1500,
                "total_dimensiones": 8,
                "consultas_hoy": 45
            }
        }


class ErrorRespuesta(BaseModel):
    """
    Schema para respuestas de error del agente.
    
    Estandariza el formato de errores para facilitar
    el manejo en el cliente.
    """
    error: bool = Field(default=True, description="Indica que es un error")
    codigo: str = Field(..., description="Código de error")
    mensaje: str = Field(..., description="Mensaje de error")
    detalles: Optional[str] = Field(None, description="Detalles técnicos del error")
    consulta_id: Optional[str] = Field(None, description="ID de la consulta que falló")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    
    class Config:
        """Configuración del schema."""
        json_schema_extra = {
            "example": {
                "error": True,
                "codigo": "LLM_TIMEOUT",
                "mensaje": "El servicio de IA tardó demasiado en responder",
                "detalles": "Timeout después de 30 segundos",
                "consulta_id": "query_67890",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
