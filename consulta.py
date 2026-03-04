"""
Schemas Pydantic para Consultas al Agente de IA

Define los modelos de validación para las consultas realizadas
al agente de IA psicométrico.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class FiltrosBusqueda(BaseModel):
    """
    Filtros opcionales para búsquedas de ítems.
    
    Permite refinar las búsquedas por dimensiones, años,
    parámetros psicométricos y otros criterios.
    """
    dimensiones: Optional[List[int]] = Field(
        None,
        description="IDs de dimensiones a incluir"
    )
    años: Optional[List[int]] = Field(
        None,
        description="Años de creación o aplicación"
    )
    solo_activos: bool = Field(
        default=True,
        description="Solo incluir ítems activos"
    )
    dificultad_min: Optional[float] = Field(
        None,
        ge=-5,
        le=5,
        description="Dificultad mínima (parámetro b IRT)"
    )
    dificultad_max: Optional[float] = Field(
        None,
        ge=-5,
        le=5,
        description="Dificultad máxima (parámetro b IRT)"
    )
    discriminacion_min: Optional[float] = Field(
        None,
        ge=0,
        description="Discriminación mínima (parámetro a IRT)"
    )
    formato_respuesta: Optional[str] = Field(
        None,
        description="Filtrar por formato de respuesta"
    )
    
    class Config:
        """Configuración del schema."""
        json_schema_extra = {
            "example": {
                "dimensiones": [1, 2],
                "años": [2020, 2021, 2022],
                "solo_activos": True,
                "dificultad_min": -1.0,
                "dificultad_max": 1.0,
                "discriminacion_min": 0.5
            }
        }


class ContextoConsulta(BaseModel):
    """
    Contexto adicional para la consulta al agente.
    
    Proporciona información sobre el propósito de la consulta,
    el perfil del usuario y restricciones específicas.
    """
    proposito: Optional[str] = Field(
        None,
        description="Propósito de la consulta (generación, análisis, búsqueda)"
    )
    perfil_usuario: Optional[str] = Field(
        None,
        description="Perfil del usuario (psicómetro, docente, investigador)"
    )
    nivel_educativo: Optional[str] = Field(
        None,
        description="Nivel educativo objetivo (básica, media, universitaria)"
    )
    poblacion_objetivo: Optional[str] = Field(
        None,
        description="Descripción de la población objetivo"
    )
    restricciones: Optional[List[str]] = Field(
        None,
        description="Restricciones o requisitos específicos"
    )
    
    class Config:
        """Configuración del schema."""
        json_schema_extra = {
            "example": {
                "proposito": "generar_nuevos_items",
                "perfil_usuario": "psicometra",
                "nivel_educativo": "media",
                "poblacion_objetivo": "estudiantes de 14-16 años"
            }
        }


class ConsultaRequest(BaseModel):
    """
    Schema para consultas al agente de IA.
    
    Representa una consulta completa que incluye el mensaje del usuario,
    filtros de búsqueda opcionales y contexto adicional.
    """
    consulta: str = Field(
        ...,
        min_length=5,
        max_length=5000,
        description="Texto de la consulta del usuario"
    )
    session_id: Optional[str] = Field(
        None,
        description="ID de sesión para mantener contexto conversacional"
    )
    filtros: Optional[FiltrosBusqueda] = Field(
        None,
        description="Filtros para refinar la búsqueda"
    )
    contexto: Optional[ContextoConsulta] = Field(
        None,
        description="Contexto adicional de la consulta"
    )
    incluir_items_similares: bool = Field(
        default=True,
        description="Incluir ítems históricos similares en la respuesta"
    )
    max_resultados: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Máximo de resultados a incluir"
    )
    
    class Config:
        """Configuración del schema."""
        json_schema_extra = {
            "example": {
                "consulta": "Necesito ítems sobre autoestima académica para estudiantes de enseñanza media",
                "session_id": "sess_12345",
                "filtros": {
                    "dimensiones": [1],
                    "solo_activos": True,
                    "dificultad_min": -1.0,
                    "dificultad_max": 1.0
                },
                "contexto": {
                    "proposito": "generar_nuevos_items",
                    "perfil_usuario": "psicometra",
                    "nivel_educativo": "media"
                },
                "incluir_items_similares": True,
                "max_resultados": 5
            }
        }


class ConsultaHistorial(BaseModel):
    """
    Schema para historial de consultas en una sesión.
    
    Permite mantener el contexto conversacional entre consultas.
    """
    session_id: str = Field(..., description="ID de la sesión")
    consultas: List[dict] = Field(default=[], description="Lista de consultas previas")
    timestamp_inicio: str = Field(..., description="Timestamp de inicio de sesión")
    
    class Config:
        """Configuración del schema."""
        json_schema_extra = {
            "example": {
                "session_id": "sess_12345",
                "consultas": [
                    {"rol": "usuario", "mensaje": "Busco ítems sobre autoestima"},
                    {"rol": "agente", "mensaje": "Encontré 3 ítems relevantes..."}
                ],
                "timestamp_inicio": "2024-01-15T10:30:00Z"
            }
        }
