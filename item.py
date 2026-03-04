"""
Schemas Pydantic para Ítems y entidades relacionadas

Define los modelos de validación para la creación, actualización
y respuesta de ítems psicométricos y sus relaciones.
"""

from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field, ConfigDict


# ============================================
# Schemas para Dimensiones
# ============================================

class DimensionBase(BaseModel):
    """Schema base para dimensiones."""
    nombre: str = Field(..., min_length=1, max_length=255, description="Nombre de la dimensión")
    descripcion: Optional[str] = Field(None, description="Descripción de la dimensión")


class DimensionCreate(DimensionBase):
    """Schema para crear una nueva dimensión."""
    pass


class DimensionResponse(DimensionBase):
    """Schema para respuesta de dimensión."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    created_at: datetime
    updated_at: datetime
    cantidad_items: int = Field(0, description="Cantidad de ítems en la dimensión")


# ============================================
# Schemas para Opciones de Respuesta
# ============================================

class OpcionRespuestaBase(BaseModel):
    """Schema base para opciones de respuesta."""
    texto_opcion: str = Field(..., min_length=1, max_length=255, description="Texto de la opción")
    valor_numerico: int = Field(..., description="Valor numérico para análisis")


class OpcionRespuestaCreate(OpcionRespuestaBase):
    """Schema para crear una opción de respuesta."""
    pass


class OpcionRespuestaResponse(OpcionRespuestaBase):
    """Schema para respuesta de opción de respuesta."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    item_id: int


# ============================================
# Schemas para Análisis Psicométricos
# ============================================

class AnalisisPsicometricoBase(BaseModel):
    """Schema base para análisis psicométricos."""
    año_aplicacion: int = Field(..., ge=2000, le=2100, description="Año de aplicación")
    modelo_irt: str = Field(default="2PL", description="Modelo IRT utilizado")
    param_a_discriminacion: Optional[float] = Field(None, ge=0, description="Parámetro de discriminación")
    param_b_dificultad: Optional[float] = Field(None, ge=-5, le=5, description="Parámetro de dificultad")
    param_c_adivinanza: Optional[float] = Field(None, ge=0, le=1, description="Parámetro de adivinanza")
    ajuste_chi_cuadrado: Optional[float] = Field(None, ge=0, description="Estadístico de ajuste")
    p_valor_ajuste: Optional[float] = Field(None, ge=0, le=1, description="Valor p del ajuste")
    confiabilidad_test_retest: Optional[float] = Field(None, ge=0, le=1, description="Confiabilidad test-retest")
    otros_indicadores: Optional[dict[str, Any]] = Field(None, description="Indicadores adicionales")


class AnalisisPsicometricoCreate(AnalisisPsicometricoBase):
    """Schema para crear un análisis psicométrico."""
    item_id: int = Field(..., description="ID del ítem analizado")


class AnalisisPsicometricoResponse(AnalisisPsicometricoBase):
    """Schema para respuesta de análisis psicométrico."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    item_id: int
    created_at: datetime
    updated_at: datetime
    tiene_buen_ajuste: Optional[bool] = Field(None, description="Indica si tiene buen ajuste")
    calidad_discriminacion: Optional[str] = Field(None, description="Calidad de discriminación")
    nivel_dificultad: Optional[str] = Field(None, description="Nivel de dificultad")


# ============================================
# Schemas para Ítems
# ============================================

class ItemBase(BaseModel):
    """Schema base para ítems psicométricos."""
    texto_item: str = Field(..., min_length=5, description="Texto del ítem")
    formato_respuesta: str = Field(
        default="Likert 4 puntos",
        description="Formato de respuesta (Likert 4 puntos, Dicotómico, etc.)"
    )
    año_creacion: int = Field(..., ge=2000, le=2100, description="Año de creación")
    activo: bool = Field(default=True, description="Si el ítem está activo")


class ItemCreate(ItemBase):
    """Schema para crear un nuevo ítem."""
    dimension_id: int = Field(..., description="ID de la dimensión")
    opciones_respuesta: Optional[List[OpcionRespuestaCreate]] = Field(
        None,
        description="Opciones de respuesta del ítem"
    )


class ItemUpdate(BaseModel):
    """Schema para actualizar un ítem existente."""
    texto_item: Optional[str] = Field(None, min_length=5, description="Texto del ítem")
    formato_respuesta: Optional[str] = Field(None, description="Formato de respuesta")
    año_creacion: Optional[int] = Field(None, ge=2000, le=2100, description="Año de creación")
    activo: Optional[bool] = Field(None, description="Si el ítem está activo")
    dimension_id: Optional[int] = Field(None, description="ID de la dimensión")


class ItemResponse(ItemBase):
    """Schema para respuesta completa de ítem."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    dimension_id: int
    embedding_id: Optional[str] = Field(None, description="ID del embedding")
    created_at: datetime
    updated_at: datetime
    
    # Relaciones
    dimension: Optional[DimensionResponse] = Field(None, description="Dimensión asociada")
    opciones_respuesta: List[OpcionRespuestaResponse] = Field(
        default=[],
        description="Opciones de respuesta"
    )
    analisis_psicometricos: List[AnalisisPsicometricoResponse] = Field(
        default=[],
        description="Análisis psicométricos"
    )
    
    # Propiedades calculadas
    tiene_embedding: bool = Field(False, description="Si tiene embedding asociado")
    ultimo_analisis: Optional[AnalisisPsicometricoResponse] = Field(
        None,
        description="Último análisis psicométrico"
    )


class ItemSearchResult(BaseModel):
    """Schema para resultado de búsqueda de ítems."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    texto_item: str
    formato_respuesta: str
    año_creacion: int
    activo: bool
    dimension: Optional[DimensionResponse] = None
    
    # Campos de búsqueda vectorial
    score_similitud: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Score de similitud (para búsquedas vectoriales)"
    )
    
    # Resumen del último análisis
    dificultad: Optional[float] = Field(None, description="Dificultad del último análisis")
    discriminacion: Optional[float] = Field(None, description="Discriminación del último análisis")


class ItemListResponse(BaseModel):
    """Schema para lista paginada de ítems."""
    items: List[ItemResponse]
    total: int = Field(..., description="Total de ítems")
    page: int = Field(..., description="Página actual")
    page_size: int = Field(..., description="Tamaño de página")
    total_pages: int = Field(..., description="Total de páginas")
