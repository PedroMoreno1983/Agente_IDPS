"""
Modelo SQLAlchemy para Análisis Psicométricos

Representa los análisis estadísticos realizados sobre los ítems,
incluyendo parámetros IRT (Teoría de Respuesta al Ítem) e índices de ajuste.
"""

from typing import TYPE_CHECKING, Optional, Any

from sqlalchemy import Column, ForeignKey, Integer, String, Float, JSON
from sqlalchemy.orm import relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.item import Item


class AnalisisPsicometrico(Base):
    """
    Modelo que representa un análisis psicométrico de un ítem.
    
    Almacena los resultados de análisis estadísticos aplicados a un ítem,
    incluyendo parámetros del modelo IRT (discriminación, dificultad, adivinanza)
    e índices de ajuste del modelo.
    
    Attributes:
        id: Identificador único del análisis
        item_id: FK al ítem analizado
        año_aplicacion: Año en que se aplicó el cuestionario
        modelo_irt: Modelo IRT utilizado (2PL, 3PL, GRM, etc.)
        param_a_discriminacion: Parámetro 'a' de discriminación
        param_b_dificultad: Parámetro 'b' de dificultad
        param_c_adivinanza: Parámetro 'c' de adivinanza (opcional)
        ajuste_chi_cuadrado: Valor del estadístico de ajuste
        p_valor_ajuste: Valor p asociado al estadístico de ajuste
        confiabilidad_test_retest: Coeficiente de confiabilidad
        otros_indicadores: JSON con indicadores adicionales
    """
    
    # ============================================
    # Columnas específicas
    # ============================================
    item_id = Column(
        Integer,
        ForeignKey("items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Referencia al ítem analizado"
    )
    
    año_aplicacion = Column(
        Integer,
        nullable=False,
        index=True,
        comment="Año en que se aplicó el cuestionario y se generó este análisis"
    )
    
    modelo_irt = Column(
        String(50),
        nullable=False,
        default="2PL",
        comment="Modelo de Teoría de Respuesta al Ítem (ej: '2PL', '3PL', 'GRM')"
    )
    
    param_a_discriminacion = Column(
        Float,
        nullable=True,
        comment="Parámetro 'a' de discriminación del modelo IRT"
    )
    
    param_b_dificultad = Column(
        Float,
        nullable=True,
        comment="Parámetro 'b' de dificultad del modelo IRT"
    )
    
    param_c_adivinanza = Column(
        Float,
        nullable=True,
        comment="Parámetro 'c' de adivinanza (solo para modelos 3PL)"
    )
    
    ajuste_chi_cuadrado = Column(
        Float,
        nullable=True,
        comment="Valor del estadístico de ajuste (ej: Chi-cuadrado)"
    )
    
    p_valor_ajuste = Column(
        Float,
        nullable=True,
        comment="Valor p asociado al estadístico de ajuste"
    )
    
    confiabilidad_test_retest = Column(
        Float,
        nullable=True,
        comment="Coeficiente de confiabilidad test-retest si se midió"
    )
    
    otros_indicadores = Column(
        JSON,
        nullable=True,
        comment="Campo flexible para almacenar otros indicadores (correlación ítem-test, etc.)"
    )
    
    # ============================================
    # Relaciones
    # ============================================
    item: "Item" = relationship(
        "Item",
        back_populates="analisis_psicometricos",
        lazy="joined",
        doc="Ítem al que pertenece este análisis"
    )
    
    def __init__(
        self,
        item_id: int,
        año_aplicacion: int,
        modelo_irt: str = "2PL",
        param_a_discriminacion: Optional[float] = None,
        param_b_dificultad: Optional[float] = None,
        param_c_adivinanza: Optional[float] = None,
        ajuste_chi_cuadrado: Optional[float] = None,
        p_valor_ajuste: Optional[float] = None,
        confiabilidad_test_retest: Optional[float] = None,
        otros_indicadores: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Inicializa un nuevo análisis psicométrico.
        
        Args:
            item_id: ID del ítem analizado
            año_aplicacion: Año de aplicación del cuestionario
            modelo_irt: Modelo IRT utilizado
            param_a_discriminacion: Parámetro de discriminación
            param_b_dificultad: Parámetro de dificultad
            param_c_adivinanza: Parámetro de adivinanza
            ajuste_chi_cuadrado: Estadístico de ajuste
            p_valor_ajuste: Valor p del ajuste
            confiabilidad_test_retest: Coeficiente de confiabilidad
            otros_indicadores: Diccionario con indicadores adicionales
        """
        self.item_id = item_id
        self.año_aplicacion = año_aplicacion
        self.modelo_irt = modelo_irt
        self.param_a_discriminacion = param_a_discriminacion
        self.param_b_dificultad = param_b_dificultad
        self.param_c_adivinanza = param_c_adivinanza
        self.ajuste_chi_cuadrado = ajuste_chi_cuadrado
        self.p_valor_ajuste = p_valor_ajuste
        self.confiabilidad_test_retest = confiabilidad_test_retest
        self.otros_indicadores = otros_indicadores
    
    def __str__(self) -> str:
        """Representación en string del análisis."""
        return (
            f"AnalisisPsicometrico(id={self.id}, item_id={self.item_id}, "
            f"año={self.año_aplicacion}, modelo={self.modelo_irt})"
        )
    
    @property
    tiene_buen_ajuste(self) -> Optional[bool]:
        """
        Determina si el ítem tiene un buen ajuste al modelo IRT.
        
        Un p-valor > 0.05 generalmente indica buen ajuste.
        
        Returns:
            bool | None: True si tiene buen ajuste, False si no, None si no hay datos
        """
        if self.p_valor_ajuste is None:
            return None
        return self.p_valor_ajuste > 0.05
    
    @property
    calidad_discriminacion(self) -> Optional[str]:
        """
        Clasifica la calidad de la discriminación del ítem.
        
        Returns:
            str | None: Clasificación de la discriminación
        """
        if self.param_a_discriminacion is None:
            return None
        
        a = self.param_a_discriminacion
        if a < 0.5:
            return "baja"
        elif a < 1.0:
            return "moderada"
        else:
            return "alta"
    
    @property
    nivel_dificultad(self) -> Optional[str]:
        """
        Clasifica el nivel de dificultad del ítem.
        
        Returns:
            str | None: Clasificación de la dificultad
        """
        if self.param_b_dificultad is None:
            return None
        
        b = self.param_b_dificultad
        if b < -1.0:
            return "muy_fácil"
        elif b < -0.5:
            return "fácil"
        elif b < 0.5:
            return "medio"
        elif b < 1.0:
            return "difícil"
        else:
            return "muy_difícil"
    
    def to_dict_resumen(self) -> dict[str, Any]:
        """
        Genera un resumen del análisis psicométrico.
        
        Returns:
            dict: Diccionario con información resumida del análisis
        """
        return {
            "año_aplicacion": self.año_aplicacion,
            "modelo_irt": self.modelo_irt,
            "param_a": self.param_a_discriminacion,
            "param_b": self.param_b_dificultad,
            "param_c": self.param_c_adivinanza,
            "ajuste": {
                "chi_cuadrado": self.ajuste_chi_cuadrado,
                "p_valor": self.p_valor_ajuste,
                "buen_ajuste": self.tiene_buen_ajuste
            },
            "calidad_discriminacion": self.calidad_discriminacion,
            "nivel_dificultad": self.nivel_dificultad,
            "confiabilidad_test_retest": self.confiabilidad_test_retest
        }
