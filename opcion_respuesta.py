"""
Modelo SQLAlchemy para Opciones de Respuesta

Representa las opciones de respuesta asociadas a cada ítem psicométrico,
incluyendo el texto de la opción y su valor numérico.
"""

from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.item import Item


class OpcionRespuesta(Base):
    """
    Modelo que representa una opción de respuesta para un ítem.
    
    Las opciones de respuesta definen las alternativas disponibles para
    responder un ítem, junto con su valor numérico para análisis estadístico.
    Por ejemplo, en una escala Likert: "Muy de acuerdo" = 4, "De acuerdo" = 3, etc.
    
    Attributes:
        id: Identificador único de la opción
        item_id: FK al ítem al que pertenece esta opción
        texto_opcion: El texto de la opción de respuesta
        valor_numerico: El valor numérico asociado para análisis
    """
    
    # ============================================
    # Columnas específicas
    # ============================================
    item_id = Column(
        Integer,
        ForeignKey("items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Referencia al ítem al que pertenece esta opción"
    )
    
    texto_opcion = Column(
        String(255),
        nullable=False,
        comment="El texto de la opción de respuesta (ej: 'Muy de acuerdo')"
    )
    
    valor_numerico = Column(
        Integer,
        nullable=False,
        comment="El valor numérico asociado a la opción para análisis"
    )
    
    # ============================================
    # Relaciones
    # ============================================
    item: "Item" = relationship(
        "Item",
        back_populates="opciones_respuesta",
        lazy="joined",
        doc="Ítem al que pertenece esta opción de respuesta"
    )
    
    def __init__(
        self,
        item_id: int,
        texto_opcion: str,
        valor_numerico: int
    ) -> None:
        """
        Inicializa una nueva opción de respuesta.
        
        Args:
            item_id: ID del ítem asociado
            texto_opcion: Texto descriptivo de la opción
            valor_numerico: Valor numérico para análisis estadístico
        """
        self.item_id = item_id
        self.texto_opcion = texto_opcion
        self.valor_numerico = valor_numerico
    
    def __str__(self) -> str:
        """Representación en string de la opción."""
        return (
            f"OpcionRespuesta(id={self.id}, item_id={self.item_id}, "
            f"texto='{self.texto_opcion}', valor={self.valor_numerico})"
        )
    
    def to_dict_simple(self) -> dict:
        """
        Retorna una representación simplificada de la opción.
        
        Returns:
            dict: Diccionario con texto y valor de la opción
        """
        return {
            "texto": self.texto_opcion,
            "valor": self.valor_numerico
        }
