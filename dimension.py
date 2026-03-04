"""
Modelo SQLAlchemy para Dimensiones Psicométricas

Representa las dimensiones o constructos psicométricos que agrupan ítems,
como "Autoestima Académica", "Convivencia Escolar", etc.
"""

from typing import TYPE_CHECKING, List

from sqlalchemy import Column, String, Text
from sqlalchemy.orm import relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.item import Item


class Dimension(Base):
    """
    Modelo que representa una dimensión psicométrica.
    
    Las dimensiones son constructos teóricos que agrupan ítems relacionados
    conceptualmente. Por ejemplo, "Autoestima Académica" es una dimensión
    que puede contener múltiples ítems relacionados.
    
    Attributes:
        id: Identificador único de la dimensión
        nombre: Nombre único de la dimensión (ej: "Autoestima Académica")
        descripcion: Descripción detallada de lo que mide la dimensión
        items: Relación con los ítems asociados a esta dimensión
    """
    
    # ============================================
    # Columnas específicas
    # ============================================
    nombre = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="Nombre de la dimensión (ej: 'Autoestima Académica')"
    )
    
    descripcion = Column(
        Text,
        nullable=True,
        comment="Descripción detallada de lo que mide la dimensión"
    )
    
    # ============================================
    # Relaciones
    # ============================================
    items: List["Item"] = relationship(
        "Item",
        back_populates="dimension",
        cascade="all, delete-orphan",
        lazy="selectin",
        doc="Ítems asociados a esta dimensión"
    )
    
    def __init__(self, nombre: str, descripcion: str | None = None) -> None:
        """
        Inicializa una nueva dimensión.
        
        Args:
            nombre: Nombre de la dimensión
            descripcion: Descripción opcional de la dimensión
        """
        self.nombre = nombre
        self.descripcion = descripcion
    
    def __str__(self) -> str:
        """Representación en string de la dimensión."""
        return f"Dimension(id={self.id}, nombre='{self.nombre}')"
    
    @property
    def cantidad_items(self) -> int:
        """
        Retorna la cantidad de ítems asociados a esta dimensión.
        
        Returns:
            int: Número de ítems en la dimensión
        """
        return len(self.items) if self.items else 0
    
    @property
    def items_activos(self) -> List["Item"]:
        """
        Retorna solo los ítems activos de esta dimensión.
        
        Returns:
            List[Item]: Lista de ítems activos
        """
        if not self.items:
            return []
        return [item for item in self.items if item.activo]
