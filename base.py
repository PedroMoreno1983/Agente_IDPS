"""
Modelo Base para SQLAlchemy

Define la clase base para todos los modelos ORM del sistema.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.orm import declarative_base, declared_attr


class BaseModel:
    """
    Clase base para todos los modelos SQLAlchemy.
    
    Proporciona:
    - Tablename automático basado en el nombre de la clase
    - Campos de auditoría (created_at, updated_at)
    - Métodos utilitarios para serialización
    """
    
    # Genera automáticamente el nombre de la tabla
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Genera el nombre de la tabla en plural y minúsculas."""
        return cls.__name__.lower() + "s"
    
    # Campos de auditoría
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convierte el modelo a un diccionario.
        
        Returns:
            dict: Representación del modelo como diccionario
        """
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def __repr__(self) -> str:
        """Representación en string del modelo."""
        attrs = [f"{key}={value}" for key, value in self.to_dict().items()]
        return f"<{self.__class__.__name__}({', '.join(attrs)})>"


# Crear la clase base declarativa
Base = declarative_base(cls=BaseModel)
