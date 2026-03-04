"""
Excepciones Personalizadas

Define excepciones específicas para el agente de IA psicométrico,
permitiendo un manejo de errores más granular e informativo.
"""

from typing import Optional, Dict, Any


class AgentePsicometricoError(Exception):
    """
    Excepción base para el agente de IA psicométrico.
    
    Todas las excepciones específicas del agente heredan de esta clase.
    """
    
    def __init__(
        self,
        message: str,
        code: str = "AGENTE_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        """
        Inicializa la excepción.
        
        Args:
            message: Mensaje descriptivo del error
            code: Código de error único
            details: Detalles adicionales del error
            status_code: Código HTTP asociado
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.status_code = status_code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la excepción a diccionario."""
        return {
            "error": True,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "status_code": self.status_code
        }


class LLMError(AgentePsicometricoError):
    """
    Error relacionado con el servicio de LLM.
    
    Se lanza cuando hay problemas con la comunicación o respuesta
de los modelos de lenguaje.
    """
    
    def __init__(
        self,
        message: str = "Error en el servicio de lenguaje",
        code: str = "LLM_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=503
        )


class LLMTimeoutError(LLMError):
    """Error cuando el LLM excede el tiempo de respuesta."""
    
    def __init__(self, timeout_seconds: float):
        super().__init__(
            message=f"El servicio de lenguaje tardó más de {timeout_seconds}s en responder",
            code="LLM_TIMEOUT",
            details={"timeout_seconds": timeout_seconds},
            status_code=504
        )


class LLMRateLimitError(LLMError):
    """Error cuando se excede el límite de peticiones al LLM."""
    
    def __init__(self, retry_after: Optional[int] = None):
        super().__init__(
            message="Se ha excedido el límite de peticiones al servicio de lenguaje",
            code="LLM_RATE_LIMIT",
            details={"retry_after_seconds": retry_after},
            status_code=429
        )


class VectorDBError(AgentePsicometricoError):
    """
    Error relacionado con la base de datos vectorial.
    
    Se lanza cuando hay problemas con Pinecone, ChromaDB u otro
    proveedor de base de datos vectorial.
    """
    
    def __init__(
        self,
        message: str = "Error en la base de datos vectorial",
        code: str = "VECTOR_DB_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=503
        )


class VectorDBConnectionError(VectorDBError):
    """Error de conexión a la base de datos vectorial."""
    
    def __init__(self, provider: str):
        super().__init__(
            message=f"No se pudo conectar a la base de datos vectorial ({provider})",
            code="VECTOR_DB_CONNECTION_ERROR",
            details={"provider": provider},
            status_code=503
        )


class DatabaseError(AgentePsicometricoError):
    """
    Error relacionado con la base de datos PostgreSQL.
    
    Se lanza cuando hay problemas con consultas o conexiones
    a la base de datos relacional.
    """
    
    def __init__(
        self,
        message: str = "Error en la base de datos",
        code: str = "DATABASE_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=500
        )


class DatabaseConnectionError(DatabaseError):
    """Error de conexión a la base de datos."""
    
    def __init__(self):
        super().__init__(
            message="No se pudo conectar a la base de datos",
            code="DATABASE_CONNECTION_ERROR",
            status_code=503
        )


class DatabaseNotFoundError(DatabaseError):
    """Error cuando no se encuentra un recurso en la base de datos."""
    
    def __init__(self, resource: str, resource_id: Any):
        super().__init__(
            message=f"{resource} con id {resource_id} no encontrado",
            code="DATABASE_NOT_FOUND",
            details={"resource": resource, "id": resource_id},
            status_code=404
        )


class ConsultaInvalidaError(AgentePsicometricoError):
    """
    Error cuando la consulta del usuario es inválida.
    
    Se lanza cuando la consulta no cumple con los requisitos
    mínimos de validez.
    """
    
    def __init__(
        self,
        message: str = "La consulta proporcionada no es válida",
        code: str = "CONSULTA_INVALIDA",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=400
        )


class ValidacionError(AgentePsicometricoError):
    """
    Error de validación de datos.
    
    Se lanza cuando los datos proporcionados no pasan las
    validaciones requeridas.
    """
    
    def __init__(
        self,
        message: str = "Error de validación de datos",
        field: Optional[str] = None,
        code: str = "VALIDACION_ERROR"
    ):
        details = {"field": field} if field else {}
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=400
        )


class AutenticacionError(AgentePsicometricoError):
    """Error de autenticación."""
    
    def __init__(self, message: str = "Error de autenticación"):
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=401
        )


class AutorizacionError(AgentePsicometricoError):
    """Error de autorización (permisos insuficientes)."""
    
    def __init__(self, message: str = "No tiene permisos para realizar esta acción"):
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            status_code=403
        )


class ConfiguracionError(AgentePsicometricoError):
    """
    Error de configuración.
    
    Se lanza cuando hay problemas con la configuración del sistema.
    """
    
    def __init__(
        self,
        message: str = "Error en la configuración del sistema",
        config_key: Optional[str] = None
    ):
        details = {"config_key": config_key} if config_key else {}
        super().__init__(
            message=message,
            code="CONFIGURACION_ERROR",
            details=details,
            status_code=500
        )
