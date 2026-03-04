"""
Configuración del Agente de IA Psicométrico

Este módulo define la configuración de la aplicación utilizando Pydantic Settings,
permitiendo la carga de variables de entorno con valores por defecto.
"""

from functools import lru_cache
from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """
    Configuración principal de la aplicación.
    
    Las variables se cargan desde el archivo .env o variables de entorno del sistema.
    """
    
    # ============================================
    # Configuración General de la Aplicación
    # ============================================
    APP_NAME: str = Field(default="Agente Psicométrico IA", description="Nombre de la aplicación")
    APP_VERSION: str = Field(default="0.1.0", description="Versión de la aplicación")
    APP_DESCRIPTION: str = Field(
        default="Agente de IA especializado en psicometría educativa para análisis y generación de ítems",
        description="Descripción de la aplicación"
    )
    DEBUG: bool = Field(default=False, description="Modo debug de la aplicación")
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Entorno de ejecución"
    )
    
    # ============================================
    # Configuración del Servidor
    # ============================================
    HOST: str = Field(default="0.0.0.0", description="Host del servidor")
    PORT: int = Field(default=8000, description="Puerto del servidor")
    WORKERS: int = Field(default=1, description="Número de workers (solo para producción)")
    
    # ============================================
    # Configuración de Base de Datos PostgreSQL
    # ============================================
    DATABASE_URL: str = Field(
        default="postgresql://user:password@localhost:5432/psicometrico",
        description="URL de conexión a PostgreSQL"
    )
    DB_POOL_SIZE: int = Field(default=10, description="Tamaño del pool de conexiones")
    DB_MAX_OVERFLOW: int = Field(default=20, description="Conexiones adicionales permitidas")
    DB_POOL_TIMEOUT: int = Field(default=30, description="Timeout del pool en segundos")
    DB_ECHO: bool = Field(default=False, description="Mostrar queries SQL en logs")
    
    # ============================================
    # Configuración de Base de Datos Vectorial
    # ============================================
    VECTOR_DB_PROVIDER: Literal["pinecone", "chromadb"] = Field(
        default="chromadb",
        description="Proveedor de base de datos vectorial"
    )
    
    # Pinecone
    PINECONE_API_KEY: Optional[str] = Field(default=None, description="API Key de Pinecone")
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None, description="Entorno de Pinecone")
    PINECONE_INDEX_NAME: str = Field(default="items-psicometricos", description="Nombre del índice")
    
    # ChromaDB
    CHROMADB_PATH: str = Field(
        default="./data/chromadb",
        description="Ruta de almacenamiento de ChromaDB"
    )
    CHROMADB_COLLECTION: str = Field(default="items_psicometricos", description="Nombre de la colección")
    
    # ============================================
    # Configuración de Modelos de Lenguaje (LLM)
    # ============================================
    LLM_PROVIDER: Literal["openai", "vertexai"] = Field(
        default="openai",
        description="Proveedor de LLM"
    )
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="API Key de OpenAI")
    OPENAI_MODEL: str = Field(default="gpt-4", description="Modelo de OpenAI a utilizar")
    OPENAI_TEMPERATURE: float = Field(default=0.3, description="Temperatura para generación")
    OPENAI_MAX_TOKENS: int = Field(default=2000, description="Máximo de tokens por respuesta")
    
    # Vertex AI (Google)
    VERTEXAI_PROJECT_ID: Optional[str] = Field(default=None, description="ID del proyecto GCP")
    VERTEXAI_LOCATION: str = Field(default="us-central1", description="Región de Vertex AI")
    VERTEXAI_MODEL: str = Field(default="gemini-pro", description="Modelo de Vertex AI")
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(
        default=None,
        description="Ruta al archivo de credenciales de GCP"
    )
    
    # ============================================
    # Configuración de Embeddings
    # ============================================
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="Modelo de embeddings de OpenAI"
    )
    EMBEDDING_DIMENSION: int = Field(default=1536, description="Dimensión de los vectores")
    EMBEDDING_BATCH_SIZE: int = Field(default=100, description="Tamaño del batch para embeddings")
    
    # ============================================
    # Configuración de Búsqueda Vectorial
    # ============================================
    VECTOR_SEARCH_TOP_K: int = Field(default=10, description="Número de resultados en búsqueda")
    VECTOR_SEARCH_THRESHOLD: float = Field(default=0.7, description="Umbral mínimo de similitud")
    
    # ============================================
    # Configuración de Seguridad
    # ============================================
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        description="Clave secreta para tokens JWT"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Expiración del token en minutos")
    ALGORITHM: str = Field(default="HS256", description="Algoritmo de encriptación JWT")
    
    # ============================================
    # Configuración de Logging
    # ============================================
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Nivel de logging"
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Formato de los logs"
    )
    
    # ============================================
    # Configuración de CORS
    # ============================================
    CORS_ORIGINS: list[str] = Field(
        default=["*"],
        description="Orígenes permitidos para CORS"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, description="Permitir credenciales CORS")
    CORS_ALLOW_METHODS: list[str] = Field(default=["*"], description="Métodos HTTP permitidos")
    CORS_ALLOW_HEADERS: list[str] = Field(default=["*"], description="Headers permitidos")
    
    # ============================================
    # Configuración de Psicometría
    # ============================================
    PSIC_DIFICULTAD_MIN: float = Field(default=-3.0, description="Dificultad mínima IRT")
    PSIC_DIFICULTAD_MAX: float = Field(default=3.0, description="Dificultad máxima IRT")
    PSIC_DISCRIMINACION_MIN: float = Field(default=0.5, description="Discriminación mínima aceptable")
    PSIC_AJUSTE_P_VALOR_MIN: float = Field(default=0.05, description="P-valor mínimo para buen ajuste")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parsea los orígenes CORS desde string o lista."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("CORS_ALLOW_METHODS", pre=True)
    def parse_cors_methods(cls, v):
        """Parsea los métodos CORS desde string o lista."""
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @validator("CORS_ALLOW_HEADERS", pre=True)
    def parse_cors_headers(cls, v):
        """Parsea los headers CORS desde string o lista."""
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v
    
    class Config:
        """Configuración de Pydantic Settings."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Retorna una instancia cacheada de la configuración.
    
    El uso de lru_cache permite que la configuración se cargue una sola vez
    y se reutilice en todas las llamadas posteriores, mejorando el rendimiento.
    
    Returns:
        Settings: Instancia de configuración de la aplicación
    """
    return Settings()


# Instancia global de configuración
settings = get_settings()
