"""
Configuración de Base de Datos - SQLAlchemy

Este módulo gestiona la conexión a PostgreSQL y proporciona
utilidades para la creación de sesiones y tablas.
"""

import logging
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from app.config import settings

# Configuración de logging
logger = logging.getLogger(__name__)


# ============================================
# Configuración de URLs de Base de Datos
# ============================================

def get_async_database_url() -> str:
    """
    Convierte la URL de base de datos a formato async (asyncpg).
    
    Returns:
        str: URL de conexión async para SQLAlchemy
    """
    url = settings.DATABASE_URL
    # Reemplazar el driver si es necesario
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def get_sync_database_url() -> str:
    """
    Obtiene la URL de base de datos en formato sincrónico.
    
    Returns:
        str: URL de conexión sincrónica para SQLAlchemy
    """
    url = settings.DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


# ============================================
# Creación de Engines
# ============================================

# Engine async para operaciones principales
async_engine = create_async_engine(
    get_async_database_url(),
    poolclass=QueuePool,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,  # Verificar conexiones antes de usarlas
    echo=settings.DB_ECHO,
    future=True,
)

# Engine sync para operaciones de migración/admin
sync_engine = create_engine(
    get_sync_database_url(),
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=settings.DB_ECHO,
    future=True,
)


# ============================================
# Configuración de Sesiones
# ============================================

# Factory para sesiones async
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# Factory para sesiones sync
SessionLocal = sessionmaker(
    sync_engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ============================================
# Gestores de Contexto para Sesiones
# ============================================

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Gestor de contexto asíncrono para sesiones de base de datos.
    
    Uso en dependencias de FastAPI:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_async_db)):
            ...
    
    Yields:
        AsyncSession: Sesión de base de datos asíncrona
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Error en transacción de base de datos: {e}")
            raise
        finally:
            await session.close()


def get_sync_db() -> Generator[Session, None, None]:
    """
    Gestor de contexto sincrónico para sesiones de base de datos.
    
    Uso en scripts o tareas de fondo:
        with get_sync_db() as db:
            ...
    
    Yields:
        Session: Sesión de base de datos sincrónica
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error en transacción de base de datos: {e}")
        raise
    finally:
        db.close()


# ============================================
# Utilidades de Base de Datos
# ============================================

async def init_db() -> None:
    """
    Inicializa la base de datos creando todas las tablas.
    
    Nota: En producción se recomienda usar Alembic para migraciones.
    """
    from app.models.base import Base
    
    logger.info("Inicializando base de datos...")
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Base de datos inicializada correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar la base de datos: {e}")
        raise


async def close_db_connections() -> None:
    """
    Cierra todas las conexiones a la base de datos.
    
    Debe llamarse al cerrar la aplicación.
    """
    logger.info("Cerrando conexiones a base de datos...")
    await async_engine.dispose()
    sync_engine.dispose()
    logger.info("Conexiones cerradas correctamente")


async def check_db_connection() -> bool:
    """
    Verifica la conexión a la base de datos.
    
    Returns:
        bool: True si la conexión es exitosa, False en caso contrario
    """
    try:
        async with async_engine.connect() as conn:
            result = await conn.execute("SELECT 1")
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Error al verificar conexión a base de datos: {e}")
        return False


# ============================================
# Eventos de SQLAlchemy
# ============================================

@event.listens_for(sync_engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """
    Configura pragmas para SQLite si se usa como base de datos.
    """
    # Esta función se ejecuta al conectar con la base de datos
    pass  # Implementar si se usa SQLite


@event.listens_for(sync_engine, "checkout")
def ping_connection(dbapi_conn, connection_record, connection_proxy):
    """
    Verifica la conexión antes de usarla del pool.
    """
    # SQLAlchemy maneja esto automáticamente con pool_pre_ping=True
    pass
