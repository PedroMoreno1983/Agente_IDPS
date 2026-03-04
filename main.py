"""
Agente de IA Psicométrico - Aplicación FastAPI

Punto de entrada principal de la aplicación FastAPI que expone
los endpoints REST del agente de IA especializado en psicometría educativa.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.router import api_router
from app.models.database import init_db, close_db_connections, check_db_connection
from app.core.exceptions import AgentePsicometricoError

# ============================================
# Configuración de Logging
# ============================================

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


# ============================================
# Eventos de Ciclo de Vida
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestor del ciclo de vida de la aplicación.
    
    Se ejecuta al iniciar y al cerrar la aplicación.
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"Iniciando {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)
    
    try:
        # Inicializar base de datos
        logger.info("Inicializando base de datos...")
        await init_db()
        
        # Verificar conexión
        if await check_db_connection():
            logger.info("✓ Conexión a base de datos establecida")
        else:
            logger.warning("✗ No se pudo verificar conexión a base de datos")
        
        logger.info("✓ Aplicación iniciada correctamente")
        
    except Exception as e:
        logger.error(f"Error durante el inicio: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Cerrando aplicación...")
    await close_db_connections()
    logger.info("✓ Aplicación cerrada correctamente")


# ============================================
# Creación de la Aplicación FastAPI
# ============================================

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# ============================================
# Middleware CORS
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# ============================================
# Manejadores de Excepciones Globales
# ============================================

@app.exception_handler(AgentePsicometricoError)
async def agente_exception_handler(request: Request, exc: AgentePsicometricoError):
    """Maneja excepciones personalizadas del agente."""
    logger.error(f"Error del agente: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Maneja excepciones no controladas."""
    logger.exception("Error no controlado")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "code": "INTERNAL_ERROR",
            "message": "Error interno del servidor",
            "detail": str(exc) if settings.DEBUG else None
        }
    )


# ============================================
# Routers
# ============================================

app.include_router(
    api_router,
    prefix="/api/v1"
)


# ============================================
# Endpoints de Health Check
# ============================================

@app.get("/", tags=["Health"])
async def root():
    """
    Endpoint raíz de la aplicación.
    
    Returns:
        Información básica de la aplicación.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check de la aplicación.
    
    Returns:
        Estado de salud del servicio.
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check para orquestación (Kubernetes).
    
    Verifica que todos los componentes estén listos.
    
    Returns:
        Estado de readiness del servicio.
    """
    try:
        # Verificar conexión a base de datos
        db_ok = await check_db_connection()
        
        if db_ok:
            return {
                "status": "ready",
                "database": "connected"
            }
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "database": "disconnected"
                }
            )
    except Exception as e:
        logger.error(f"Error en readiness check: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "error": str(e)
            }
        )


# ============================================
# Punto de Entrada
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1
    )
