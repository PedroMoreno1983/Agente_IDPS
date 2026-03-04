"""
Endpoints para Consultas al Agente de IA

Este módulo define los endpoints REST para interactuar con el
agente de IA psicométrico.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_async_db
from app.services.agente_service import AgenteService
from app.schemas.consulta import ConsultaRequest, FiltrosBusqueda
from app.schemas.respuesta import RespuestaAgente, EstadoAgente, ErrorRespuesta
from app.core.exceptions import AgentePsicometricoError, ConsultaInvalidaError

# Configuración de logging
logger = logging.getLogger(__name__)

# Crear router
router = APIRouter()

# Instancia del servicio del agente (singleton)
agente_service = AgenteService()


@router.post(
    "/",
    response_model=RespuestaAgente,
    responses={
        400: {"model": ErrorRespuesta, "description": "Consulta inválida"},
        500: {"model": ErrorRespuesta, "description": "Error interno del servidor"},
        503: {"model": ErrorRespuesta, "description": "Servicio no disponible"}
    },
    summary="Enviar consulta al agente de IA",
    description="""
    Envía una consulta al agente de IA psicométrico y recibe una respuesta
    estructurada que puede incluir búsqueda de ítems, sugerencias o análisis.
    
    El agente utiliza técnicas de RAG (Retrieval Augmented Generation) para
    proporcionar respuestas informadas basadas en el repositorio histórico.
    """
)
async def consultar_agente(
    request: ConsultaRequest,
    db: AsyncSession = Depends(get_async_db)
) -> RespuestaAgente:
    """
    Procesa una consulta al agente de IA.
    
    Args:
        request: Datos de la consulta incluyendo texto, filtros y contexto
        db: Sesión de base de datos
        
    Returns:
        RespuestaAgente: Respuesta estructurada del agente
        
    Raises:
        HTTPException: Si hay errores en el procesamiento
    """
    try:
        # Inicializar servicio si es necesario
        await agente_service.initialize()
        
        # Procesar consulta
        respuesta = await agente_service.procesar_consulta(request, db)
        
        return respuesta
        
    except ConsultaInvalidaError as e:
        logger.warning(f"Consulta inválida: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except AgentePsicometricoError as e:
        logger.error(f"Error del agente: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.exception("Error inesperado procesando consulta")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": True,
                "code": "INTERNAL_ERROR",
                "message": f"Error interno del servidor: {str(e)}"
            }
        )


@router.post(
    "/buscar",
    response_model=RespuestaAgente,
    summary="Buscar ítems usando el agente",
    description="""
    Realiza una búsqueda semántica de ítems utilizando el agente de IA.
    Combina búsqueda vectorial con filtros de base de datos.
    """
)
async def buscar_items_agente(
    consulta: str,
    filtros: Optional[FiltrosBusqueda] = None,
    max_resultados: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_async_db)
) -> RespuestaAgente:
    """
    Busca ítems usando el agente de IA.
    
    Args:
        consulta: Texto de búsqueda
        filtros: Filtros opcionales
        max_resultados: Máximo de resultados
        db: Sesión de base de datos
        
    Returns:
        RespuestaAgente: Resultados de búsqueda
    """
    try:
        await agente_service.initialize()
        
        request = ConsultaRequest(
            consulta=consulta,
            filtros=filtros,
            incluir_items_similares=True,
            max_resultados=max_resultados
        )
        
        return await agente_service.procesar_consulta(request, db)
        
    except Exception as e:
        logger.exception("Error en búsqueda de ítems")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": True,
                "code": "SEARCH_ERROR",
                "message": f"Error en la búsqueda: {str(e)}"
            }
        )


@router.post(
    "/generar",
    response_model=RespuestaAgente,
    summary="Generar sugerencias de ítems",
    description="""
    Solicita al agente que genere sugerencias de nuevos ítems
    para una dimensión específica, basándose en ítems históricos.
    """
)
async def generar_sugerencias(
    dimension: str,
    cantidad: int = Query(default=3, ge=1, le=10),
    contexto: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db)
) -> RespuestaAgente:
    """
    Genera sugerencias de ítems para una dimensión.
    
    Args:
        dimension: Dimensión objetivo
        cantidad: Cantidad de sugerencias
        contexto: Contexto adicional
        db: Sesión de base de datos
        
    Returns:
        RespuestaAgente: Sugerencias generadas
    """
    try:
        await agente_service.initialize()
        
        sugerencias = await agente_service.generar_sugerencias(
            dimension=dimension,
            cantidad=cantidad,
            contexto=contexto,
            db=db
        )
        
        return RespuestaAgente(
            mensaje=f"Se generaron {len(sugerencias)} sugerencias para la dimensión '{dimension}'",
            tipo_respuesta="generacion",
            sugerencias=sugerencias
        )
        
    except Exception as e:
        logger.exception("Error generando sugerencias")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": True,
                "code": "GENERATION_ERROR",
                "message": f"Error generando sugerencias: {str(e)}"
            }
        )


@router.post(
    "/analizar",
    response_model=RespuestaAgente,
    summary="Analizar un ítem propuesto",
    description="""
    Solicita al agente que analice un ítem propuesto o existente,
    proporcionando evaluación psicométrica y recomendaciones.
    """
)
async def analizar_item(
    item_text: str,
    dimension: str,
    db: AsyncSession = Depends(get_async_db)
) -> RespuestaAgente:
    """
    Analiza un ítem psicométrico.
    
    Args:
        item_text: Texto del ítem a analizar
        dimension: Dimensión del ítem
        db: Sesión de base de datos
        
    Returns:
        RespuestaAgente: Análisis del ítem
    """
    try:
        await agente_service.initialize()
        
        analisis = await agente_service.analizar_item(
            item_text=item_text,
            dimension=dimension,
            db=db
        )
        
        return RespuestaAgente(
            mensaje=f"Análisis completado para el ítem de '{dimension}'",
            tipo_respuesta="analisis",
            analisis=analisis
        )
        
    except Exception as e:
        logger.exception("Error analizando ítem")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": True,
                "code": "ANALYSIS_ERROR",
                "message": f"Error en el análisis: {str(e)}"
            }
        )


@router.get(
    "/estado",
    response_model=EstadoAgente,
    summary="Obtener estado del agente",
    description="""
    Retorna el estado actual del agente de IA, incluyendo el estado
    de sus componentes (base de datos, vector DB, LLM) y estadísticas.
    """
)
async def obtener_estado(
    db: AsyncSession = Depends(get_async_db)
) -> EstadoAgente:
    """
    Obtiene el estado del agente.
    
    Args:
        db: Sesión de base de datos
        
    Returns:
        EstadoAgente: Estado del agente
    """
    try:
        await agente_service.initialize()
        return await agente_service.obtener_estado(db)
        
    except Exception as e:
        logger.exception("Error obteniendo estado del agente")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": True,
                "code": "STATUS_ERROR",
                "message": f"Error obteniendo estado: {str(e)}"
            }
        )


@router.get(
    "/health",
    summary="Health check del agente",
    description="Endpoint simple para verificar que el servicio está respondiendo."
)
async def health_check() -> dict:
    """
    Verifica la salud del servicio.
    
    Returns:
        dict: Estado de salud
    """
    return {
        "status": "ok",
        "service": "agente-psicometrico",
        "version": "0.1.0"
    }
