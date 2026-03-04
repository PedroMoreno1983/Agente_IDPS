"""
Endpoints para Gestión de Ítems

Este módulo define los endpoints REST para la gestión CRUD de ítems
psicométricos en el repositorio.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import selectinload

from app.models.database import get_async_db
from app.models.item import Item
from app.models.dimension import Dimension
from app.models.opcion_respuesta import OpcionRespuesta
from app.models.analisis_psicometrico import AnalisisPsicometrico
from app.schemas.item import (
    ItemCreate,
    ItemUpdate,
    ItemResponse,
    ItemListResponse,
    ItemSearchResult
)
from app.core.exceptions import DatabaseNotFoundError

# Configuración de logging
logger = logging.getLogger(__name__)

# Crear router
router = APIRouter()


@router.get(
    "/",
    response_model=ItemListResponse,
    summary="Listar ítems",
    description="""
    Obtiene una lista paginada de ítems psicométricos con opciones de filtrado
    por dimensión, año, estado activo, etc.
    """
)
async def listar_items(
    skip: int = Query(default=0, ge=0, description="Ítems a saltar"),
    limit: int = Query(default=20, ge=1, le=100, description="Ítems por página"),
    dimension_id: Optional[int] = Query(None, description="Filtrar por dimensión"),
    año: Optional[int] = Query(None, description="Filtrar por año"),
    activo: Optional[bool] = Query(None, description="Filtrar por estado activo"),
    busqueda: Optional[str] = Query(None, description="Búsqueda en texto"),
    db: AsyncSession = Depends(get_async_db)
) -> ItemListResponse:
    """
    Lista ítems con filtros y paginación.
    
    Args:
        skip: Ítems a saltar (paginación)
        limit: Ítems por página
        dimension_id: Filtro por dimensión
        año: Filtro por año
        activo: Filtro por estado
        busqueda: Texto de búsqueda
        db: Sesión de base de datos
        
    Returns:
        ItemListResponse: Lista paginada de ítems
    """
    try:
        # Construir query base
        query = select(Item).options(
            selectinload(Item.dimension),
            selectinload(Item.opciones_respuesta),
            selectinload(Item.analisis_psicometricos)
        )
        
        # Aplicar filtros
        filters = []
        if dimension_id:
            filters.append(Item.dimension_id == dimension_id)
        if año:
            filters.append(Item.año_creacion == año)
        if activo is not None:
            filters.append(Item.activo == activo)
        if busqueda:
            filters.append(Item.texto_item.ilike(f"%{busqueda}%"))
        
        if filters:
            query = query.where(and_(*filters))
        
        # Contar total
        count_query = select(Item)
        if filters:
            count_query = count_query.where(and_(*filters))
        total_result = await db.execute(count_query)
        total = len(total_result.scalars().all())
        
        # Aplicar paginación
        query = query.offset(skip).limit(limit)
        
        # Ejecutar query
        result = await db.execute(query)
        items = result.scalars().all()
        
        # Calcular páginas
        total_pages = (total + limit - 1) // limit
        current_page = (skip // limit) + 1
        
        return ItemListResponse(
            items=[ItemResponse.model_validate(item) for item in items],
            total=total,
            page=current_page,
            page_size=limit,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.exception("Error listando ítems")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar ítems: {str(e)}"
        )


@router.get(
    "/{item_id}",
    response_model=ItemResponse,
    responses={
        404: {"description": "Ítem no encontrado"}
    },
    summary="Obtener ítem por ID",
    description="Obtiene la información completa de un ítem específico."
)
async def obtener_item(
    item_id: int,
    db: AsyncSession = Depends(get_async_db)
) -> ItemResponse:
    """
    Obtiene un ítem por su ID.
    
    Args:
        item_id: ID del ítem
        db: Sesión de base de datos
        
    Returns:
        ItemResponse: Información del ítem
        
    Raises:
        HTTPException: Si el ítem no existe
    """
    try:
        result = await db.execute(
            select(Item)
            .options(
                selectinload(Item.dimension),
                selectinload(Item.opciones_respuesta),
                selectinload(Item.analisis_psicometricos)
            )
            .where(Item.id == item_id)
        )
        item = result.scalar_one_or_none()
        
        if not item:
            raise DatabaseNotFoundError("Item", item_id)
        
        return ItemResponse.model_validate(item)
        
    except DatabaseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ítem con ID {item_id} no encontrado"
        )
    except Exception as e:
        logger.exception(f"Error obteniendo ítem {item_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener ítem: {str(e)}"
        )


@router.post(
    "/",
    response_model=ItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Crear nuevo ítem",
    description="""
    Crea un nuevo ítem psicométrico en el repositorio.
    Opcionalmente puede incluir opciones de respuesta.
    """
)
async def crear_item(
    item_data: ItemCreate,
    db: AsyncSession = Depends(get_async_db)
) -> ItemResponse:
    """
    Crea un nuevo ítem.
    
    Args:
        item_data: Datos del ítem a crear
        db: Sesión de base de datos
        
    Returns:
        ItemResponse: Ítem creado
    """
    try:
        # Verificar que la dimensión existe
        dim_result = await db.execute(
            select(Dimension).where(Dimension.id == item_data.dimension_id)
        )
        if not dim_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dimensión con ID {item_data.dimension_id} no existe"
            )
        
        # Crear ítem
        nuevo_item = Item(
            dimension_id=item_data.dimension_id,
            texto_item=item_data.texto_item,
            formato_respuesta=item_data.formato_respuesta,
            año_creacion=item_data.año_creacion,
            activo=item_data.activo
        )
        
        db.add(nuevo_item)
        await db.flush()  # Para obtener el ID
        
        # Crear opciones de respuesta si se proporcionan
        if item_data.opciones_respuesta:
            for opcion_data in item_data.opciones_respuesta:
                opcion = OpcionRespuesta(
                    item_id=nuevo_item.id,
                    texto_opcion=opcion_data.texto_opcion,
                    valor_numerico=opcion_data.valor_numerico
                )
                db.add(opcion)
        
        await db.commit()
        
        # Recargar con relaciones
        result = await db.execute(
            select(Item)
            .options(
                selectinload(Item.dimension),
                selectinload(Item.opciones_respuesta),
                selectinload(Item.analisis_psicometricos)
            )
            .where(Item.id == nuevo_item.id)
        )
        item = result.scalar_one()
        
        logger.info(f"Ítem creado: {item.id}")
        return ItemResponse.model_validate(item)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.exception("Error creando ítem")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al crear ítem: {str(e)}"
        )


@router.put(
    "/{item_id}",
    response_model=ItemResponse,
    summary="Actualizar ítem",
    description="Actualiza los datos de un ítem existente."
)
async def actualizar_item(
    item_id: int,
    item_data: ItemUpdate,
    db: AsyncSession = Depends(get_async_db)
) -> ItemResponse:
    """
    Actualiza un ítem existente.
    
    Args:
        item_id: ID del ítem a actualizar
        item_data: Datos a actualizar
        db: Sesión de base de datos
        
    Returns:
        ItemResponse: Ítem actualizado
    """
    try:
        # Buscar ítem
        result = await db.execute(
            select(Item).where(Item.id == item_id)
        )
        item = result.scalar_one_or_none()
        
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ítem con ID {item_id} no encontrado"
            )
        
        # Verificar dimensión si se cambia
        if item_data.dimension_id:
            dim_result = await db.execute(
                select(Dimension).where(Dimension.id == item_data.dimension_id)
            )
            if not dim_result.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Dimensión con ID {item_data.dimension_id} no existe"
                )
        
        # Actualizar campos
        update_data = item_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(item, field, value)
        
        await db.commit()
        
        # Recargar con relaciones
        result = await db.execute(
            select(Item)
            .options(
                selectinload(Item.dimension),
                selectinload(Item.opciones_respuesta),
                selectinload(Item.analisis_psicometricos)
            )
            .where(Item.id == item_id)
        )
        item = result.scalar_one()
        
        logger.info(f"Ítem actualizado: {item_id}")
        return ItemResponse.model_validate(item)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.exception(f"Error actualizando ítem {item_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar ítem: {str(e)}"
        )


@router.delete(
    "/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Eliminar ítem",
    description="Elimina un ítem del repositorio."
)
async def eliminar_item(
    item_id: int,
    db: AsyncSession = Depends(get_async_db)
) -> None:
    """
    Elimina un ítem.
    
    Args:
        item_id: ID del ítem a eliminar
        db: Sesión de base de datos
    """
    try:
        result = await db.execute(
            select(Item).where(Item.id == item_id)
        )
        item = result.scalar_one_or_none()
        
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ítem con ID {item_id} no encontrado"
            )
        
        await db.delete(item)
        await db.commit()
        
        logger.info(f"Ítem eliminado: {item_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.exception(f"Error eliminando ítem {item_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar ítem: {str(e)}"
        )


@router.get(
    "/dimension/{dimension_id}",
    response_model=List[ItemSearchResult],
    summary="Obtener ítems por dimensión",
    description="Obtiene todos los ítems asociados a una dimensión específica."
)
async def obtener_items_por_dimension(
    dimension_id: int,
    solo_activos: bool = Query(default=True),
    db: AsyncSession = Depends(get_async_db)
) -> List[ItemSearchResult]:
    """
    Obtiene ítems por dimensión.
    
    Args:
        dimension_id: ID de la dimensión
        solo_activos: Solo ítems activos
        db: Sesión de base de datos
        
    Returns:
        List[ItemSearchResult]: Ítems de la dimensión
    """
    try:
        query = select(Item).options(
            selectinload(Item.dimension),
            selectinload(Item.analisis_psicometricos)
        ).where(Item.dimension_id == dimension_id)
        
        if solo_activos:
            query = query.where(Item.activo == True)
        
        result = await db.execute(query)
        items = result.scalars().all()
        
        return [
            ItemSearchResult(
                id=item.id,
                texto_item=item.texto_item,
                formato_respuesta=item.formato_respuesta,
                año_creacion=item.año_creacion,
                activo=item.activo,
                dimension=item.dimension.nombre if item.dimension else None,
                dificultad=item.ultimo_analisis.param_b_dificultad if item.ultimo_analisis else None,
                discriminacion=item.ultimo_analisis.param_a_discriminacion if item.ultimo_analisis else None
            )
            for item in items
        ]
        
    except Exception as e:
        logger.exception(f"Error obteniendo ítems por dimensión {dimension_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener ítems: {str(e)}"
        )
