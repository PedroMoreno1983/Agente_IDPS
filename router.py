"""
Router Principal de la API

Define el router principal que agrupa todos los endpoints de la API.
"""

from fastapi import APIRouter

from app.api.endpoints import consultas, items

# Crear router principal
api_router = APIRouter()

# Incluir routers de endpoints
api_router.include_router(
    consultas.router,
    prefix="/consultas",
    tags=["Consultas al Agente"]
)

api_router.include_router(
    items.router,
    prefix="/items",
    tags=["Gestión de Ítems"]
)
