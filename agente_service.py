"""
Servicio Principal del Agente de IA Psicométrico

Este módulo implementa el orquestador principal que coordina todos los
componentes del agente: PLN, búsqueda vectorial, LLM y reglas psicométricas.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.config import settings
from app.services.pln_service import PLNService, IntencionConsulta
from app.services.vector_service import VectorService, VectorSearchResult
from app.services.llm_service import LLMService, LLMResponse
from app.schemas.consulta import ConsultaRequest, FiltrosBusqueda
from app.schemas.respuesta import (
    RespuestaAgente,
    ResultadoBusqueda,
    SugerenciaItem,
    AnalisisItem,
    EstadoAgente
)
from app.models.item import Item
from app.models.dimension import Dimension

# Configuración de logging
logger = logging.getLogger(__name__)


class AgenteService:
    """
    Orquestador principal del Agente de IA Psicométrico.
    
    Este servicio coordina:
    - Procesamiento de lenguaje natural (PLN)
    - Búsqueda semántica en base de datos vectorial
    - Consulta a modelos de lenguaje (LLM)
    - Aplicación de reglas psicométricas
    - Integración con base de datos relacional
    
    Implementa el patrón RAG (Retrieval Augmented Generation) para
    proporcionar respuestas informadas basadas en datos históricos.
    """
    
    def __init__(self) -> None:
        """Inicializa el servicio del agente."""
        self.pln_service = PLNService()
        self.vector_service = VectorService()
        self.llm_service = LLMService()
        self._initialized = False
        
        # Cache simple de sesiones (en producción usar Redis)
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
    
    async def initialize(self) -> None:
        """Inicializa todos los componentes del agente."""
        if self._initialized:
            return
            
        logger.info("Inicializando AgenteService...")
        
        try:
            # Inicializar servicios
            await self.vector_service.initialize()
            await self.llm_service.initialize()
            
            self._initialized = True
            logger.info("AgenteService inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar AgenteService: {e}")
            raise
    
    async def procesar_consulta(
        self,
        request: ConsultaRequest,
        db: AsyncSession
    ) -> RespuestaAgente:
        """
        Procesa una consulta del usuario y genera una respuesta.
        
        Este es el método principal del agente. Analiza la consulta,
        busca información relevante, consulta el LLM y estructura
        la respuesta.
        
        Args:
            request: Consulta del usuario con filtros y contexto
            db: Sesión de base de datos
            
        Returns:
            RespuestaAgente: Respuesta estructurada del agente
        """
        start_time = datetime.utcnow()
        consulta_id = f"query_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Procesando consulta {consulta_id}: {request.consulta[:100]}...")
        
        try:
            # 1. Validar consulta
            es_valida, error_msg = self.pln_service.es_consulta_valida(request.consulta)
            if not es_valida:
                return self._crear_respuesta_error(
                    consulta_id, "CONSULTA_INVALIDA", error_msg or "Consulta inválida"
                )
            
            # 2. Detectar intención
            intencion = self.pln_service.detectar_intencion(request.consulta)
            logger.info(f"Intención detectada: {intencion.tipo} (confianza: {intencion.confianza:.2f})")
            
            # 3. Extraer entidades
            entidades = self.pln_service.extraer_entidades(request.consulta)
            
            # 4. Buscar ítems relevantes si es necesario
            items_similares: List[VectorSearchResult] = []
            if request.incluir_items_similares:
                items_similares = await self._buscar_items_similares(
                    request.consulta,
                    request.filtros,
                    request.max_resultados
                )
            
            # 5. Obtener información de base de datos
            contexto_db = await self._obtener_contexto_db(entidades, db)
            
            # 6. Construir contexto para LLM
            contexto_llm = self._construir_contexto_llm(
                items_similares,
                contexto_db,
                entidades
            )
            
            # 7. Obtener historial de sesión
            historial = self._obtener_historial(request.session_id)
            
            # 8. Generar respuesta con LLM
            llm_response = await self.llm_service.generate_response(
                user_message=request.consulta,
                context=contexto_llm,
                conversation_history=historial,
                temperature=0.3
            )
            
            # 9. Estructurar respuesta según tipo
            respuesta = await self._estructurar_respuesta(
                consulta_id=consulta_id,
                intencion=intencion,
                llm_response=llm_response,
                items_similares=items_similares,
                request=request,
                start_time=start_time
            )
            
            # 10. Actualizar historial
            self._actualizar_historial(request.session_id, request.consulta, respuesta.mensaje)
            
            logger.info(f"Consulta {consulta_id} procesada exitosamente")
            return respuesta
            
        except Exception as e:
            logger.error(f"Error procesando consulta {consulta_id}: {e}")
            return self._crear_respuesta_error(
                consulta_id,
                "ERROR_INTERNO",
                f"Error al procesar la consulta: {str(e)}"
            )
    
    async def buscar_items(
        self,
        consulta: str,
        filtros: Optional[FiltrosBusqueda] = None,
        top_k: int = 10,
        db: AsyncSession
    ) -> List[ResultadoBusqueda]:
        """
        Busca ítems relevantes usando búsqueda híbrida.
        
        Combina búsqueda semántica (vectorial) con filtros
        de base de datos relacional.
        
        Args:
            consulta: Texto de búsqueda
            filtros: Filtros adicionales
            top_k: Número de resultados
            db: Sesión de base de datos
            
        Returns:
            List[ResultadoBusqueda]: Resultados de búsqueda
        """
        # 1. Búsqueda vectorial
        resultados_vectoriales = await self._buscar_items_similares(
            consulta,
            filtros,
            top_k
        )
        
        # 2. Obtener información completa de la base de datos
        resultados: List[ResultadoBusqueda] = []
        
        for rv in resultados_vectoriales:
            # Buscar ítem en BD
            item_result = await db.execute(
                select(Item).where(Item.id == int(rv.id))
            )
            item = item_result.scalar_one_or_none()
            
            if item:
                resultados.append(ResultadoBusqueda(
                    item_id=item.id,
                    texto_item=item.texto_item,
                    dimension=item.dimension.nombre if item.dimension else "Desconocida",
                    año_creacion=item.año_creacion,
                    formato_respuesta=item.formato_respuesta,
                    score_similitud=rv.score,
                    parametros_psicometricos=self._extraer_parametros_item(item),
                    activo=item.activo
                ))
        
        return resultados
    
    async def generar_sugerencias(
        self,
        dimension: str,
        cantidad: int = 3,
        contexto: Optional[str] = None,
        db: AsyncSession
    ) -> List[SugerenciaItem]:
        """
        Genera sugerencias de nuevos ítems para una dimensión.
        
        Args:
            dimension: Dimensión objetivo
            cantidad: Cantidad de sugerencias
            contexto: Contexto adicional
            db: Sesión de base de datos
            
        Returns:
            List[SugerenciaItem]: Sugerencias generadas
        """
        # 1. Buscar ítems de referencia de la dimensión
        items_referencia = await self._obtener_items_dimension(dimension, db, limit=5)
        
        # 2. Generar sugerencias con LLM
        llm_response = await self.llm_service.generate_suggestions(
            dimension=dimension,
            cantidad=cantidad,
            items_referencia=items_referencia
        )
        
        # 3. Parsear y estructurar sugerencias
        sugerencias = self._parsear_sugerencias(llm_response.content)
        
        return sugerencias
    
    async def analizar_item(
        self,
        item_text: str,
        dimension: str,
        db: AsyncSession
    ) -> AnalisisItem:
        """
        Analiza un ítem propuesto o existente.
        
        Args:
            item_text: Texto del ítem
            dimension: Dimensión del ítem
            db: Sesión de base de datos
            
        Returns:
            AnalisisItem: Análisis del ítem
        """
        # 1. Buscar ítems similares para comparación
        items_similares = await self._buscar_items_similares(item_text, None, 5)
        
        # 2. Obtener parámetros históricos
        parametros_historicos = await self._obtener_parametros_historicos(
            [int(r.id) for r in items_similares],
            db
        )
        
        # 3. Generar análisis con LLM
        llm_response = await self.llm_service.analyze_item(
            item_text=item_text,
            dimension=dimension,
            parametros_historicos=parametros_historicos
        )
        
        # 4. Parsear y estructurar análisis
        analisis = self._parsear_analisis(item_text, llm_response.content)
        
        return analisis
    
    async def obtener_estado(self, db: AsyncSession) -> EstadoAgente:
        """
        Obtiene el estado actual del agente.
        
        Args:
            db: Sesión de base de datos
            
        Returns:
            EstadoAgente: Estado del agente
        """
        # Verificar componentes
        db_status = "conectada"
        try:
            await db.execute(select(Item).limit(1))
        except Exception:
            db_status = "error"
        
        vector_status = "conectada"
        try:
            await self.vector_service.obtener_estadisticas()
        except Exception:
            vector_status = "desconectada"
        
        llm_status = "disponible" if await self.llm_service.health_check() else "no_disponible"
        
        # Determinar estado general
        if db_status == "error" or llm_status == "no_disponible":
            estado_general = "no_disponible"
        elif vector_status == "desconectada":
            estado_general = "degradado"
        else:
            estado_general = "operativo"
        
        # Obtener estadísticas
        try:
            stats_vector = await self.vector_service.obtener_estadisticas()
            total_items = stats_vector.get("total_vectors", 0)
            
            dim_result = await db.execute(select(Dimension))
            total_dimensiones = len(dim_result.scalars().all())
        except Exception:
            total_items = None
            total_dimensiones = None
        
        return EstadoAgente(
            estado=estado_general,
            version=settings.APP_VERSION,
            base_datos=db_status,
            base_datos_vectorial=vector_status,
            servicio_llm=llm_status,
            total_items_indexados=total_items,
            total_dimensiones=total_dimensiones,
            consultas_hoy=len(self.sessions)
        )
    
    # ============================================
    # Métodos privados auxiliares
    # ============================================
    
    async def _buscar_items_similares(
        self,
        consulta: str,
        filtros: Optional[FiltrosBusqueda],
        top_k: int
    ) -> List[VectorSearchResult]:
        """Busca ítems similares usando búsqueda vectorial."""
        try:
            # Construir filtros para vector DB
            filter_dict = None
            if filtros:
                filter_dict = {}
                if filtros.años:
                    filter_dict["año"] = {"$in": filtros.años}
                if filtros.solo_activos:
                    filter_dict["activo"] = True
            
            return await self.vector_service.buscar_items(
                consulta=consulta,
                top_k=top_k,
                filtros=filter_dict
            )
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {e}")
            return []
    
    async def _obtener_contexto_db(
        self,
        entidades: Any,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Obtiene información relevante de la base de datos."""
        contexto = {
            "dimensiones": [],
            "total_items": 0,
            "parametros_promedio": {}
        }
        
        try:
            # Contar ítems
            result = await db.execute(select(Item))
            contexto["total_items"] = len(result.scalars().all())
            
            # Obtener dimensiones mencionadas
            if entidades.dimensiones:
                for dim_nombre in entidades.dimensiones:
                    dim_result = await db.execute(
                        select(Dimension).where(Dimension.nombre.ilike(f"%{dim_nombre}%"))
                    )
                    dimension = dim_result.scalar_one_or_none()
                    if dimension:
                        contexto["dimensiones"].append({
                            "id": dimension.id,
                            "nombre": dimension.nombre,
                            "cantidad_items": len(dimension.items)
                        })
        except Exception as e:
            logger.error(f"Error obteniendo contexto de BD: {e}")
        
        return contexto
    
    def _construir_contexto_llm(
        self,
        items_similares: List[VectorSearchResult],
        contexto_db: Dict[str, Any],
        entidades: Any
    ) -> str:
        """Construye el contexto para el LLM."""
        contexto_parts = []
        
        # Información de la base de datos
        contexto_parts.append(f"Total de ítems en repositorio: {contexto_db.get('total_items', 0)}")
        
        # Dimensiones relevantes
        if contexto_db.get("dimensiones"):
            contexto_parts.append("\nDimensiones relevantes:")
            for dim in contexto_db["dimensiones"]:
                contexto_parts.append(f"- {dim['nombre']}: {dim['cantidad_items']} ítems")
        
        # Ítems similares encontrados
        if items_similares:
            contexto_parts.append(f"\nÍtems históricos similares encontrados ({len(items_similares)}):")
            for i, item in enumerate(items_similares[:5], 1):
                contexto_parts.append(f"\n{i}. {item.metadata.get('texto', 'N/A')[:100]}...")
                contexto_parts.append(f"   Score de similitud: {item.score:.3f}")
                if item.metadata.get("dificultad"):
                    contexto_parts.append(f"   Dificultad: {item.metadata['dificultad']}")
                if item.metadata.get("discriminacion"):
                    contexto_parts.append(f"   Discriminación: {item.metadata['discriminacion']}")
        
        # Entidades extraídas
        if entidades.años:
            contexto_parts.append(f"\nAños mencionados: {entidades.años}")
        
        return "\n".join(contexto_parts)
    
    def _obtener_historial(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        """Obtiene el historial de una sesión."""
        if not session_id or session_id not in self.sessions:
            return []
        return self.sessions[session_id][-5:]  # Últimos 5 mensajes
    
    def _actualizar_historial(
        self,
        session_id: Optional[str],
        consulta: str,
        respuesta: str
    ) -> None:
        """Actualiza el historial de una sesión."""
        if not session_id:
            return
            
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "role": "user",
            "content": consulta
        })
        self.sessions[session_id].append({
            "role": "assistant",
            "content": respuesta
        })
    
    async def _estructurar_respuesta(
        self,
        consulta_id: str,
        intencion: IntencionConsulta,
        llm_response: LLMResponse,
        items_similares: List[VectorSearchResult],
        request: ConsultaRequest,
        start_time: datetime
    ) -> RespuestaAgente:
        """Estructura la respuesta final según el tipo de consulta."""
        tiempo_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Determinar tipo de respuesta
        tipo_respuesta = intencion.tipo
        if intencion.tipo == "desconocida":
            tipo_respuesta = "informacion"
        
        # Construir respuesta base
        respuesta = RespuestaAgente(
            mensaje=llm_response.content,
            tipo_respuesta=tipo_respuesta,
            consulta_id=consulta_id,
            session_id=request.session_id,
            tokens_usados=llm_response.tokens_usados,
            tiempo_procesamiento_ms=tiempo_ms,
            fuentes_consultadas=["base_datos_vectorial", "postgresql"],
            confianza=intencion.confianza
        )
        
        # Agregar items encontrados si aplica
        if items_similares and tipo_respuesta == "busqueda":
            respuesta.items_encontrados = [
                ResultadoBusqueda(
                    item_id=int(r.id),
                    texto_item=r.metadata.get("texto", ""),
                    dimension=r.metadata.get("dimension", "Desconocida"),
                    año_creacion=r.metadata.get("año", 0),
                    formato_respuesta=r.metadata.get("formato", ""),
                    score_similitud=r.score,
                    parametros_psicometricos={
                        "dificultad": r.metadata.get("dificultad"),
                        "discriminacion": r.metadata.get("discriminacion")
                    },
                    activo=r.metadata.get("activo", True)
                )
                for r in items_similares
            ]
        
        return respuesta
    
    def _crear_respuesta_error(
        self,
        consulta_id: str,
        codigo: str,
        mensaje: str
    ) -> RespuestaAgente:
        """Crea una respuesta de error."""
        return RespuestaAgente(
            mensaje=f"Lo siento, ocurrió un error: {mensaje}",
            tipo_respuesta="error",
            consulta_id=consulta_id,
            notas=f"Código de error: {codigo}"
        )
    
    async def _obtener_items_dimension(
        self,
        dimension: str,
        db: AsyncSession,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Obtiene ítems de una dimensión específica."""
        try:
            result = await db.execute(
                select(Item)
                .join(Dimension)
                .where(Dimension.nombre.ilike(f"%{dimension}%"))
                .where(Item.activo == True)
                .limit(limit)
            )
            items = result.scalars().all()
            
            return [
                {
                    "id": item.id,
                    "texto": item.texto_item,
                    "formato": item.formato_respuesta,
                    "año": item.año_creacion
                }
                for item in items
            ]
        except Exception as e:
            logger.error(f"Error obteniendo ítems de dimensión: {e}")
            return []
    
    async def _obtener_parametros_historicos(
        self,
        item_ids: List[int],
        db: AsyncSession
    ) -> Dict[str, float]:
        """Obtiene parámetros promedio de ítems históricos."""
        # Implementación simplificada
        return {
            "dificultad_promedio": 0.0,
            "discriminacion_promedio": 1.0
        }
    
    def _extraer_parametros_item(self, item: Item) -> Dict[str, Any]:
        """Extrae parámetros psicométricos de un ítem."""
        if not item.ultimo_analisis:
            return {}
        
        analisis = item.ultimo_analisis
        return {
            "dificultad": analisis.param_b_dificultad,
            "discriminacion": analisis.param_a_discriminacion,
            "ajuste": "bueno" if analisis.tiene_buen_ajuste else "regular"
        }
    
    def _parsear_sugerencias(self, content: str) -> List[SugerenciaItem]:
        """Parsea las sugeriones del LLM."""
        # Implementación simplificada - en producción usar JSON parsing
        return []
    
    def _parsear_analisis(self, item_text: str, content: str) -> AnalisisItem:
        """Parsea el análisis del LLM."""
        # Implementación simplificada
        return AnalisisItem(
            item_id=0,
            texto_item=item_text,
            fortalezas=[],
            debilidades=[],
            recomendaciones=[]
        )
