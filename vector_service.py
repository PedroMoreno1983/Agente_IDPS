"""
Servicio de Base de Datos Vectorial

Este módulo implementa la interfaz para interactuar con bases de datos
vectoriales (Pinecone o ChromaDB) para almacenar y buscar embeddings
de ítems psicométricos.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from app.config import settings
from app.core.embeddings import EmbeddingGenerator

# Configuración de logging
logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Resultado de búsqueda vectorial."""
    id: str
    score: float
    metadata: Dict[str, Any]
    text: Optional[str] = None


class BaseVectorStore(ABC):
    """Clase base abstracta para stores vectoriales."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Inicializa la conexión al store vectorial."""
        pass
    
    @abstractmethod
    async def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None
    ) -> bool:
        """Inserta o actualiza vectores en el store."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """Busca vectores similares."""
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str], namespace: Optional[str] = None) -> bool:
        """Elimina vectores por ID."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del store."""
        pass


class ChromaDBStore(BaseVectorStore):
    """
    Implementación de store vectorial usando ChromaDB.
    
    ChromaDB es una base de datos vectorial open-source que puede
    ejecutarse localmente o en la nube.
    """
    
    def __init__(self) -> None:
        """Inicializa el store ChromaDB."""
        self.client = None
        self.collection = None
        self.embedding_generator = EmbeddingGenerator()
        self._initialized = False
        
    async def initialize(self) -> None:
        """Inicializa la conexión a ChromaDB."""
        if self._initialized:
            return
            
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            logger.info(f"Inicializando ChromaDB en: {settings.CHROMADB_PATH}")
            
            # Crear cliente ChromaDB
            self.client = chromadb.PersistentClient(
                path=settings.CHROMADB_PATH,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Obtener o crear colección
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMADB_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
            
            self._initialized = True
            logger.info("ChromaDB inicializado correctamente")
            
        except ImportError:
            logger.error("chromadb no está instalado. Ejecuta: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Error al inicializar ChromaDB: {e}")
            raise
    
    async def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None
    ) -> bool:
        """
        Inserta o actualiza vectores en ChromaDB.
        
        Args:
            vectors: Lista de vectores con id, embedding y metadata
            namespace: Namespace opcional (no usado en ChromaDB)
            
        Returns:
            bool: True si la operación fue exitosa
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            ids = [v["id"] for v in vectors]
            embeddings = [v["embedding"] for v in vectors]
            metadatas = [v.get("metadata", {}) for v in vectors]
            documents = [v.get("text", "") for v in vectors]
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Upsert exitoso: {len(vectors)} vectores")
            return True
            
        except Exception as e:
            logger.error(f"Error en upsert: {e}")
            return False
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """
        Busca vectores similares en ChromaDB.
        
        Args:
            query_vector: Vector de consulta
            top_k: Número de resultados
            filter_dict: Filtros para la búsqueda
            namespace: Namespace opcional
            
        Returns:
            List[VectorSearchResult]: Resultados de la búsqueda
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=filter_dict,
                include=["metadatas", "documents", "distances"]
            )
            
            search_results = []
            for i, item_id in enumerate(results["ids"][0]):
                # ChromaDB devuelve distancia, convertimos a similitud
                distance = results["distances"][0][i]
                score = 1 - distance  # Convertir distancia a similitud
                
                search_results.append(VectorSearchResult(
                    id=item_id,
                    score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    text=results["documents"][0][i] if results["documents"] else None
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    async def delete(self, ids: List[str], namespace: Optional[str] = None) -> bool:
        """
        Elimina vectores por ID.
        
        Args:
            ids: Lista de IDs a eliminar
            namespace: Namespace opcional
            
        Returns:
            bool: True si la operación fue exitosa
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Eliminados {len(ids)} vectores")
            return True
        except Exception as e:
            logger.error(f"Error al eliminar vectores: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la colección.
        
        Returns:
            Dict[str, Any]: Estadísticas del store
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            count = self.collection.count()
            return {
                "total_vectors": count,
                "collection_name": settings.CHROMADB_COLLECTION,
                "provider": "chromadb"
            }
        except Exception as e:
            logger.error(f"Error al obtener estadísticas: {e}")
            return {"total_vectors": 0, "error": str(e)}


class PineconeStore(BaseVectorStore):
    """
    Implementación de store vectorial usando Pinecone.
    
    Pinecone es un servicio de base de datos vectorial gestionado
    en la nube, optimizado para búsquedas de similitud.
    """
    
    def __init__(self) -> None:
        """Inicializa el store Pinecone."""
        self.client = None
        self.index = None
        self.embedding_generator = EmbeddingGenerator()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Inicializa la conexión a Pinecone."""
        if self._initialized:
            return
            
        try:
            import pinecone
            
            logger.info("Inicializando Pinecone...")
            
            # Configurar Pinecone
            pinecone.init(
                api_key=settings.PINECONE_API_KEY,
                environment=settings.PINECONE_ENVIRONMENT
            )
            
            # Verificar/crear índice
            if settings.PINECONE_INDEX_NAME not in pinecone.list_indexes():
                logger.info(f"Creando índice: {settings.PINECONE_INDEX_NAME}")
                pinecone.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=settings.EMBEDDING_DIMENSION,
                    metric="cosine"
                )
            
            self.index = pinecone.Index(settings.PINECONE_INDEX_NAME)
            self._initialized = True
            
            logger.info("Pinecone inicializado correctamente")
            
        except ImportError:
            logger.error("pinecone-client no está instalado. Ejecuta: pip install pinecone-client")
            raise
        except Exception as e:
            logger.error(f"Error al inicializar Pinecone: {e}")
            raise
    
    async def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None
    ) -> bool:
        """
        Inserta o actualiza vectores en Pinecone.
        
        Args:
            vectors: Lista de vectores con id, embedding y metadata
            namespace: Namespace opcional
            
        Returns:
            bool: True si la operación fue exitosa
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Formatear vectores para Pinecone
            pinecone_vectors = []
            for v in vectors:
                pinecone_vectors.append({
                    "id": v["id"],
                    "values": v["embedding"],
                    "metadata": v.get("metadata", {})
                })
            
            self.index.upsert(vectors=pinecone_vectors, namespace=namespace)
            logger.info(f"Upsert exitoso: {len(vectors)} vectores")
            return True
            
        except Exception as e:
            logger.error(f"Error en upsert: {e}")
            return False
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """
        Busca vectores similares en Pinecone.
        
        Args:
            query_vector: Vector de consulta
            top_k: Número de resultados
            filter_dict: Filtros para la búsqueda
            namespace: Namespace opcional
            
        Returns:
            List[VectorSearchResult]: Resultados de la búsqueda
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_dict,
                namespace=namespace,
                include_metadata=True
            )
            
            search_results = []
            for match in results["matches"]:
                search_results.append(VectorSearchResult(
                    id=match["id"],
                    score=match["score"],
                    metadata=match.get("metadata", {})
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    async def delete(self, ids: List[str], namespace: Optional[str] = None) -> bool:
        """
        Elimina vectores por ID.
        
        Args:
            ids: Lista de IDs a eliminar
            namespace: Namespace opcional
            
        Returns:
            bool: True si la operación fue exitosa
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            self.index.delete(ids=ids, namespace=namespace)
            logger.info(f"Eliminados {len(ids)} vectores")
            return True
        except Exception as e:
            logger.error(f"Error al eliminar vectores: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del índice.
        
        Returns:
            Dict[str, Any]: Estadísticas del store
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_name": settings.PINECONE_INDEX_NAME,
                "provider": "pinecone"
            }
        except Exception as e:
            logger.error(f"Error al obtener estadísticas: {e}")
            return {"total_vectors": 0, "error": str(e)}


class VectorService:
    """
    Servicio de Base de Datos Vectorial.
    
    Proporciona una interfaz unificada para interactuar con
    diferentes proveedores de bases de datos vectoriales.
    """
    
    def __init__(self) -> None:
        """Inicializa el servicio de vectores."""
        self.store: Optional[BaseVectorStore] = None
        self.embedding_generator = EmbeddingGenerator()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Inicializa el servicio según la configuración."""
        if self._initialized:
            return
            
        # Crear el store según el proveedor configurado
        if settings.VECTOR_DB_PROVIDER == "pinecone":
            self.store = PineconeStore()
        elif settings.VECTOR_DB_PROVIDER == "chromadb":
            self.store = ChromaDBStore()
        else:
            raise ValueError(f"Proveedor no soportado: {settings.VECTOR_DB_PROVIDER}")
        
        # Inicializar el store
        await self.store.initialize()
        self._initialized = True
        
        logger.info(f"VectorService inicializado con {settings.VECTOR_DB_PROVIDER}")
    
    async def indexar_item(
        self,
        item_id: str,
        texto: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Indexa un ítem en la base de datos vectorial.
        
        Args:
            item_id: Identificador único del ítem
            texto: Texto para generar embedding
            metadata: Metadatos asociados
            
        Returns:
            bool: True si la indexación fue exitosa
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Generar embedding
            embedding = await self.embedding_generator.generate(texto)
            
            # Preparar vector
            vector = {
                "id": item_id,
                "embedding": embedding,
                "metadata": metadata or {},
                "text": texto
            }
            
            # Insertar en el store
            return await self.store.upsert([vector])
            
        except Exception as e:
            logger.error(f"Error al indexar ítem {item_id}: {e}")
            return False
    
    async def buscar_items(
        self,
        consulta: str,
        top_k: int = 10,
        filtros: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Busca ítems similares a una consulta.
        
        Args:
            consulta: Texto de consulta
            top_k: Número de resultados
            filtros: Filtros para la búsqueda
            
        Returns:
            List[VectorSearchResult]: Resultados de la búsqueda
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Generar embedding de la consulta
            query_embedding = await self.embedding_generator.generate(consulta)
            
            # Realizar búsqueda
            return await self.store.search(
                query_vector=query_embedding,
                top_k=top_k,
                filter_dict=filtros
            )
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    async def eliminar_item(self, item_id: str) -> bool:
        """
        Elimina un ítem de la base de datos vectorial.
        
        Args:
            item_id: ID del ítem a eliminar
            
        Returns:
            bool: True si la eliminación fue exitosa
        """
        if not self._initialized:
            await self.initialize()
            
        return await self.store.delete([item_id])
    
    async def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la base de datos vectorial.
        
        Returns:
            Dict[str, Any]: Estadísticas del servicio
        """
        if not self._initialized:
            await self.initialize()
            
        return await self.store.get_stats()
