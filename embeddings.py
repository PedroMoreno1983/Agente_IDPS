"""
Generador de Embeddings

Este módulo implementa la generación de embeddings (representaciones vectoriales)
utilizando modelos de lenguaje pre-entrenados.
"""

import logging
from typing import List, Optional

from app.config import settings

# Configuración de logging
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generador de embeddings para texto.
    
    Utiliza modelos de embeddings de OpenAI o modelos locales
    para convertir texto en vectores numéricos.
    """
    
    def __init__(self, model: Optional[str] = None) -> None:
        """
        Inicializa el generador de embeddings.
        
        Args:
            model: Modelo de embeddings a usar (opcional)
        """
        self.model = model or settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self._client = None
        
        logger.info(f"EmbeddingGenerator inicializado con modelo: {self.model}")
    
    def _get_client(self):
        """Obtiene o crea el cliente de OpenAI."""
        if self._client is None:
            try:
                import openai
                openai.api_key = settings.OPENAI_API_KEY
                self._client = openai.AsyncOpenAI()
            except ImportError:
                logger.error("openai no está instalado. Ejecuta: pip install openai")
                raise
        return self._client
    
    async def generate(self, text: str) -> List[float]:
        """
        Genera el embedding para un texto.
        
        Args:
            text: Texto para generar embedding
            
        Returns:
            List[float]: Vector de embeddings
        """
        if not text or not text.strip():
            logger.warning("Texto vacío proporcionado para embedding")
            return [0.0] * self.dimension
        
        try:
            client = self._get_client()
            
            response = await client.embeddings.create(
                model=self.model,
                input=text[:8000]  # Limitar longitud
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            # Retornar vector vacío en caso de error
            return [0.0] * self.dimension
    
    async def generate_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Genera embeddings para un batch de textos.
        
        Args:
            texts: Lista de textos
            batch_size: Tamaño del batch (opcional)
            
        Returns:
            List[List[float]]: Lista de vectores de embeddings
        """
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                client = self._get_client()
                
                response = await client.embeddings.create(
                    model=self.model,
                    input=[t[:8000] for t in batch]
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error generando batch de embeddings: {e}")
                # Retornar vectores vacíos para el batch fallido
                embeddings.extend([[0.0] * self.dimension] * len(batch))
        
        return embeddings
    
    def generate_sync(self, text: str) -> List[float]:
        """
        Genera el embedding para un texto (versión sincrónica).
        
        Args:
            text: Texto para generar embedding
            
        Returns:
            List[float]: Vector de embeddings
        """
        import asyncio
        return asyncio.run(self.generate(text))
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calcula la similitud coseno entre dos vectores.
        
        Args:
            vec1: Primer vector
            vec2: Segundo vector
            
        Returns:
            float: Similitud coseno (0-1)
        """
        import math
        
        if len(vec1) != len(vec2):
            raise ValueError("Los vectores deben tener la misma dimensión")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
