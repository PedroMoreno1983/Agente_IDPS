"""
================================================================================
SERVICIO DE GENERACIÓN DE ÍTEMS
================================================================================

Servicio para la generación inteligente de ítems psicométricos utilizando
LLM (Large Language Models) y contexto histórico.

Funcionalidades principales:
- Generación de ítems sugeridos basados en dimensión y dificultad objetivo
- Sugerencia de mejoras para ítems existentes
- Validación de nuevos ítems propuestos
- Búsqueda de ítems similares usando embeddings

Este servicio integra el motor de reglas psicométricas con LLMs para
proporcionar generación de ítems de alta calidad.

Autor: Agente de IA Psicométrico
Versión: 1.0.0
================================================================================
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import re

# Importar módulos del núcleo
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.reglas_psicometricas import (
    validar_parametros_irt,
    clasificar_dificultad,
    UmbralesPsicometricos
)

from core.utils_psicometricos import (
    validar_texto_item,
    validar_opciones_respuesta,
    validar_dimension,
    detectar_lenguaje_inclusivo,
    extraer_palabras_clave,
    calcular_legibilidad,
    ConstantesPsicometricas
)

from core.prompts import (
    ContextoGeneracion,
    construir_prompt_generacion,
    construir_prompt_sugerencias,
    construir_prompt_validacion
)


# ==============================================================================
# ESTRUCTURAS DE DATOS
# ==============================================================================

@dataclass
class ItemGenerado:
    """Estructura para un ítem generado."""
    texto: str
    opciones_respuesta: List[str]
    valores_opciones: List[int]
    direccion: str  # "positiva" o "negativa"
    justificacion: str
    nivel_dificultad_estimado: str
    consideraciones: List[str]
    
    # Metadatos de generación
    dimension: str = ""
    metodo_generacion: str = "llm"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "texto": self.texto,
            "opciones_respuesta": self.opciones_respuesta,
            "valores_opciones": self.valores_opciones,
            "direccion": self.direccion,
            "justificacion": self.justificacion,
            "nivel_dificultad_estimado": self.nivel_dificultad_estimado,
            "consideraciones": self.consideraciones,
            "dimension": self.dimension,
            "metodo_generacion": self.metodo_generacion,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ResultadoValidacion:
    """Resultado de la validación de un ítem."""
    es_valido: bool
    puntuacion: float
    criterios: Dict[str, Dict[str, Any]]
    fortalezas: List[str]
    debilidades: List[str]
    sugerencias: List[str]
    recomendacion_final: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "es_valido": self.es_valido,
            "puntuacion": self.puntuacion,
            "criterios": self.criterios,
            "fortalezas": self.fortalezas,
            "debilidades": self.debilidades,
            "sugerencias": self.sugerencias,
            "recomendacion_final": self.recomendacion_final
        }


@dataclass
class ItemSimilar:
    """Estructura para ítems similares encontrados."""
    item_id: int
    texto: str
    similitud_score: float
    tipo_similitud: str  # "semantica", "tematica", "estructural"
    parametros: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "item_id": self.item_id,
            "texto": self.texto[:100] + "..." if len(self.texto) > 100 else self.texto,
            "similitud_score": round(self.similitud_score, 4),
            "tipo_similitud": self.tipo_similitud,
            "parametros": self.parametros
        }


@dataclass
class SugerenciaMejora:
    """Sugerencia de mejora para un ítem."""
    tipo: str
    descripcion: str
    prioridad: str  # "alta", "media", "baja"
    ejemplo_original: str
    ejemplo_mejorado: str
    impacto_esperado: str


# ==============================================================================
# SERVICIO DE GENERACIÓN
# ==============================================================================

class ServicioGeneracionItems:
    """
    Servicio para generación inteligente de ítems psicométricos.
    
    Este servicio combina el poder de los LLMs con el conocimiento
    psicométrico para generar ítems de alta calidad.
    """
    
    def __init__(
        self,
        llm_client=None,
        vector_db=None,
        db_connection=None
    ):
        """
        Inicializa el servicio de generación.
        
        Args:
            llm_client: Cliente para el LLM (OpenAI, Anthropic, etc.)
            vector_db: Base de datos vectorial para búsqueda de similitud
            db_connection: Conexión a la base de datos principal
        """
        self.llm = llm_client
        self.vector_db = vector_db
        self.db = db_connection
        
        # Caché de ítems de referencia
        self.cache_items_referencia: Dict[int, List[Dict]] = {}
    
    # --------------------------------------------------------------------------
    # GENERACIÓN DE ÍTEMS
    # --------------------------------------------------------------------------
    
    def generar_item_sugerido(
        self,
        dimension_id: int,
        dificultad_objetivo: Optional[str] = None,
        formato_respuesta: str = "likert_4",
        nivel_educativo: str = "general",
        poblacion_objetivo: str = "estudiantes",
        usar_historial: bool = True,
        restricciones: Optional[List[str]] = None
    ) -> ItemGenerado:
        """
        Genera un ítem sugerido usando LLM + contexto histórico.
        
        Args:
            dimension_id: ID de la dimensión para la que generar el ítem
            dificultad_objetivo: Nivel de dificultad deseado
            formato_respuesta: Formato de las opciones de respuesta
            nivel_educativo: Nivel educativo objetivo
            poblacion_objetivo: Descripción de la población
            usar_historial: Si se deben usar ítems históricos como referencia
            restricciones: Lista de restricciones adicionales
        
        Returns:
            ItemGenerado: El ítem generado
        
        Ejemplo:
            >>> servicio = ServicioGeneracionItems(llm_client=cliente_openai)
            >>> item = servicio.generar_item_sugerido(
            ...     dimension_id=1,
            ...     dificultad_objetivo="moderada"
            ... )
        """
        # Obtener información de la dimensión
        dimension_info = self._obtener_info_dimension(dimension_id)
        
        # Obtener ítems de referencia si se solicita
        items_referencia = []
        if usar_historial:
            items_referencia = self._obtener_items_referencia(
                dimension_id,
                dificultad_objetivo,
                top_k=3
            )
        
        # Construir contexto
        contexto = ContextoGeneracion(
            dimension=dimension_info.get('nombre', 'Dimensión desconocida'),
            descripcion_dimension=dimension_info.get('descripcion', ''),
            nivel_educativo=nivel_educativo,
            poblacion_objetivo=poblacion_objetivo,
            formato_respuesta=formato_respuesta,
            dificultad_objetivo=dificultad_objetivo,
            items_referencia=items_referencia,
            restricciones=restricciones
        )
        
        # Generar prompt
        prompt = construir_prompt_generacion(contexto)
        
        # Llamar al LLM (simulado si no hay cliente)
        if self.llm:
            respuesta = self._llamar_llm(prompt)
        else:
            # Modo demo: generar ítem de ejemplo
            respuesta = self._generar_item_demo(contexto)
        
        # Parsear respuesta
        item_generado = self._parsear_item_generado(respuesta, contexto.dimension)
        
        return item_generado
    
    def generar_varios_items(
        self,
        dimension_id: int,
        cantidad: int = 3,
        distribucion_dificultad: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> List[ItemGenerado]:
        """
        Genera múltiples ítems con distribución de dificultad.
        
        Args:
            dimension_id: ID de la dimensión
            cantidad: Número total de ítems a generar
            distribucion_dificultad: Distribución deseada por nivel
            **kwargs: Parámetros adicionales para generar_item_sugerido
        
        Returns:
            Lista de ítems generados
        """
        if distribucion_dificultad is None:
            # Distribución equilibrada por defecto
            distribucion_dificultad = {
                "facil": cantidad // 3,
                "moderada": cantidad // 3,
                "dificil": cantidad - 2 * (cantidad // 3)
            }
        
        items_generados = []
        
        for dificultad, num_items in distribucion_dificultad.items():
            for _ in range(num_items):
                try:
                    item = self.generar_item_sugerido(
                        dimension_id=dimension_id,
                        dificultad_objetivo=dificultad,
                        **kwargs
                    )
                    items_generados.append(item)
                except Exception as e:
                    # Continuar con el siguiente si hay error
                    continue
        
        return items_generados
    
    # --------------------------------------------------------------------------
    # SUGERENCIAS DE MEJORA
    # --------------------------------------------------------------------------
    
    def sugerir_mejoras(
        self,
        item_id: int,
        incluir_analisis_psicometrico: bool = True
    ) -> Dict[str, Any]:
        """
        Sugiere mejoras para un ítem existente.
        
        Args:
            item_id: ID del ítem a mejorar
            incluir_analisis_psicometrico: Si se debe incluir el análisis IRT
        
        Returns:
            Diccionario con sugerencias estructuradas
        """
        # Obtener datos del ítem
        item_data = self._obtener_item_db(item_id)
        
        if not item_data:
            return {"error": f"Ítem {item_id} no encontrado"}
        
        # Obtener problemas detectados
        problemas = []
        if incluir_analisis_psicometrico:
            problemas = self._obtener_problemas_item(item_id)
        
        # Construir prompt
        prompt = construir_prompt_sugerencias(
            texto_item=item_data.get('texto', ''),
            opciones=item_data.get('opciones', []),
            problemas=problemas,
            dimension=item_data.get('dimension_nombre', ''),
            nivel_educativo=item_data.get('nivel_educativo', 'general'),
            dificultad_objetivo=item_data.get('dificultad_objetivo', 'moderada')
        )
        
        # Llamar al LLM
        if self.llm:
            respuesta = self._llamar_llm(prompt)
            sugerencias = self._parsear_sugerencias(respuesta)
        else:
            # Modo demo
            sugerencias = self._generar_sugerencias_demo(item_data, problemas)
        
        return sugerencias
    
    def generar_versiones_alternativas(
        self,
        item_id: int,
        num_versiones: int = 2
    ) -> List[ItemGenerado]:
        """
        Genera versiones alternativas de un ítem existente.
        
        Args:
            item_id: ID del ítem original
            num_versiones: Número de versiones a generar
        
        Returns:
            Lista de versiones alternativas
        """
        item_original = self._obtener_item_db(item_id)
        
        if not item_original:
            return []
        
        versiones = []
        
        for i in range(num_versiones):
            prompt = f"""
            Genera una versión alternativa del siguiente ítem psicométrico.
            La nueva versión debe medir el MISMO constructo pero con redacción diferente.
            
            ÍTEM ORIGINAL:
            {item_original.get('texto', '')}
            
            OPCIONES ORIGINALES:
            {', '.join(item_original.get('opciones', []))}
            
            INSTRUCCIONES:
            - Mantén el mismo nivel de dificultad
            - Usa lenguaje diferente pero equivalente
            - Mantén el mismo formato de respuesta
            - Asegúrate de que mida exactamente lo mismo
            
            Genera la versión alternativa {i+1} en formato JSON.
            """
            
            if self.llm:
                respuesta = self._llamar_llm(prompt)
                version = self._parsear_item_generado(
                    respuesta,
                    item_original.get('dimension_nombre', '')
                )
                versiones.append(version)
        
        return versiones
    
    # --------------------------------------------------------------------------
    # VALIDACIÓN DE ÍTEMS
    # --------------------------------------------------------------------------
    
    def validar_nuevo_item(
        self,
        texto_item: str,
        dimension_id: int,
        opciones_respuesta: Optional[List[str]] = None,
        usar_llm: bool = True
    ) -> ResultadoValidacion:
        """
        Valida un ítem propuesto según criterios psicométricos.
        
        Args:
            texto_item: Texto del ítem a validar
            dimension_id: ID de la dimensión
            opciones_respuesta: Opciones de respuesta (si aplica)
            usar_llm: Si se debe usar LLM para validación avanzada
        
        Returns:
            ResultadoValidacion: Resultado de la validación
        """
        # Validación básica
        criterios = {}
        fortalezas = []
        debilidades = []
        
        # 1. Validar texto
        es_valido_texto, errores_texto = validar_texto_item(texto_item)
        puntos_texto = 25 if es_valido_texto else max(0, 25 - len(errores_texto) * 5)
        criterios["texto"] = {
            "puntos": puntos_texto,
            "comentario": "Texto válido" if es_valido_texto else f"Errores: {', '.join(errores_texto[:3])}"
        }
        
        if es_valido_texto:
            fortalezas.append("Texto claro y bien estructurado")
        else:
            debilidades.extend(errores_texto[:3])
        
        # 2. Validar dimensión
        dimension_info = self._obtener_info_dimension(dimension_id)
        es_valida_dim, msg_dim = validar_dimension(dimension_info.get('nombre', ''))
        puntos_dimension = 25 if es_valida_dim else 15
        criterios["dimension"] = {
            "puntos": puntos_dimension,
            "comentario": msg_dim
        }
        
        # 3. Validar opciones
        puntos_opciones = 25
        if opciones_respuesta:
            es_valido_op, errores_op = validar_opciones_respuesta(opciones_respuesta)
            puntos_opciones = 25 if es_valido_op else max(0, 25 - len(errores_op) * 5)
            criterios["opciones"] = {
                "puntos": puntos_opciones,
                "comentario": "Opciones válidas" if es_valido_op else f"Errores: {', '.join(errores_op[:3])}"
            }
            if not es_valido_op:
                debilidades.extend(errores_op[:2])
        else:
            criterios["opciones"] = {"puntos": 25, "comentario": "No aplica"}
        
        # 4. Validar lenguaje inclusivo
        es_inclusivo, sugerencias_inclusivo = detectar_lenguaje_inclusivo(texto_item)
        puntos_inclusivo = 25 if es_inclusivo else max(0, 25 - len(sugerencias_inclusivo) * 5)
        criterios["lenguaje_inclusivo"] = {
            "puntos": puntos_inclusivo,
            "comentario": "Lenguaje inclusivo" if es_inclusivo else f"Sugerencias: {', '.join(sugerencias_inclusivo[:2])}"
        }
        
        if es_inclusivo:
            fortalezas.append("Usa lenguaje inclusivo")
        else:
            debilidades.extend(sugerencias_inclusivo[:2])
        
        # Calcular puntuación total
        puntuacion_total = sum(c["puntos"] for c in criterios.values())
        
        # Validación con LLM si está disponible
        if usar_llm and self.llm:
            prompt = construir_prompt_validacion(
                texto_item=texto_item,
                dimension=dimension_info.get('nombre', ''),
                descripcion_dimension=dimension_info.get('descripcion', '')
            )
            respuesta_llm = self._llamar_llm(prompt)
            validacion_llm = self._parsear_validacion_llm(respuesta_llm)
            
            # Combinar resultados
            puntuacion_total = (puntuacion_total + validacion_llm.get('puntuacion', 0)) / 2
            fortalezas.extend(validacion_llm.get('fortalezas', []))
            debilidades.extend(validacion_llm.get('debilidades', []))
        
        # Determinar recomendación final
        if puntuacion_total >= 80:
            recomendacion = "aceptar"
        elif puntuacion_total >= 60:
            recomendacion = "revisar"
        else:
            recomendacion = "rechazar"
        
        # Generar sugerencias
        sugerencias = self._generar_sugerencias_validacion(debilidades)
        
        return ResultadoValidacion(
            es_valido=puntuacion_total >= 60,
            puntuacion=puntuacion_total,
            criterios=criterios,
            fortalezas=list(set(fortalezas)),  # Eliminar duplicados
            debilidades=list(set(debilidades)),
            sugerencias=sugerencias,
            recomendacion_final=recomendacion
        )
    
    def validar_batch_items(
        self,
        items: List[Dict[str, Any]]
    ) -> List[ResultadoValidacion]:
        """
        Valida un lote de ítems.
        
        Args:
            items: Lista de diccionarios con datos de ítems
        
        Returns:
            Lista de resultados de validación
        """
        resultados = []
        for item_data in items:
            resultado = self.validar_nuevo_item(
                texto_item=item_data.get('texto', ''),
                dimension_id=item_data.get('dimension_id', 0),
                opciones_respuesta=item_data.get('opciones_respuesta')
            )
            resultados.append(resultado)
        return resultados
    
    # --------------------------------------------------------------------------
    # BÚSQUEDA DE ÍTEMS SIMILARES
    # --------------------------------------------------------------------------
    
    def buscar_items_similares(
        self,
        texto_item: str,
        dimension_id: Optional[int] = None,
        top_k: int = 5,
        umbral_similitud: float = 0.7
    ) -> List[ItemSimilar]:
        """
        Busca ítems similares usando embeddings.
        
        Args:
            texto_item: Texto del ítem de consulta
            dimension_id: Filtrar por dimensión (None = todas)
            top_k: Número máximo de resultados
            umbral_similitud: Umbral mínimo de similitud
        
        Returns:
            Lista de ítems similares ordenados por similitud
        
        Ejemplo:
            >>> servicio = ServicioGeneracionItems(vector_db=pinecone_client)
            >>> similares = servicio.buscar_items_similares(
            ...     "Me siento seguro de mis habilidades",
            ...     top_k=5
            ... )
        """
        items_similares = []
        
        if self.vector_db:
            # Generar embedding del texto de consulta
            embedding_consulta = self._generar_embedding(texto_item)
            
            # Buscar en base de datos vectorial
            resultados = self._buscar_vector_db(
                embedding_consulta,
                dimension_id,
                top_k=top_k * 2  # Buscar más para filtrar
            )
            
            # Filtrar por umbral y formatear
            for resultado in resultados:
                score = resultado.get('score', 0)
                if score >= umbral_similitud:
                    items_similares.append(ItemSimilar(
                        item_id=resultado.get('item_id', 0),
                        texto=resultado.get('texto', ''),
                        similitud_score=score,
                        tipo_similitud="semantica",
                        parametros=resultado.get('parametros')
                    ))
        
        # Búsqueda por palabras clave como respaldo
        if len(items_similares) < top_k:
            palabras_clave = extraer_palabras_clave(texto_item, n=5)
            resultados_keyword = self._buscar_por_keywords(
                palabras_clave,
                dimension_id,
                top_k=top_k - len(items_similares)
            )
            
            for resultado in resultados_keyword:
                # Evitar duplicados
                if not any(r.item_id == resultado.get('item_id') for r in items_similares):
                    items_similares.append(ItemSimilar(
                        item_id=resultado.get('item_id', 0),
                        texto=resultado.get('texto', ''),
                        similitud_score=resultado.get('score', 0.5),
                        tipo_similitud="tematica",
                        parametros=resultado.get('parametros')
                    ))
        
        # Ordenar por score y limitar
        items_similares.sort(key=lambda x: x.similitud_score, reverse=True)
        return items_similares[:top_k]
    
    def detectar_redundancia(
        self,
        texto_item: str,
        dimension_id: int,
        umbral_redundancia: float = 0.85
    ) -> Dict[str, Any]:
        """
        Detecta si un ítem es redundante con ítems existentes.
        
        Args:
            texto_item: Texto del ítem a verificar
            dimension_id: ID de la dimensión
            umbral_redundancia: Umbral para considerar redundante
        
        Returns:
            Resultado de la detección de redundancia
        """
        similares = self.buscar_items_similares(
            texto_item,
            dimension_id,
            top_k=3,
            umbral_similitud=umbral_redundancia
        )
        
        es_redundante = len(similares) > 0 and similares[0].similitud_score >= umbral_redundancia
        
        return {
            "es_redundante": es_redundante,
            "score_maximo": similares[0].similitud_score if similares else 0,
            "items_similares": [s.to_dict() for s in similares],
            "recomendacion": (
                "El ítem es potencialmente redundante. Considere modificarlo o usar el ítem existente."
                if es_redundante else
                "El ítem no parece redundante con ítems existentes."
            )
        }
    
    # --------------------------------------------------------------------------
    # MÉTODOS PRIVADOS
    # --------------------------------------------------------------------------
    
    def _obtener_info_dimension(self, dimension_id: int) -> Dict[str, str]:
        """Obtiene información de una dimensión."""
        if self.db:
            # Consultar base de datos
            pass
        
        # Valores por defecto
        return {
            'nombre': f'Dimensión {dimension_id}',
            'descripcion': 'Descripción no disponible'
        }
    
    def _obtener_items_referencia(
        self,
        dimension_id: int,
        dificultad: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """Obtiene ítems de referencia de alta calidad."""
        # Verificar caché
        cache_key = f"{dimension_id}_{dificultad}"
        if cache_key in self.cache_items_referencia:
            return self.cache_items_referencia[cache_key]
        
        items = []
        
        if self.db:
            # Consultar ítems con buenos parámetros
            pass
        
        # Guardar en caché
        self.cache_items_referencia[cache_key] = items
        
        return items
    
    def _obtener_item_db(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene un ítem de la base de datos."""
        if self.db:
            # Consultar base de datos
            pass
        return None
    
    def _obtener_problemas_item(self, item_id: int) -> List[Dict[str, Any]]:
        """Obtiene los problemas detectados de un ítem."""
        # Integrar con servicio de análisis
        return []
    
    def _llamar_llm(self, prompt: str) -> str:
        """Llama al LLM con el prompt proporcionado."""
        if self.llm:
            # Implementación específica del cliente LLM
            # Ejemplo para OpenAI:
            # response = self.llm.chat.completions.create(...)
            # return response.choices[0].message.content
            pass
        return ""
    
    def _generar_embedding(self, texto: str) -> List[float]:
        """Genera embedding vectorial del texto."""
        if self.llm:
            # Usar API de embeddings
            pass
        return []
    
    def _buscar_vector_db(
        self,
        embedding: List[float],
        dimension_id: Optional[int],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Busca en la base de datos vectorial."""
        if self.vector_db:
            # Implementación específica
            pass
        return []
    
    def _buscar_por_keywords(
        self,
        keywords: List[str],
        dimension_id: Optional[int],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Busca ítems por palabras clave."""
        if self.db:
            # Consulta SQL con LIKE
            pass
        return []
    
    def _parsear_item_generado(self, respuesta: str, dimension: str) -> ItemGenerado:
        """Parsea la respuesta del LLM a ItemGenerado."""
        try:
            # Intentar extraer JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', respuesta, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = respuesta
            
            data = json.loads(json_str)
            
            return ItemGenerado(
                texto=data.get('texto_item', ''),
                opciones_respuesta=data.get('opciones_respuesta', []),
                valores_opciones=data.get('valores_opciones', list(range(1, len(data.get('opciones_respuesta', [])) + 1))),
                direccion=data.get('direccion', 'positiva'),
                justificacion=data.get('justificacion', ''),
                nivel_dificultad_estimado=data.get('nivel_dificultad_estimado', 'moderada'),
                consideraciones=data.get('consideraciones', []),
                dimension=dimension
            )
        except json.JSONDecodeError:
            # Fallback: crear ítem básico
            return ItemGenerado(
                texto="Error al parsear respuesta del LLM",
                opciones_respuesta=[],
                valores_opciones=[],
                direccion="positiva",
                justificacion="Error en generación",
                nivel_dificultad_estimado="moderada",
                consideraciones=["Revisar manualmente"],
                dimension=dimension
            )
    
    def _parsear_sugerencias(self, respuesta: str) -> Dict[str, Any]:
        """Parsea las sugerencias del LLM."""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', respuesta, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(respuesta)
        except:
            return {"sugerencias_texto": respuesta}
    
    def _parsear_validacion_llm(self, respuesta: str) -> Dict[str, Any]:
        """Parsea la validación del LLM."""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', respuesta, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(respuesta)
        except:
            return {"puntuacion": 70, "fortalezas": [], "debilidades": []}
    
    def _generar_sugerencias_validacion(self, debilidades: List[str]) -> List[str]:
        """Genera sugerencias basadas en debilidades."""
        sugerencias = []
        
        for debilidad in debilidades:
            if "doble negación" in debilidad.lower():
                sugerencias.append("Reformular eliminando las negaciones múltiples")
            elif "largo" in debilidad.lower():
                sugerencias.append("Simplificar el texto para hacerlo más conciso")
            elif "corto" in debilidad.lower():
                sugerencias.append("Agregar más contexto al ítem")
            elif "inclusivo" in debilidad.lower():
                sugerencias.append("Revisar el lenguaje para hacerlo más inclusivo")
            elif "ambiguo" in debilidad.lower():
                sugerencias.append("Aclarar el enunciado para eliminar ambigüedades")
        
        return sugerencias if sugerencias else ["Revisar el ítem manualmente"]
    
    def _generar_item_demo(self, contexto: ContextoGeneracion) -> str:
        """Genera un ítem de demostración cuando no hay LLM."""
        # Plantillas de ítems por dimensión
        plantillas = {
            "Autoestima Académica": {
                "facil": "Me siento bien con mis logros en la escuela.",
                "moderada": "Creo que puedo aprender cosas difíciles si me esfuerzo.",
                "dificil": "Incluso cuando las tareas son complejas, confío en mis habilidades para completarlas."
            },
            "Autoeficacia": {
                "facil": "Puedo resolver problemas simples.",
                "moderada": "Sé que puedo alcanzar mis metas con trabajo duro.",
                "dificil": "Cuando enfrento obstáculos inesperados, encuentro formas creativas de superarlos."
            },
            "Convivencia Escolar": {
                "facil": "Me llevo bien con mis compañeros.",
                "moderada": "Respeto las opiniones de otros incluso cuando no estoy de acuerdo.",
                "dificil": "Puedo mediar en conflictos entre compañeros de manera imparcial."
            }
        }
        
        dim = contexto.dimension
        dif = contexto.dificultad_objetivo or "moderada"
        
        texto = plantillas.get(dim, {}).get(dif, "Ítem de ejemplo para evaluación.")
        
        return json.dumps({
            "texto_item": texto,
            "opciones_respuesta": ["Totalmente en desacuerdo", "En desacuerdo", "De acuerdo", "Totalmente de acuerdo"],
            "valores_opciones": [1, 2, 3, 4],
            "direccion": "positiva",
            "justificacion": f"Ítem generado para medir {dim} con dificultad {dif}",
            "nivel_dificultad_estimado": dif,
            "consideraciones": ["Revisar alineación con constructo", "Validar con población objetivo"]
        }, ensure_ascii=False)
    
    def _generar_sugerencias_demo(
        self,
        item_data: Dict[str, Any],
        problemas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Genera sugerencias de demostración."""
        return {
            "cambios_redaccion": [
                {
                    "original": item_data.get('texto', ''),
                    "sugerencia": "Considerar simplificar el lenguaje",
                    "justificacion": "Mejora la comprensión"
                }
            ],
            "versiones_alternativas": [],
            "consideraciones_implementacion": [
                "Realizar prueba piloto",
                "Recoger feedback de estudiantes"
            ]
        }


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def estimar_dificultad_texto(texto: str) -> str:
    """
    Estima la dificultad de un ítem basándose en características del texto.
    
    Args:
        texto: Texto del ítem
    
    Returns:
        Nivel de dificultad estimado
    """
    legibilidad = calcular_legibilidad(texto)
    num_palabras = legibilidad.get('palabras', 0)
    
    # Palabras que indican mayor dificultad cognitiva
    palabras_complejas = [
        'analizar', 'evaluar', 'sintetizar', 'comparar', 'contrastar',
        'justificar', 'argumentar', 'inferir', 'deducir', 'extrapolar'
    ]
    
    texto_lower = texto.lower()
    complejidad = sum(1 for p in palabras_complejas if p in texto_lower)
    
    # Estimación
    if num_palabras < 10 and complejidad == 0:
        return "muy_facil"
    elif num_palabras < 15 and complejidad <= 1:
        return "facil"
    elif num_palabras < 25 and complejidad <= 2:
        return "moderada"
    elif num_palabras < 35 or complejidad > 2:
        return "dificil"
    else:
        return "muy_dificil"


def generar_variante_dificultad(
    texto_original: str,
    direccion: str  # "aumentar" o "disminuir"
) -> str:
    """
    Genera una variante del ítem cambiando su nivel de dificultad.
    
    Args:
        texto_original: Texto del ítem original
        direccion: "aumentar" o "disminuir"
    
    Returns:
        Texto modificado
    """
    if direccion == "aumentar":
        # Agregar complejidad
        modificaciones = [
            (r'\bsiento\b', 'percibo y analizo'),
            (r'\bpuedo\b', 'tengo la capacidad de'),
            (r'\bcreo\b', 'estoy convencido de que'),
        ]
    else:
        # Simplificar
        modificaciones = [
            (r'\bpercibo y analizo\b', 'siento'),
            (r'\btengo la capacidad de\b', 'puedo'),
            (r'\bestoy convencido de que\b', 'creo'),
        ]
    
    texto_modificado = texto_original
    for patron, reemplazo in modificaciones:
        texto_modificado = re.sub(patron, reemplazo, texto_modificado, flags=re.IGNORECASE)
    
    return texto_modificado


# ==============================================================================
# EXPORTAR
# ==============================================================================

__all__ = [
    'ServicioGeneracionItems',
    'ItemGenerado',
    'ResultadoValidacion',
    'ItemSimilar',
    'SugerenciaMejora',
    'estimar_dificultad_texto',
    'generar_variante_dificultad'
]
