"""
Servicio de Procesamiento de Lenguaje Natural (PLN)

Este módulo implementa funcionalidades para el procesamiento,
análisis y extracción de información de texto en consultas
relacionadas con psicometría educativa.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass

from app.config import settings

# Configuración de logging
logger = logging.getLogger(__name__)


@dataclass
class EntidadesExtraidas:
    """Entidades extraídas de una consulta."""
    dimensiones: List[str]
    constructos: List[str]
    años: List[int]
    formatos_respuesta: List[str]
    parametros: Dict[str, Any]
    poblacion_objetivo: Optional[str]
    nivel_educativo: Optional[str]


@dataclass
class IntencionConsulta:
    """Intención detectada en una consulta."""
    tipo: Literal["busqueda", "generacion", "analisis", "informacion", "desconocida"]
    confianza: float
    subtipo: Optional[str]


class PLNService:
    """
    Servicio de Procesamiento de Lenguaje Natural.
    
    Responsable de:
    - Analizar y entender las consultas de los usuarios
    - Extraer entidades relevantes (dimensiones, años, etc.)
    - Detectar la intención de la consulta
    - Preprocesar texto para búsquedas
    """
    
    # ============================================
    # Constantes y vocabulario
    # ============================================
    
    # Palabras clave para detección de intenciones
    PALABRAS_BUSQUEDA = [
        "buscar", "encontrar", "busco", "necesito", "dame", "muéstrame",
        "lista", "listado", "cuáles", "qué ítems", "mostrar", "ver"
    ]
    
    PALABRAS_GENERACION = [
        "generar", "crear", "nuevo", "nuevos", "sugerir", "proponer",
        "diseñar", "elaborar", "construir", "desarrollar", "propón"
    ]
    
    PALABRAS_ANALISIS = [
        "analizar", "evaluar", "revisar", "qué tal", "cómo es",
        "calidad", "parámetros", "psicométrico", "dificultad",
        "discriminación", "ajuste"
    ]
    
    PALABRAS_INFORMACION = [
        "información", "cuéntame", "explica", "qué es", "cómo funciona",
        "dime", "describir", "descripción"
    ]
    
    # Formatos de respuesta conocidos
    FORMATOS_RESPUESTA = [
        "likert 4", "likert 5", "likert", "dicotómico", "dicotomico",
        "si/no", "sí/no", "escala", "ordinal", "nominal"
    ]
    
    # Niveles educativos
    NIVELES_EDUCATIVOS = {
        "básica": ["básica", "basica", "primaria", "básico"],
        "media": ["media", "secundaria", "medio", "enseñanza media"],
        "universitaria": ["universitaria", "universidad", "superior", "terciaria"]
    }
    
    def __init__(self) -> None:
        """Inicializa el servicio de PLN."""
        logger.info("PLNService inicializado")
    
    # ============================================
    # Métodos de detección de intención
    # ============================================
    
    def detectar_intencion(self, consulta: str) -> IntencionConsulta:
        """
        Detecta la intención principal de una consulta.
        
        Args:
            consulta: Texto de la consulta del usuario
            
        Returns:
            IntencionConsulta: Tipo de intención y nivel de confianza
        """
        consulta_lower = consulta.lower()
        
        # Contar coincidencias para cada tipo
        scores = {
            "busqueda": sum(1 for palabra in self.PALABRAS_BUSQUEDA if palabra in consulta_lower),
            "generacion": sum(1 for palabra in self.PALABRAS_GENERACION if palabra in consulta_lower),
            "analisis": sum(1 for palabra in self.PALABRAS_ANALISIS if palabra in consulta_lower),
            "informacion": sum(1 for palabra in self.PALABRAS_INFORMACION if palabra in consulta_lower),
        }
        
        # Detectar subtipos específicos
        subtipo = None
        if "similar" in consulta_lower or "parecido" in consulta_lower:
            subtipo = "busqueda_similar"
        elif "mejor" in consulta_lower or "óptimo" in consulta_lower or "optimo" in consulta_lower:
            subtipo = "busqueda_optimos"
        elif "redactar" in consulta_lower or "redacción" in consulta_lower:
            subtipo = "mejora_redaccion"
        
        # Determinar el tipo con mayor score
        if max(scores.values()) == 0:
            return IntencionConsulta(
                tipo="desconocida",
                confianza=0.0,
                subtipo=None
            )
        
        tipo_detectado = max(scores, key=scores.get)
        confianza = min(scores[tipo_detectado] / 3, 1.0)  # Normalizar a 0-1
        
        return IntencionConsulta(
            tipo=tipo_detectado,
            confianza=confianza,
            subtipo=subtipo
        )
    
    # ============================================
    # Métodos de extracción de entidades
    # ============================================
    
    def extraer_entidades(self, consulta: str) -> EntidadesExtraidas:
        """
        Extrae entidades relevantes de una consulta.
        
        Args:
            consulta: Texto de la consulta del usuario
            
        Returns:
            EntidadesExtraidas: Entidades extraídas de la consulta
        """
        consulta_lower = consulta.lower()
        
        return EntidadesExtraidas(
            dimensiones=self._extraer_dimensiones(consulta_lower),
            constructos=self._extraer_constructos(consulta_lower),
            años=self._extraer_años(consulta),
            formatos_respuesta=self._extraer_formatos(consulta_lower),
            parametros=self._extraer_parametros(consulta_lower),
            poblacion_objetivo=self._extraer_poblacion(consulta_lower),
            nivel_educativo=self._extraer_nivel_educativo(consulta_lower)
        )
    
    def _extraer_dimensiones(self, consulta: str) -> List[str]:
        """Extrae menciones a dimensiones psicométricas."""
        dimensiones = []
        
        # Patrones comunes de dimensiones
        patrones_dimension = [
            r"sobre\s+([\w\s]+?)(?:\s+(?:para|de|con)|$)",
            r"dimensión\s+(?:de\s+)?([\w\s]+?)(?:\s+(?:para|de|con)|$)",
            r"([\w\s]+?)\s+(?:académica|escolar|personal|social)",
        ]
        
        dimensiones_conocidas = [
            "autoestima", "autoestima académica", "autoestima social",
            "convivencia", "convivencia escolar", "clima escolar",
            "habitos", "hábitos de vida", "hábitos saludables",
            "motivación", "motivación escolar", "motivación intrínseca",
            "bienestar", "bienestar emocional", "bienestar psicológico",
            "ansiedad", "ansiedad escolar", "estrés",
            "compromiso", "compromiso escolar", "engagement"
        ]
        
        for dimension in dimensiones_conocidas:
            if dimension in consulta:
                dimensiones.append(dimension)
        
        return list(set(dimensiones))  # Eliminar duplicados
    
    def _extraer_constructos(self, consulta: str) -> List[str]:
        """Extrae menciones a constructos psicológicos."""
        constructos = []
        
        constructos_conocidos = [
            "autoconcepto", "autoconfianza", "autoefficacia",
            "resiliencia", "empatía", "empatia", "asertividad",
            "cooperación", "responsabilidad", "perseverancia",
            "autorregulación", "metacognición", "metacognicion"
        ]
        
        for constructo in constructos_conocidos:
            if constructo in consulta:
                constructos.append(constructo)
        
        return constructos
    
    def _extraer_años(self, consulta: str) -> List[int]:
        """Extrae años mencionados en la consulta."""
        años = []
        
        # Patrón para años (2014-2030)
        patron_año = r"\b(20(?:1[4-9]|[2-9]\d))\b"
        años_encontrados = re.findall(patron_año, consulta)
        años = [int(año) for año in años_encontrados]
        
        # Buscar rangos de años
        patron_rango = r"(?:desde|del|de)\s+(\d{4})\s+(?:hasta|al|a)\s+(\d{4})"
        rangos = re.findall(patron_rango, consulta.lower())
        for inicio, fin in rangos:
            años.extend(range(int(inicio), int(fin) + 1))
        
        return list(set(años))
    
    def _extraer_formatos(self, consulta: str) -> List[str]:
        """Extrae formatos de respuesta mencionados."""
        formatos = []
        
        for formato in self.FORMATOS_RESPUESTA:
            if formato in consulta:
                formatos.append(formato)
        
        return formatos
    
    def _extraer_parametros(self, consulta: str) -> Dict[str, Any]:
        """Extrae parámetros psicométricos mencionados."""
        parametros = {}
        
        # Dificultad
        patron_dificultad = r"dificultad\s*(?:entre|de)?\s*([-\d.]+)\s*(?:y|a|y|hasta)?\s*([-\d.]+)?"
        match_dif = re.search(patron_dificultad, consulta)
        if match_dif:
            parametros["dificultad_min"] = float(match_dif.group(1))
            if match_dif.group(2):
                parametros["dificultad_max"] = float(match_dif.group(2))
        
        # Discriminación
        patron_disc = r"discriminaci[oó]n\s*(?:mayor|mínima|minima|mayor a|>?)\s*([\d.]+)"
        match_disc = re.search(patron_disc, consulta)
        if match_disc:
            parametros["discriminacion_min"] = float(match_disc.group(1))
        
        # Ajuste
        if "buen ajuste" in consulta or "buenos ajustes" in consulta:
            parametros["buen_ajuste"] = True
        
        return parametros
    
    def _extraer_poblacion(self, consulta: str) -> Optional[str]:
        """Extrae descripción de población objetivo."""
        patrones = [
            r"(?:para|dirigido a)\s+([\w\s]+?)(?:\s+(?:de|con|\.|,)|$)",
            r"poblaci[oó]n\s+(?:objetivo|de)?\s*[:\s]*([\w\s]+?)(?:\s+(?:de|con|\.|,)|$)",
        ]
        
        for patron in patrones:
            match = re.search(patron, consulta)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extraer_nivel_educativo(self, consulta: str) -> Optional[str]:
        """Extrae nivel educativo mencionado."""
        for nivel, variantes in self.NIVELES_EDUCATIVOS.items():
            for variante in variantes:
                if variante in consulta:
                    return nivel
        
        return None
    
    # ============================================
    # Métodos de preprocesamiento
    # ============================================
    
    def limpiar_texto(self, texto: str) -> str:
        """
        Limpia y normaliza texto para procesamiento.
        
        Args:
            texto: Texto a limpiar
            
        Returns:
            str: Texto limpio y normalizado
        """
        # Eliminar espacios extra
        texto = re.sub(r'\s+', ' ', texto)
        
        # Eliminar caracteres especiales innecesarios
        texto = re.sub(r'[^\w\s.,;:!?¿¡áéíóúÁÉÍÓÚñÑ-]', '', texto)
        
        return texto.strip()
    
    def normalizar_consulta(self, consulta: str) -> str:
        """
        Normaliza una consulta para búsqueda.
        
        Args:
            consulta: Consulta del usuario
            
        Returns:
            str: Consulta normalizada
        """
        # Convertir a minúsculas
        consulta = consulta.lower()
        
        # Eliminar signos de puntuación
        consulta = re.sub(r'[^\w\s]', ' ', consulta)
        
        # Normalizar espacios
        consulta = re.sub(r'\s+', ' ', consulta)
        
        return consulta.strip()
    
    def generar_variaciones_busqueda(self, consulta: str) -> List[str]:
        """
        Genera variaciones de una consulta para mejorar búsqueda.
        
        Args:
            consulta: Consulta original
            
        Returns:
            List[str]: Lista de variaciones de la consulta
        """
        variaciones = [consulta]
        
        # Variaciones con sinónimos comunes
        sinonimos = {
            "autoestima": ["autoconcepto", "autovaloración"],
            "buscar": ["encontrar", "localizar"],
            "ítem": ["pregunta", "ítem", "item"],
        }
        
        for palabra, alternativas in sinonimos.items():
            if palabra in consulta.lower():
                for alternativa in alternativas:
                    variacion = consulta.lower().replace(palabra, alternativa)
                    if variacion not in variaciones:
                        variaciones.append(variacion)
        
        return variaciones
    
    # ============================================
    # Métodos de análisis de texto
    # ============================================
    
    def analizar_sentimiento(self, texto: str) -> Dict[str, float]:
        """
        Análisis básico de sentimiento del texto.
        
        Args:
            texto: Texto a analizar
            
        Returns:
            Dict[str, float]: Scores de sentimiento
        """
        # Implementación simplificada
        palabras_positivas = ["bueno", "excelente", " óptimo", "mejor", "positivo"]
        palabras_negativas = ["malo", "pobre", "deficiente", "negativo", "problema"]
        
        texto_lower = texto.lower()
        
        positivas = sum(1 for p in palabras_positivas if p in texto_lower)
        negativas = sum(1 for n in palabras_negativas if n in texto_lower)
        total = positivas + negativas
        
        if total == 0:
            return {"positivo": 0.5, "negativo": 0.5, "neutral": 1.0}
        
        return {
            "positivo": positivas / total,
            "negativo": negativas / total,
            "neutral": 0.0 if total > 0 else 1.0
        }
    
    def es_consulta_valida(self, consulta: str) -> tuple[bool, Optional[str]]:
        """
        Valida si una consulta es válida para procesar.
        
        Args:
            consulta: Consulta a validar
            
        Returns:
            tuple[bool, Optional[str]]: (es_válida, mensaje_error)
        """
        if not consulta or len(consulta.strip()) < 5:
            return False, "La consulta debe tener al menos 5 caracteres"
        
        if len(consulta) > 5000:
            return False, "La consulta no puede exceder 5000 caracteres"
        
        # Verificar que no sea solo caracteres especiales
        if not re.search(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]', consulta):
            return False, "La consulta debe contener texto válido"
        
        return True, None
