"""
================================================================================
SERVICIO DE ANÁLISIS PSICOMÉTRICO
================================================================================

Servicio principal para el análisis de ítems psicométricos. Proporciona
funcionalidades para:
- Análisis individual de ítems con todos sus parámetros
- Comparación de múltiples ítems
- Identificación de ítems problemáticos
- Cálculo de estadísticas agregadas por dimensión

Este servicio integra el motor de reglas psicométricas con la capa de datos.

Autor: Agente de IA Psicométrico
Versión: 1.0.0
================================================================================
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import json

# Importar módulos del núcleo psicométrico
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.reglas_psicometricas import (
    validar_parametros_irt,
    evaluar_calidad_item,
    clasificar_dificultad,
    clasificar_discriminacion,
    detectar_problemas_ajuste,
    detectar_todos_problemas,
    generar_recomendacion,
    comparar_items,
    NivelDificultad,
    NivelDiscriminacion,
    CalidadItem,
    TipoProblema,
    UmbralesPsicometricos,
    calcular_probabilidad_2pl,
    calcular_informacion_item
)

from core.utils_psicometricos import (
    calcular_estadisticas_descriptivas,
    formatear_parametro_psicometrico,
    es_valor_valido
)


# ==============================================================================
# ESTRUCTURAS DE DATOS
# ==============================================================================

@dataclass
class DatosItem:
    """Estructura para almacenar datos completos de un ítem."""
    item_id: int
    texto: str
    dimension_id: int
    dimension_nombre: str
    formato_respuesta: str
    opciones_respuesta: List[str] = field(default_factory=list)
    valores_opciones: List[int] = field(default_factory=list)
    anio_creacion: Optional[int] = None
    activo: bool = True


@dataclass
class AnalisisItem:
    """Resultado completo del análisis de un ítem."""
    item_id: int
    texto_item: str
    dimension: str
    
    # Parámetros IRT
    param_a: float
    param_b: float
    param_c: Optional[float] = None
    modelo_irt: str = "2PL"
    
    # Índices de ajuste
    chi_cuadrado: Optional[float] = None
    p_valor: Optional[float] = None
    rmsea: Optional[float] = None
    cfi: Optional[float] = None
    
    # Confiabilidad
    confiabilidad_test_retest: Optional[float] = None
    
    # Clasificaciones
    nivel_dificultad: str = ""
    nivel_discriminacion: str = ""
    calidad_general: str = ""
    
    # Problemas y recomendaciones
    problemas: List[str] = field(default_factory=list)
    recomendaciones: List[str] = field(default_factory=list)
    
    # Metadatos
    fecha_analisis: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el análisis a diccionario."""
        return {
            "item_id": self.item_id,
            "texto_item": self.texto_item,
            "dimension": self.dimension,
            "parametros_irt": {
                "a_discriminacion": self.param_a,
                "b_dificultad": self.param_b,
                "c_adivinanza": self.param_c,
                "modelo": self.modelo_irt
            },
            "indices_ajuste": {
                "chi_cuadrado": self.chi_cuadrado,
                "p_valor": self.p_valor,
                "rmsea": self.rmsea,
                "cfi": self.cfi
            },
            "confiabilidad": {
                "test_retest": self.confiabilidad_test_retest
            },
            "clasificaciones": {
                "dificultad": self.nivel_dificultad,
                "discriminacion": self.nivel_discriminacion,
                "calidad_general": self.calidad_general
            },
            "problemas": self.problemas,
            "recomendaciones": self.recomendaciones,
            "fecha_analisis": self.fecha_analisis.isoformat()
        }


@dataclass
class EstadisticasDimension:
    """Estadísticas agregadas de una dimensión."""
    dimension_id: int
    dimension_nombre: str
    total_items: int
    items_activos: int
    
    # Estadísticas discriminación
    discriminacion_media: float
    discriminacion_desv_est: float
    discriminacion_min: float
    discriminacion_max: float
    
    # Estadísticas dificultad
    dificultad_media: float
    dificultad_desv_est: float
    dificultad_min: float
    dificultad_max: float
    
    # Distribución de calidad
    items_excelentes: int
    items_buenos: int
    items_aceptables: int
    items_cuestionables: int
    items_deficientes: int
    
    # Problemas
    items_con_problemas: int
    problemas_frecuentes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte las estadísticas a diccionario."""
        return {
            "dimension_id": self.dimension_id,
            "dimension_nombre": self.dimension_nombre,
            "conteos": {
                "total_items": self.total_items,
                "items_activos": self.items_activos,
                "items_con_problemas": self.items_con_problemas
            },
            "discriminacion": {
                "media": round(self.discriminacion_media, 4),
                "desv_est": round(self.discriminacion_desv_est, 4),
                "min": round(self.discriminacion_min, 4),
                "max": round(self.discriminacion_max, 4)
            },
            "dificultad": {
                "media": round(self.dificultad_media, 4),
                "desv_est": round(self.dificultad_desv_est, 4),
                "min": round(self.dificultad_min, 4),
                "max": round(self.dificultad_max, 4)
            },
            "distribucion_calidad": {
                "excelentes": self.items_excelentes,
                "buenos": self.items_buenos,
                "aceptables": self.items_aceptables,
                "cuestionables": self.items_cuestionables,
                "deficientes": self.items_deficientes
            },
            "problemas_frecuentes": self.problemas_frecuentes
        }


# ==============================================================================
# SERVICIO DE ANÁLISIS
# ==============================================================================

class ServicioAnalisisPsicometrico:
    """
    Servicio principal para análisis psicométrico de ítems.
    
    Esta clase proporciona métodos para analizar ítems individuales,
    comparar múltiples ítems, identificar problemas y calcular
    estadísticas agregadas.
    """
    
    def __init__(self, db_connection=None):
        """
        Inicializa el servicio de análisis.
        
        Args:
            db_connection: Conexión opcional a la base de datos
        """
        self.db = db_connection
        self.cache_analisis: Dict[int, AnalisisItem] = {}
    
    # --------------------------------------------------------------------------
    # ANÁLISIS INDIVIDUAL
    # --------------------------------------------------------------------------
    
    def analizar_item(
        self,
        item_id: int,
        datos_item: Optional[DatosItem] = None,
        parametros: Optional[Dict[str, float]] = None
    ) -> AnalisisItem:
        """
        Analiza un ítem específico con todos sus parámetros psicométricos.
        
        Args:
            item_id: Identificador del ítem
            datos_item: Datos del ítem (texto, dimensión, etc.)
            parametros: Parámetros IRT y de ajuste
        
        Returns:
            AnalisisItem: Resultado completo del análisis
        
        Ejemplo:
            >>> servicio = ServicioAnalisisPsicometrico()
            >>> datos = DatosItem(item_id=1, texto="¿Me siento...", ...)
            >>> params = {"param_a": 1.2, "param_b": 0.5, "p_valor": 0.25}
            >>> resultado = servicio.analizar_item(1, datos, params)
        """
        # Verificar caché
        if item_id in self.cache_analisis:
            return self.cache_analisis[item_id]
        
        # Obtener datos si no se proporcionaron
        if datos_item is None and self.db:
            datos_item = self._obtener_datos_item_db(item_id)
        
        if parametros is None and self.db:
            parametros = self._obtener_parametros_db(item_id)
        
        # Valores por defecto si no hay datos
        if datos_item is None:
            datos_item = DatosItem(
                item_id=item_id,
                texto="Ítem no encontrado",
                dimension_id=0,
                dimension_nombre="Desconocida",
                formato_respuesta="likert_4"
            )
        
        if parametros is None:
            parametros = {}
        
        # Extraer parámetros
        param_a = parametros.get('param_a', 0)
        param_b = parametros.get('param_b', 0)
        param_c = parametros.get('param_c')
        p_valor = parametros.get('p_valor')
        chi_cuadrado = parametros.get('chi_cuadrado')
        confiabilidad = parametros.get('confiabilidad_test_retest')
        modelo = parametros.get('modelo_irt', '2PL')
        
        # Validar parámetros
        es_valido, errores = validar_parametros_irt(param_a, param_b, param_c, modelo)
        
        # Clasificar
        nivel_dif = clasificar_dificultad(param_b)
        nivel_disc = clasificar_discriminacion(param_a)
        calidad = evaluar_calidad_item(
            param_a=param_a,
            param_b=param_b,
            p_valor=p_valor,
            param_c=param_c,
            confiabilidad_test_retest=confiabilidad
        )
        
        # Detectar problemas
        problemas_tipos = detectar_todos_problemas(
            param_a=param_a,
            param_b=param_b,
            p_valor=p_valor,
            param_c=param_c,
            confiabilidad=confiabilidad
        )
        problemas_str = [p.value for p in problemas_tipos]
        
        # Generar recomendaciones
        item_data = {
            'param_a': param_a,
            'param_b': param_b,
            'p_valor': p_valor,
            'param_c': param_c,
            'confiabilidad_test_retest': confiabilidad
        }
        recomendaciones = generar_recomendacion(item_data)
        
        # Agregar errores de validación a problemas
        if not es_valido:
            problemas_str.extend(errores)
        
        # Crear resultado
        analisis = AnalisisItem(
            item_id=item_id,
            texto_item=datos_item.texto,
            dimension=datos_item.dimension_nombre,
            param_a=param_a,
            param_b=param_b,
            param_c=param_c,
            modelo_irt=modelo,
            chi_cuadrado=chi_cuadrado,
            p_valor=p_valor,
            confiabilidad_test_retest=confiabilidad,
            nivel_dificultad=nivel_dif.value,
            nivel_discriminacion=nivel_disc.value,
            calidad_general=calidad.value,
            problemas=problemas_str,
            recomendaciones=recomendaciones
        )
        
        # Guardar en caché
        self.cache_analisis[item_id] = analisis
        
        return analisis
    
    def analizar_item_completo(
        self,
        item_id: int,
        incluir_historial: bool = False
    ) -> Dict[str, Any]:
        """
        Realiza un análisis completo incluyendo información adicional.
        
        Args:
            item_id: Identificador del ítem
            incluir_historial: Si se debe incluir el historial de aplicaciones
        
        Returns:
            Diccionario con análisis completo
        """
        # Análisis básico
        analisis = self.analizar_item(item_id)
        
        resultado = analisis.to_dict()
        
        # Agregar información de función de información
        resultado["funcion_informacion"] = self._calcular_funcion_informacion(
            analisis.param_a,
            analisis.param_b,
            analisis.param_c
        )
        
        # Agregar curva característica del ítem
        resultado["curva_caracteristica"] = self._calcular_curva_caracteristica(
            analisis.param_a,
            analisis.param_b,
            analisis.param_c
        )
        
        # Agregar historial si se solicita
        if incluir_historial and self.db:
            resultado["historial"] = self._obtener_historial_item(item_id)
        
        return resultado
    
    # --------------------------------------------------------------------------
    # ANÁLISIS COMPARATIVO
    # --------------------------------------------------------------------------
    
    def comparar_items(
        self,
        item_ids: List[int],
        incluir_textos: bool = True
    ) -> Dict[str, Any]:
        """
        Compara múltiples ítems psicométricamente.
        
        Args:
            item_ids: Lista de identificadores de ítems
            incluir_textos: Si se deben incluir los textos en la comparación
        
        Returns:
            Diccionario con el análisis comparativo
        
        Ejemplo:
            >>> servicio = ServicioAnalisisPsicometrico()
            >>> resultado = servicio.comparar_items([1, 2, 3])
        """
        if len(item_ids) < 2:
            return {"error": "Se necesitan al menos 2 ítems para comparar"}
        
        # Analizar todos los ítems
        analisis_items = []
        for item_id in item_ids:
            try:
                analisis = self.analizar_item(item_id)
                analisis_items.append(analisis)
            except Exception as e:
                return {"error": f"Error analizando ítem {item_id}: {str(e)}"}
        
        # Calcular estadísticas comparativas
        discriminaciones = [a.param_a for a in analisis_items]
        dificultades = [a.param_b for a in analisis_items]
        
        # Comparaciones por pares
        comparaciones_pares = []
        for i in range(len(analisis_items)):
            for j in range(i + 1, len(analisis_items)):
                item1 = analisis_items[i]
                item2 = analisis_items[j]
                
                comp = comparar_items(
                    {
                        'param_a': item1.param_a,
                        'param_b': item1.param_b,
                        'texto': item1.texto_item if incluir_textos else ""
                    },
                    {
                        'param_a': item2.param_a,
                        'param_b': item2.param_b,
                        'texto': item2.texto_item if incluir_textos else ""
                    }
                )
                
                comparaciones_pares.append({
                    "item1_id": item1.item_id,
                    "item2_id": item2.item_id,
                    "comparacion": comp
                })
        
        # Resultado
        return {
            "items_analizados": [a.item_id for a in analisis_items],
            "resumen": {
                "discriminacion": {
                    "media": round(statistics.mean(discriminaciones), 4),
                    "desv_est": round(statistics.stdev(discriminaciones), 4) if len(discriminaciones) > 1 else 0,
                    "rango": (round(min(discriminaciones), 4), round(max(discriminaciones), 4))
                },
                "dificultad": {
                    "media": round(statistics.mean(dificultades), 4),
                    "desv_est": round(statistics.stdev(dificultades), 4) if len(dificultades) > 1 else 0,
                    "rango": (round(min(dificultades), 4), round(max(dificultades), 4))
                }
            },
            "comparaciones_pares": comparaciones_pares,
            "items": [a.to_dict() for a in analisis_items]
        }
    
    # --------------------------------------------------------------------------
    # IDENTIFICACIÓN DE ÍTEMS PROBLEMÁTICOS
    # --------------------------------------------------------------------------
    
    def identificar_items_problematicos(
        self,
        dimension_id: Optional[int] = None,
        umbral_discriminacion: float = 0.5,
        umbral_p_valor: float = 0.05,
        solo_activos: bool = True
    ) -> Dict[str, Any]:
        """
        Identifica ítems con problemas psicométricos.
        
        Args:
            dimension_id: Filtrar por dimensión (None = todas)
            umbral_discriminacion: Umbral mínimo de discriminación
            umbral_p_valor: Umbral máximo de p-valor
            solo_activos: Solo ítems activos
        
        Returns:
            Diccionario con ítems problemáticos clasificados
        
        Ejemplo:
            >>> servicio = ServicioAnalisisPsicometrico()
            >>> problemas = servicio.identificar_items_problematicos(dimension_id=1)
        """
        # Obtener ítems a analizar
        if self.db:
            items = self._obtener_items_db(dimension_id, solo_activos)
        else:
            # Modo demo sin base de datos
            return {"error": "No hay conexión a base de datos disponible"}
        
        # Clasificar problemas
        problemas_criticos = []
        problemas_altos = []
        problemas_medios = []
        problemas_bajos = []
        
        for item_data in items:
            item_id = item_data.get('item_id')
            
            try:
                analisis = self.analizar_item(item_id)
                
                # Determinar severidad
                tiene_problema_critico = False
                tiene_problema_alto = False
                tiene_problema_medio = False
                
                # Discriminación negativa o muy baja
                if analisis.param_a < 0:
                    tiene_problema_critico = True
                elif analisis.param_a < umbral_discriminacion:
                    tiene_problema_alto = True
                
                # Mal ajuste
                if analisis.p_valor is not None and analisis.p_valor < umbral_p_valor:
                    if analisis.p_valor < 0.01:
                        tiene_problema_critico = True
                    else:
                        tiene_problema_alto = True
                
                # Dificultad extrema
                if abs(analisis.param_b) > 3.0:
                    tiene_problema_medio = True
                
                # Confiabilidad baja
                if analisis.confiabilidad_test_retest is not None:
                    if analisis.confiabilidad_test_retest < 0.60:
                        tiene_problema_alto = True
                    elif analisis.confiabilidad_test_retest < 0.70:
                        tiene_problema_medio = True
                
                # Clasificar
                item_resumen = {
                    "item_id": item_id,
                    "texto": analisis.texto_item[:100] + "..." if len(analisis.texto_item) > 100 else analisis.texto_item,
                    "param_a": analisis.param_a,
                    "param_b": analisis.param_b,
                    "p_valor": analisis.p_valor,
                    "problemas": analisis.problemas[:3]  # Top 3 problemas
                }
                
                if tiene_problema_critico:
                    problemas_criticos.append(item_resumen)
                elif tiene_problema_alto:
                    problemas_altos.append(item_resumen)
                elif tiene_problema_medio:
                    problemas_medios.append(item_resumen)
                elif len(analisis.problemas) > 0:
                    problemas_bajos.append(item_resumen)
                
            except Exception as e:
                continue  # Ignorar errores individuales
        
        return {
            "filtros_aplicados": {
                "dimension_id": dimension_id,
                "umbral_discriminacion": umbral_discriminacion,
                "umbral_p_valor": umbral_p_valor,
                "solo_activos": solo_activos
            },
            "resumen": {
                "total_items_revisados": len(items),
                "items_con_problemas": len(problemas_criticos) + len(problemas_altos) + len(problemas_medios) + len(problemas_bajos),
                "criticos": len(problemas_criticos),
                "altos": len(problemas_altos),
                "medios": len(problemas_medios),
                "bajos": len(problemas_bajos)
            },
            "problemas": {
                "criticos": problemas_criticos,
                "altos": problemas_altos,
                "medios": problemas_medios,
                "bajos": problemas_bajos
            }
        }
    
    # --------------------------------------------------------------------------
    # ESTADÍSTICAS POR DIMENSIÓN
    # --------------------------------------------------------------------------
    
    def calcular_estadisticas_dimension(
        self,
        dimension_id: int,
        anio: Optional[int] = None
    ) -> EstadisticasDimension:
        """
        Calcula estadísticas agregadas por dimensión.
        
        Args:
            dimension_id: Identificador de la dimensión
            anio: Año de análisis (None = más reciente)
        
        Returns:
            EstadisticasDimension: Estadísticas de la dimensión
        
        Ejemplo:
            >>> servicio = ServicioAnalisisPsicometrico()
            >>> stats = servicio.calcular_estadisticas_dimension(1)
        """
        # Obtener ítems de la dimensión
        if self.db:
            items = self._obtener_items_dimension_db(dimension_id, anio)
        else:
            return EstadisticasDimension(
                dimension_id=dimension_id,
                dimension_nombre="Sin conexión a DB",
                total_items=0,
                items_activos=0,
                discriminacion_media=0,
                discriminacion_desv_est=0,
                discriminacion_min=0,
                discriminacion_max=0,
                dificultad_media=0,
                dificultad_desv_est=0,
                dificultad_min=0,
                dificultad_max=0,
                items_excelentes=0,
                items_buenos=0,
                items_aceptables=0,
                items_cuestionables=0,
                items_deficientes=0,
                items_con_problemas=0
            )
        
        if not items:
            return EstadisticasDimension(
                dimension_id=dimension_id,
                dimension_nombre="Sin datos",
                total_items=0,
                items_activos=0,
                discriminacion_media=0,
                discriminacion_desv_est=0,
                discriminacion_min=0,
                discriminacion_max=0,
                dificultad_media=0,
                dificultad_desv_est=0,
                dificultad_min=0,
                dificultad_max=0,
                items_excelentes=0,
                items_buenos=0,
                items_aceptables=0,
                items_cuestionables=0,
                items_deficientes=0,
                items_con_problemas=0
            )
        
        # Analizar todos los ítems
        analisis_items = []
        for item_data in items:
            try:
                analisis = self.analizar_item(item_data['item_id'])
                analisis_items.append(analisis)
            except:
                continue
        
        if not analisis_items:
            return EstadisticasDimension(
                dimension_id=dimension_id,
                dimension_nombre=items[0].get('dimension_nombre', 'Desconocida'),
                total_items=len(items),
                items_activos=sum(1 for i in items if i.get('activo', True)),
                discriminacion_media=0,
                discriminacion_desv_est=0,
                discriminacion_min=0,
                discriminacion_max=0,
                dificultad_media=0,
                dificultad_desv_est=0,
                dificultad_min=0,
                dificultad_max=0,
                items_excelentes=0,
                items_buenos=0,
                items_aceptables=0,
                items_cuestionables=0,
                items_deficientes=0,
                items_con_problemas=0
            )
        
        # Calcular estadísticas
        discriminaciones = [a.param_a for a in analisis_items]
        dificultades = [a.param_b for a in analisis_items]
        
        # Contar por calidad
        calidades = {'excelente': 0, 'buena': 0, 'aceptable': 0, 'cuestionable': 0, 'deficiente': 0}
        for a in analisis_items:
            calidades[a.calidad_general] = calidades.get(a.calidad_general, 0) + 1
        
        # Contar problemas
        items_con_problemas = sum(1 for a in analisis_items if len(a.problemas) > 0)
        
        # Problemas frecuentes
        conteo_problemas = {}
        for a in analisis_items:
            for p in a.problemas:
                conteo_problemas[p] = conteo_problemas.get(p, 0) + 1
        
        problemas_frecuentes = [
            {"tipo": k, "frecuencia": v, "porcentaje": round(v / len(analisis_items) * 100, 2)}
            for k, v in sorted(conteo_problemas.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        return EstadisticasDimension(
            dimension_id=dimension_id,
            dimension_nombre=items[0].get('dimension_nombre', 'Desconocida'),
            total_items=len(items),
            items_activos=sum(1 for i in items if i.get('activo', True)),
            discriminacion_media=statistics.mean(discriminaciones),
            discriminacion_desv_est=statistics.stdev(discriminaciones) if len(discriminaciones) > 1 else 0,
            discriminacion_min=min(discriminaciones),
            discriminacion_max=max(discriminaciones),
            dificultad_media=statistics.mean(dificultades),
            dificultad_desv_est=statistics.stdev(dificultades) if len(dificultades) > 1 else 0,
            dificultad_min=min(dificultades),
            dificultad_max=max(dificultades),
            items_excelentes=calidades.get('excelente', 0),
            items_buenos=calidades.get('buena', 0),
            items_aceptables=calidades.get('aceptable', 0),
            items_cuestionables=calidades.get('cuestionable', 0),
            items_deficientes=calidades.get('deficiente', 0),
            items_con_problemas=items_con_problemas,
            problemas_frecuentes=problemas_frecuentes
        )
    
    # --------------------------------------------------------------------------
    # MÉTODOS PRIVADOS
    # --------------------------------------------------------------------------
    
    def _calcular_funcion_informacion(
        self,
        param_a: float,
        param_b: float,
        param_c: Optional[float] = None,
        rango_theta: Tuple[float, float] = (-4, 4),
        paso: float = 0.5
    ) -> List[Dict[str, float]]:
        """Calcula la función de información del ítem."""
        puntos = []
        theta = rango_theta[0]
        while theta <= rango_theta[1]:
            info = calcular_informacion_item(theta, param_a, param_b, param_c)
            puntos.append({"theta": round(theta, 2), "informacion": round(info, 4)})
            theta += paso
        return puntos
    
    def _calcular_curva_caracteristica(
        self,
        param_a: float,
        param_b: float,
        param_c: Optional[float] = None,
        rango_theta: Tuple[float, float] = (-4, 4),
        paso: float = 0.5
    ) -> List[Dict[str, float]]:
        """Calcula la curva característica del ítem."""
        puntos = []
        theta = rango_theta[0]
        while theta <= rango_theta[1]:
            if param_c is not None:
                from core.reglas_psicometricas import calcular_probabilidad_3pl
                prob = calcular_probabilidad_3pl(theta, param_a, param_b, param_c)
            else:
                prob = calcular_probabilidad_2pl(theta, param_a, param_b)
            puntos.append({"theta": round(theta, 2), "probabilidad": round(prob, 4)})
            theta += paso
        return puntos
    
    def _obtener_datos_item_db(self, item_id: int) -> Optional[DatosItem]:
        """Obtiene datos del ítem desde la base de datos."""
        # Implementación simulada - reemplazar con consulta real
        return None
    
    def _obtener_parametros_db(self, item_id: int) -> Optional[Dict[str, float]]:
        """Obtiene parámetros psicométricos desde la base de datos."""
        # Implementación simulada - reemplazar con consulta real
        return None
    
    def _obtener_items_db(
        self,
        dimension_id: Optional[int] = None,
        solo_activos: bool = True
    ) -> List[Dict[str, Any]]:
        """Obtiene lista de ítems desde la base de datos."""
        # Implementación simulada - reemplazar con consulta real
        return []
    
    def _obtener_items_dimension_db(
        self,
        dimension_id: int,
        anio: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Obtiene ítems de una dimensión desde la base de datos."""
        # Implementación simulada - reemplazar con consulta real
        return []
    
    def _obtener_historial_item(self, item_id: int) -> List[Dict[str, Any]]:
        """Obtiene el historial de aplicaciones de un ítem."""
        # Implementación simulada - reemplazar con consulta real
        return []


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def generar_reporte_analisis(
    analisis_items: List[AnalisisItem],
    formato: str = "json"
) -> Union[str, Dict[str, Any]]:
    """
    Genera un reporte de análisis de múltiples ítems.
    
    Args:
        analisis_items: Lista de análisis de ítems
        formato: Formato de salida ("json", "dict")
    
    Returns:
        Reporte en el formato especificado
    """
    reporte = {
        "total_items": len(analisis_items),
        "resumen_calidad": {},
        "distribucion_dificultad": {},
        "distribucion_discriminacion": {},
        "items": []
    }
    
    # Contar por calidad
    for calidad in CalidadItem:
        count = sum(1 for a in analisis_items if a.calidad_general == calidad.value)
        reporte["resumen_calidad"][calidad.value] = count
    
    # Contar por dificultad
    for nivel in NivelDificultad:
        count = sum(1 for a in analisis_items if a.nivel_dificultad == nivel.value)
        reporte["distribucion_dificultad"][nivel.value] = count
    
    # Contar por discriminación
    for nivel in NivelDiscriminacion:
        count = sum(1 for a in analisis_items if a.nivel_discriminacion == nivel.value)
        reporte["distribucion_discriminacion"][nivel.value] = count
    
    # Agregar items
    reporte["items"] = [a.to_dict() for a in analisis_items]
    
    if formato == "json":
        return json.dumps(reporte, indent=2, ensure_ascii=False)
    return reporte


# ==============================================================================
# EXPORTAR
# ==============================================================================

__all__ = [
    'ServicioAnalisisPsicometrico',
    'DatosItem',
    'AnalisisItem',
    'EstadisticasDimension',
    'generar_reporte_analisis'
]
