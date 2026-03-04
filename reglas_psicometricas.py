"""
================================================================================
MOTOR DE REGLAS PSICOMÉTRICAS
================================================================================

Módulo central que implementa las reglas y algoritmos psicométricos para el
Agente de IA especializado en psicometría educativa.

Este motor encapsula el conocimiento psicométrico y proporciona funciones para:
- Validación de parámetros IRT (2PL, GRM)
- Evaluación de calidad de ítems
- Clasificación por dificultad y discriminación
- Detección de problemas de ajuste
- Generación de recomendaciones

Autor: Agente de IA Psicométrico
Versión: 1.0.0
================================================================================
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math

# ==============================================================================
# CONSTANTES Y UMBRALES PSICOMÉTRICOS
# ==============================================================================

class UmbralesPsicometricos:
    """Constantes y umbrales para evaluación psicométrica."""
    
    # Rangos para parámetro de discriminación (param_a)
    DISCRIMINACION_BAJA_MIN = 0.0
    DISCRIMINACION_BAJA_MAX = 0.64
    DISCRIMINACION_MODERADA_MIN = 0.65
    DISCRIMINACION_MODERADA_MAX = 1.34
    DISCRIMINACION_ALTA_MIN = 1.35
    DISCRIMINACION_ALTA_MAX = 1.69
    DISCRIMINACION_MUY_ALTA_MIN = 1.70
    
    # Rangos para parámetro de dificultad (param_b) en escala logit
    DIFICULTAD_MUY_FACIL_MAX = -2.0
    DIFICULTAD_FACIL_MAX = -1.0
    DIFICULTAD_MODERADA_MIN = -1.0
    DIFICULTAD_MODERADA_MAX = 1.0
    DIFICULTAD_DIFICIL_MIN = 1.0
    DIFICULTAD_DIFICIL_MAX = 2.0
    DIFICULTAD_MUY_DIFICIL_MIN = 2.0
    
    # Rangos para parámetro de adivinanza (param_c) - modelo 3PL
    ADIVINANZA_MIN = 0.0
    ADIVINANZA_MAX = 0.50
    ADIVINANZA_ACEPTABLE_MAX = 0.35
    
    # Umbrales para índices de ajuste
    P_VALOR_UMBRAL_SIGNIFICATIVO = 0.05
    P_VALOR_UMBRAL_MUY_SIGNIFICATIVO = 0.01
    
    # Rangos de confiabilidad
    CONFIABILIDAD_BAJA_MAX = 0.70
    CONFIABILIDAD_MODERADA_MIN = 0.70
    CONFIABILIDAD_MODERADA_MAX = 0.80
    CONFIABILIDAD_ALTA_MIN = 0.80
    CONFIABILIDAD_MUY_ALTA_MIN = 0.90


# ==============================================================================
# ENUMERACIONES PARA CLASIFICACIÓN
# ==============================================================================

class NivelDificultad(Enum):
    """Niveles de dificultad de un ítem según parámetro b."""
    MUY_FACIL = "muy_facil"
    FACIL = "facil"
    MODERADA = "moderada"
    DIFICIL = "dificil"
    MUY_DIFICIL = "muy_dificil"


class NivelDiscriminacion(Enum):
    """Niveles de discriminación de un ítem según parámetro a."""
    BAJA = "baja"
    MODERADA = "moderada"
    ALTA = "alta"
    MUY_ALTA = "muy_alta"


class CalidadItem(Enum):
    """Clasificación general de calidad de un ítem."""
    EXCELENTE = "excelente"
    BUENA = "buena"
    ACEPTABLE = "aceptable"
    CUESTIONABLE = "cuestionable"
    DEFICIENTE = "deficiente"


class TipoProblema(Enum):
    """Tipos de problemas que puede tener un ítem."""
    DISCRIMINACION_BAJA = "discriminacion_baja"
    DISCRIMINACION_NEGATIVA = "discriminacion_negativa"
    DIFICULTAD_EXTREMA = "dificultad_extrema"
    MAL_AJUSTE = "mal_ajuste"
    ADIVINANZA_ALTA = "adivinanza_alta"
    CONFIABILIDAD_BAJA = "confiabilidad_baja"


# ==============================================================================
# ESTRUCTURAS DE DATOS
# ==============================================================================

@dataclass
class ParametrosIRT:
    """Estructura para almacenar parámetros IRT de un ítem."""
    param_a: float  # Discriminación
    param_b: float  # Dificultad
    param_c: Optional[float] = None  # Adivinanza (solo para 3PL)
    modelo: str = "2PL"  # Modelo IRT utilizado
    
    def __post_init__(self):
        """Validación post-inicialización."""
        if self.modelo == "3PL" and self.param_c is None:
            raise ValueError("El modelo 3PL requiere el parámetro c (adivinanza)")


@dataclass
class IndicesAjuste:
    """Estructura para almacenar índices de ajuste del modelo."""
    chi_cuadrado: Optional[float] = None
    p_valor: Optional[float] = None
    grados_libertad: Optional[int] = None
    rmsea: Optional[float] = None  # Root Mean Square Error of Approximation
    cfi: Optional[float] = None  # Comparative Fit Index
    tli: Optional[float] = None  # Tucker-Lewis Index
    
    @property
    def ajuste_aceptable(self) -> bool:
        """Determina si el ajuste es estadísticamente aceptable."""
        if self.p_valor is None:
            return True  # Sin información, asumimos aceptable
        return self.p_valor >= UmbralesPsicometricos.P_VALOR_UMBRAL_SIGNIFICATIVO


@dataclass
class ResultadoEvaluacion:
    """Resultado completo de la evaluación de un ítem."""
    item_id: int
    parametros: ParametrosIRT
    indices_ajuste: IndicesAjuste
    nivel_dificultad: NivelDificultad
    nivel_discriminacion: NivelDiscriminacion
    calidad_general: CalidadItem
    problemas_detectados: List[TipoProblema]
    recomendaciones: List[str]
    confiabilidad_test_retest: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario."""
        return {
            "item_id": self.item_id,
            "parametros": {
                "a_discriminacion": self.parametros.param_a,
                "b_dificultad": self.parametros.param_b,
                "c_adivinanza": self.parametros.param_c,
                "modelo": self.parametros.modelo
            },
            "indices_ajuste": {
                "chi_cuadrado": self.indices_ajuste.chi_cuadrado,
                "p_valor": self.indices_ajuste.p_valor,
                "rmsea": self.indices_ajuste.rmsea,
                "cfi": self.indices_ajuste.cfi,
                "tli": self.indices_ajuste.tli
            },
            "clasificacion": {
                "nivel_dificultad": self.nivel_dificultad.value,
                "nivel_discriminacion": self.nivel_discriminacion.value,
                "calidad_general": self.calidad_general.value
            },
            "problemas": [p.value for p in self.problemas_detectados],
            "recomendaciones": self.recomendaciones,
            "confiabilidad_test_retest": self.confiabilidad_test_retest
        }


# ==============================================================================
# FUNCIONES DE VALIDACIÓN
# ==============================================================================

def validar_parametros_irt(
    param_a: float,
    param_b: float,
    param_c: Optional[float] = None,
    modelo: str = "2PL"
) -> Tuple[bool, List[str]]:
    """
    Valida que los parámetros IRT estén en rangos aceptables.
    
    Args:
        param_a: Parámetro de discriminación
        param_b: Parámetro de dificultad (logits)
        param_c: Parámetro de adivinanza (solo para modelo 3PL)
        modelo: Modelo IRT utilizado ("2PL", "3PL", "GRM")
    
    Returns:
        Tuple con (es_valido, lista_de_errores)
    
    Ejemplo:
        >>> validar_parametros_irt(1.2, 0.5)
        (True, [])
        >>> validar_parametros_irt(-0.5, 0.5)
        (False, ['Discriminación negativa o muy baja: -0.5'])
    """
    errores = []
    
    # Validar discriminación (param_a)
    if param_a <= 0:
        errores.append(f"Discriminación negativa o nula: {param_a}. "
                      f"Debe ser positiva para un ítem válido.")
    elif param_a < UmbralesPsicometricos.DISCRIMINACION_BAJA_MAX:
        # Es baja pero válida - solo advertencia
        pass
    elif param_a > 3.0:
        errores.append(f"Discriminación extremadamente alta: {param_a}. "
                      f"Verificar posible sobreajuste.")
    
    # Validar dificultad (param_b)
    if abs(param_b) > 4.0:
        errores.append(f"Dificultad extrema: {param_b}. "
                      f"Considerar revisar el ítem o la muestra utilizada.")
    
    # Validar adivinanza (param_c) solo para modelo 3PL
    if modelo == "3PL":
        if param_c is None:
            errores.append("El modelo 3PL requiere el parámetro c (adivinanza)")
        else:
            if param_c < UmbralesPsicometricos.ADIVINANZA_MIN:
                errores.append(f"Adivinanza negativa: {param_c}")
            elif param_c > UmbralesPsicometricos.ADIVINANZA_MAX:
                errores.append(f"Adivinanza demasiado alta: {param_c}. "
                              f"Máximo recomendado: {UmbralesPsicometricos.ADIVINANZA_MAX}")
    
    # Validar modelo
    modelos_validos = ["2PL", "3PL", "GRM", "1PL"]
    if modelo not in modelos_validos:
        errores.append(f"Modelo no reconocido: {modelo}. "
                      f"Modelos válidos: {', '.join(modelos_validos)}")
    
    return len(errores) == 0, errores


def validar_rango_parametro_a(param_a: float) -> Tuple[bool, str]:
    """
    Valida específicamente el parámetro de discriminación.
    
    Returns:
        Tuple con (es_valido, mensaje)
    """
    if param_a <= 0:
        return False, f"Discriminación inválida: {param_a} (debe ser > 0)"
    if param_a < 0.3:
        return True, f"Discriminación muy baja: {param_a}"
    return True, f"Discriminación válida: {param_a}"


def validar_rango_parametro_b(param_b: float) -> Tuple[bool, str]:
    """
    Valida específicamente el parámetro de dificultad.
    
    Returns:
        Tuple con (es_valido, mensaje)
    """
    if abs(param_b) > 4.0:
        return False, f"Dificultad extrema: {param_b} (rango recomendado: -4 a 4)"
    return True, f"Dificultad válida: {param_b}"


# ==============================================================================
# FUNCIONES DE CLASIFICACIÓN
# ==============================================================================

def clasificar_dificultad(param_b: float) -> NivelDificultad:
    """
    Clasifica un ítem por su nivel de dificultad según el parámetro b.
    
    La dificultad en IRT se mide en logits, donde:
    - Valores negativos indican ítems más fáciles
    - Valores positivos indican ítems más difíciles
    - 0 representa dificultad media (50% de probabilidad de acierto
      para persona con habilidad promedio)
    
    Args:
        param_b: Parámetro de dificultad (logits)
    
    Returns:
        NivelDificultad: Clasificación del nivel de dificultad
    
    Ejemplo:
        >>> clasificar_dificultad(-2.5)
        <NivelDificultad.MUY_FACIL: 'muy_facil'>
        >>> clasificar_dificultad(0.5)
        <NivelDificultad.MODERADA: 'moderada'>
    """
    umbrales = UmbralesPsicometricos
    
    if param_b <= umbrales.DIFICULTAD_MUY_FACIL_MAX:
        return NivelDificultad.MUY_FACIL
    elif param_b <= umbrales.DIFICULTAD_FACIL_MAX:
        return NivelDificultad.FACIL
    elif param_b <= umbrales.DIFICULTAD_MODERADA_MAX:
        return NivelDificultad.MODERADA
    elif param_b <= umbrales.DIFICULTAD_DIFICIL_MAX:
        return NivelDificultad.DIFICIL
    else:
        return NivelDificultad.MUY_DIFICIL


def clasificar_discriminacion(param_a: float) -> NivelDiscriminacion:
    """
    Clasifica un ítem por su nivel de discriminación según el parámetro a.
    
    La discriminación indica qué tan bien el ítem diferencia entre
    personas con niveles altos y bajos del constructo medido.
    
    Args:
        param_a: Parámetro de discriminación
    
    Returns:
        NivelDiscriminacion: Clasificación del nivel de discriminación
    
    Ejemplo:
        >>> clasificar_discriminacion(0.5)
        <NivelDiscriminacion.BAJA: 'baja'>
        >>> clasificar_discriminacion(1.5)
        <NivelDiscriminacion.ALTA: 'alta'>
    """
    umbrales = UmbralesPsicometricos
    
    if param_a < umbrales.DISCRIMINACION_MODERADA_MIN:
        return NivelDiscriminacion.BAJA
    elif param_a < umbrales.DISCRIMINACION_ALTA_MIN:
        return NivelDiscriminacion.MODERADA
    elif param_a < umbrales.DISCRIMINACION_MUY_ALTA_MIN:
        return NivelDiscriminacion.ALTA
    else:
        return NivelDiscriminacion.MUY_ALTA


def obtener_descripcion_dificultad(nivel: NivelDificultad) -> str:
    """Obtiene una descripción textual del nivel de dificultad."""
    descripciones = {
        NivelDificultad.MUY_FACIL: "Ítem muy fácil. La mayoría de los estudiantes responden correctamente.",
        NivelDificultad.FACIL: "Ítem fácil. Estudiantes con habilidad baja-media responden correctamente.",
        NivelDificultad.MODERADA: "Ítem de dificultad moderada. Diferencia bien entre niveles de habilidad.",
        NivelDificultad.DIFICIL: "Ítem difícil. Solo estudiantes con alta habilidad responden correctamente.",
        NivelDificultad.MUY_DIFICIL: "Ítem muy difícil. Muy pocos estudiantes responden correctamente."
    }
    return descripciones.get(nivel, "Nivel de dificultad no clasificado")


def obtener_descripcion_discriminacion(nivel: NivelDiscriminacion) -> str:
    """Obtiene una descripción textual del nivel de discriminación."""
    descripciones = {
        NivelDiscriminacion.BAJA: "Discriminación baja. El ítem diferencia poco entre niveles de habilidad.",
        NivelDiscriminacion.MODERADA: "Discriminación moderada. El ítem diferencia adecuadamente.",
        NivelDiscriminacion.ALTA: "Discriminación alta. El ítem diferencia muy bien entre niveles.",
        NivelDiscriminacion.MUY_ALTA: "Discriminación muy alta. Excelente capacidad diferenciadora."
    }
    return descripciones.get(nivel, "Nivel de discriminación no clasificado")


# ==============================================================================
# FUNCIONES DE DETECCIÓN DE PROBLEMAS
# ==============================================================================

def detectar_problemas_ajuste(
    p_valor: Optional[float],
    umbral: float = UmbralesPsicometricos.P_VALOR_UMBRAL_SIGNIFICATIVO
) -> Tuple[bool, str]:
    """
    Detecta si un ítem tiene problemas de ajuste al modelo IRT.
    
    Un p-valor bajo (< 0.05) indica que el ítem no se ajusta bien
    al modelo, lo que puede deberse a:
    - Violación de la unidimensionalidad
    - Dependencia local entre ítems
    - Respuestas no modeladas adecuadamente
    
    Args:
        p_valor: Valor p del estadístico de ajuste
        umbral: Umbral para considerar significativo (default: 0.05)
    
    Returns:
        Tuple con (tiene_problema, descripcion)
    
    Ejemplo:
        >>> detectar_problemas_ajuste(0.001)
        (True, 'Mal ajuste detectado: p-valor = 0.001 < 0.05')
        >>> detectar_problemas_ajuste(0.15)
        (False, 'Ajuste aceptable: p-valor = 0.15 >= 0.05')
    """
    if p_valor is None:
        return False, "No hay información de ajuste disponible"
    
    if p_valor < umbral:
        return True, f"Mal ajuste detectado: p-valor = {p_valor:.4f} < {umbral}"
    else:
        return False, f"Ajuste aceptable: p-valor = {p_valor:.4f} >= {umbral}"


def detectar_todos_problemas(
    param_a: float,
    param_b: float,
    p_valor: Optional[float] = None,
    param_c: Optional[float] = None,
    confiabilidad: Optional[float] = None
) -> List[TipoProblema]:
    """
    Detecta todos los problemas potenciales de un ítem.
    
    Args:
        param_a: Parámetro de discriminación
        param_b: Parámetro de dificultad
        p_valor: Valor p del estadístico de ajuste
        param_c: Parámetro de adivinanza (si aplica)
        confiabilidad: Coeficiente de confiabilidad test-retest
    
    Returns:
        Lista de tipos de problemas detectados
    """
    problemas = []
    
    # Detectar problemas de discriminación
    if param_a < 0:
        problemas.append(TipoProblema.DISCRIMINACION_NEGATIVA)
    elif param_a < UmbralesPsicometricos.DISCRIMINACION_BAJA_MAX:
        problemas.append(TipoProblema.DISCRIMINACION_BAJA)
    
    # Detectar problemas de dificultad extrema
    if abs(param_b) > 3.0:
        problemas.append(TipoProblema.DIFICULTAD_EXTREMA)
    
    # Detectar problemas de ajuste
    if p_valor is not None:
        if p_valor < UmbralesPsicometricos.P_VALOR_UMBRAL_SIGNIFICATIVO:
            problemas.append(TipoProblema.MAL_AJUSTE)
    
    # Detectar problemas de adivinanza
    if param_c is not None:
        if param_c > UmbralesPsicometricos.ADIVINANZA_ACEPTABLE_MAX:
            problemas.append(TipoProblema.ADIVINANZA_ALTA)
    
    # Detectar problemas de confiabilidad
    if confiabilidad is not None:
        if confiabilidad < UmbralesPsicometricos.CONFIABILIDAD_BAJA_MAX:
            problemas.append(TipoProblema.CONFIABILIDAD_BAJA)
    
    return problemas


# ==============================================================================
# FUNCIONES DE EVALUACIÓN DE CALIDAD
# ==============================================================================

def evaluar_calidad_item(
    param_a: float,
    param_b: float,
    ajuste_chi_cuadrado: Optional[float] = None,
    p_valor: Optional[float] = None,
    param_c: Optional[float] = None,
    confiabilidad_test_retest: Optional[float] = None
) -> CalidadItem:
    """
    Evalúa la calidad general de un ítem basándose en todos sus parámetros.
    
    La evaluación considera:
    - Discriminación (peso alto)
    - Dificultad (peso medio)
    - Ajuste del modelo (peso alto)
    - Confiabilidad (peso medio)
    
    Args:
        param_a: Parámetro de discriminación
        param_b: Parámetro de dificultad
        ajuste_chi_cuadrado: Valor del estadístico chi-cuadrado
        p_valor: Valor p del estadístico de ajuste
        param_c: Parámetro de adivinanza (si aplica)
        confiabilidad_test_retest: Coeficiente de confiabilidad
    
    Returns:
        CalidadItem: Clasificación de calidad general
    
    Ejemplo:
        >>> evaluar_calidad_item(1.5, 0.3, p_valor=0.25)
        <CalidadItem.EXCELENTE: 'excelente'>
    """
    # Puntuación base
    puntaje = 0
    puntaje_maximo = 0
    
    # Evaluar discriminación (peso: 3)
    puntaje_maximo += 3
    if param_a >= UmbralesPsicometricos.DISCRIMINACION_MUY_ALTA_MIN:
        puntaje += 3
    elif param_a >= UmbralesPsicometricos.DISCRIMINACION_ALTA_MIN:
        puntaje += 2.5
    elif param_a >= UmbralesPsicometricos.DISCRIMINACION_MODERADA_MIN:
        puntaje += 2
    elif param_a >= 0.5:
        puntaje += 1
    elif param_a > 0:
        puntaje += 0.5
    else:
        puntaje += 0  # Discriminación negativa
    
    # Evaluar dificultad (peso: 2) - preferimos dificultad moderada
    puntaje_maximo += 2
    if abs(param_b) <= 1.0:
        puntaje += 2  # Dificultad óptima
    elif abs(param_b) <= 2.0:
        puntaje += 1.5  # Dificultad aceptable
    elif abs(param_b) <= 3.0:
        puntaje += 1  # Dificultad extrema pero usable
    else:
        puntaje += 0.5  # Dificultad muy extrema
    
    # Evaluar ajuste (peso: 3)
    puntaje_maximo += 3
    if p_valor is not None:
        if p_valor >= 0.10:
            puntaje += 3  # Excelente ajuste
        elif p_valor >= 0.05:
            puntaje += 2.5  # Buen ajuste
        elif p_valor >= 0.01:
            puntaje += 1.5  # Ajuste cuestionable
        else:
            puntaje += 0.5  # Mal ajuste
    else:
        puntaje += 2  # Sin información, asumimos aceptable
    
    # Evaluar confiabilidad (peso: 2)
    if confiabilidad_test_retest is not None:
        puntaje_maximo += 2
        if confiabilidad_test_retest >= 0.85:
            puntaje += 2
        elif confiabilidad_test_retest >= 0.70:
            puntaje += 1.5
        elif confiabilidad_test_retest >= 0.60:
            puntaje += 1
        else:
            puntaje += 0.5
    
    # Calcular porcentaje
    porcentaje = (puntaje / puntaje_maximo) * 100 if puntaje_maximo > 0 else 0
    
    # Clasificar
    if porcentaje >= 85:
        return CalidadItem.EXCELENTE
    elif porcentaje >= 70:
        return CalidadItem.BUENA
    elif porcentaje >= 55:
        return CalidadItem.ACEPTABLE
    elif porcentaje >= 40:
        return CalidadItem.CUESTIONABLE
    else:
        return CalidadItem.DEFICIENTE


# ==============================================================================
# FUNCIONES DE GENERACIÓN DE RECOMENDACIONES
# ==============================================================================

def generar_recomendacion(item_data: Dict[str, Any]) -> List[str]:
    """
    Genera recomendaciones basadas en el análisis psicométrico de un ítem.
    
    Args:
        item_data: Diccionario con los datos del ítem y sus parámetros psicométricos.
                  Debe contener las claves: 'param_a', 'param_b', opcionalmente
                  'p_valor', 'param_c', 'confiabilidad_test_retest'
    
    Returns:
        Lista de recomendaciones específicas para mejorar el ítem
    
    Ejemplo:
        >>> item_data = {
        ...     'param_a': 0.4,
        ...     'param_b': 2.5,
        ...     'p_valor': 0.001
        ... }
        >>> generar_recomendacion(item_data)
        ['La discriminación es baja (0.4). Considerar revisar la redacción...', ...]
    """
    recomendaciones = []
    
    # Extraer parámetros
    param_a = item_data.get('param_a', 0)
    param_b = item_data.get('param_b', 0)
    p_valor = item_data.get('p_valor')
    param_c = item_data.get('param_c')
    confiabilidad = item_data.get('confiabilidad_test_retest')
    
    # Recomendaciones por discriminación
    if param_a < 0:
        recomendaciones.append(
            f"⚠️ DISCRIMINACIÓN NEGATIVA ({param_a:.2f}): El ítem presenta "
            f"discriminación negativa, lo cual es un problema grave. Posibles causas: "
            f"(1) La clave de respuesta está invertida, (2) El ítem mide un "
            f"constructo diferente, (3) Error en el procesamiento de datos. "
            f"RECOMENDACIÓN: Revisar la clave de respuesta y verificar que el "
            f"ítem esté alineado con la dimensión."
        )
    elif param_a < 0.5:
        recomendaciones.append(
            f"📉 DISCRIMINACIÓN MUY BAJA ({param_a:.2f}): El ítem tiene poca "
            f"capacidad para diferenciar entre estudiantes con diferentes niveles "
            f"del constructo. RECOMENDACIÓN: Considerar reformular el ítem para "
            f"que sea más específico del constructo medido, o eliminarlo si "
            f"persiste el problema."
        )
    elif param_a < UmbralesPsicometricos.DISCRIMINACION_MODERADA_MIN:
        recomendaciones.append(
            f"📊 DISCRIMINACIÓN BAJA ({param_a:.2f}): El ítem podría mejorar su "
            f"capacidad diferenciadora. RECOMENDACIÓN: Analizar si el lenguaje "
            f"es claro y si las opciones de respuesta son apropiadas."
        )
    elif param_a > 2.5:
        recomendaciones.append(
            f"⚡ DISCRIMINACIÓN MUY ALTA ({param_a:.2f}): Excelente capacidad "
            f"diferenciadora, pero verificar que no haya sobreajuste en la muestra."
        )
    
    # Recomendaciones por dificultad
    if param_b < -3.0:
        recomendaciones.append(
            f"🟢 DIFICULTAD MUY BAJA ({param_b:.2f}): El ítem es extremadamente "
            f"fácil. Casi todos los estudiantes responden correctamente. "
            f"RECOMENDACIÓN: Considerar si el ítem aporta información útil. "
            f"Podría eliminarse para acortar el cuestionario."
        )
    elif param_b > 3.0:
        recomendaciones.append(
            f"🔴 DIFICULTAD MUY ALTA ({param_b:.2f}): El ítem es extremadamente "
            f"difícil. Muy pocos estudiantes responden correctamente. "
            f"RECOMENDACIÓN: Revisar si el ítem está alineado con el nivel "
            f"educativo de la población objetivo. Considerar simplificar el "
            f"lenguaje o los conceptos evaluados."
        )
    elif abs(param_b) > 2.0:
        recomendaciones.append(
            f"🟡 DIFICULTAD EXTREMA ({param_b:.2f}): El ítem está en un extremo "
            f"de la distribución de dificultad. RECOMENDACIÓN: Asegurar que "
            f"haya suficientes ítems en toda la gama de dificultad para "
            f"cubrir todos los niveles de habilidad."
        )
    
    # Recomendaciones por ajuste
    if p_valor is not None:
        if p_valor < 0.01:
            recomendaciones.append(
                f"❌ MAL AJUSTE SIGNIFICATIVO (p = {p_valor:.4f}): El ítem no se "
                f"ajusta bien al modelo IRT. Posibles causas: (1) Violación de "
                f"unidimensionalidad, (2) Dependencia local con otros ítems, "
                f"(3) Efecto de respuesta diferencial (DIF). RECOMENDACIÓN: "
                f"Realizar análisis de DIF y revisar la estructura factorial."
            )
        elif p_valor < 0.05:
            recomendaciones.append(
                f"⚠️ AJUSTE CUESTIONABLE (p = {p_valor:.4f}): El ajuste es "
                f"marginalmente significativo. RECOMENDACIÓN: Monitorear este "
                f"ítem en futuras aplicaciones y considerar revisión."
            )
    
    # Recomendaciones por adivinanza
    if param_c is not None and param_c > UmbralesPsicometricos.ADIVINANZA_ACEPTABLE_MAX:
        recomendaciones.append(
            f"🎲 ADVINANZA ALTA (c = {param_c:.2f}): La probabilidad de acertar "
            f"por azar es alta. Esto reduce la información que proporciona el ítem. "
            f"RECOMENDACIÓN: Revisar el número de opciones de respuesta o "
            f"considerar usar un modelo 3PL si es apropiado."
        )
    
    # Recomendaciones por confiabilidad
    if confiabilidad is not None:
        if confiabilidad < 0.60:
            recomendaciones.append(
                f"📉 CONFIABILIDAD BAJA (r = {confiabilidad:.2f}): El ítem no es "
                f"consistente en el tiempo. RECOMENDACIÓN: Revisar la redacción "
                f"para eliminar ambigüedades o considerar eliminar el ítem."
            )
        elif confiabilidad < 0.70:
            recomendaciones.append(
                f"📊 CONFIABILIDAD MODERADA (r = {confiabilidad:.2f}): Aceptable "
                f"pero podría mejorar. RECOMENDACIÓN: Monitorear en futuras aplicaciones."
            )
    
    # Si no hay problemas, dar retroalimentación positiva
    if len(recomendaciones) == 0:
        recomendaciones.append(
            f"✅ ÍTEM DE BUENA CALIDAD: Discriminación = {param_a:.2f}, "
            f"Dificultad = {param_b:.2f}. El ítem cumple con los criterios "
            f"psicométricos recomendados."
        )
    
    return recomendaciones


def generar_recomendacion_mejora(
    item_actual: Dict[str, Any],
    items_referencia: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Genera recomendaciones específicas para mejorar un ítem existente.
    
    Args:
        item_actual: Datos del ítem a mejorar
        items_referencia: Lista de ítems de referencia con buenos parámetros
    
    Returns:
        Diccionario con recomendaciones estructuradas
    """
    recomendaciones = {
        "prioridad": "media",
        "acciones": [],
        "ejemplos": [],
        "consideraciones": []
    }
    
    param_a = item_actual.get('param_a', 0)
    param_b = item_actual.get('param_b', 0)
    
    # Determinar prioridad
    if param_a < 0 or (item_actual.get('p_valor') and item_actual.get('p_valor') < 0.01):
        recomendaciones["prioridad"] = "alta"
    elif param_a >= 0.8 and abs(param_b) <= 1.5:
        recomendaciones["prioridad"] = "baja"
    
    # Generar acciones específicas
    if param_a < 0.5:
        recomendaciones["acciones"].extend([
            "Revisar la alineación del ítem con la dimensión",
            "Simplificar el lenguaje para mayor claridad",
            "Asegurar que todas las opciones de respuesta sean plausibles",
            "Evitar el uso de negaciones o dobles negaciones"
        ])
    
    if abs(param_b) > 2.5:
        if param_b > 0:
            recomendaciones["acciones"].extend([
                "Simplificar los conceptos evaluados",
                "Usar vocabulario más accesible",
                "Dividir el ítem en preguntas más simples si es complejo"
            ])
        else:
            recomendaciones["acciones"].extend([
                "Aumentar la complejidad de la pregunta",
                "Incluir más contexto o información",
                "Hacer que las distractoras sean más atractivas"
            ])
    
    # Consideraciones generales
    recomendaciones["consideraciones"] = [
        "Mantener el mismo constructo al reformular",
        "Asegurar que la nueva versión sea comparable con la anterior",
        "Realizar prueba piloto antes de la aplicación masiva",
        "Documentar los cambios realizados"
    ]
    
    return recomendaciones


# ==============================================================================
# FUNCIONES DE ANÁLISIS COMPARATIVO
# ==============================================================================

def comparar_items(
    item1: Dict[str, Any],
    item2: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compara dos ítems y genera un análisis comparativo.
    
    Args:
        item1: Datos del primer ítem
        item2: Datos del segundo ítem
    
    Returns:
        Diccionario con el análisis comparativo
    """
    resultado = {
        "diferencias": {},
        "similitudes": [],
        "conclusion": ""
    }
    
    # Comparar discriminación
    diff_a = abs(item1.get('param_a', 0) - item2.get('param_a', 0))
    resultado["diferencias"]["discriminacion"] = {
        "valor": round(diff_a, 3),
        "interpretacion": "similar" if diff_a < 0.3 else "diferente"
    }
    
    # Comparar dificultad
    diff_b = abs(item1.get('param_b', 0) - item2.get('param_b', 0))
    resultado["diferencias"]["dificultad"] = {
        "valor": round(diff_b, 3),
        "interpretacion": "similar" if diff_b < 0.5 else "diferente"
    }
    
    # Identificar similitudes
    nivel1 = clasificar_dificultad(item1.get('param_b', 0))
    nivel2 = clasificar_dificultad(item2.get('param_b', 0))
    if nivel1 == nivel2:
        resultado["similitudes"].append(f"Mismo nivel de dificultad: {nivel1.value}")
    
    disc1 = clasificar_discriminacion(item1.get('param_a', 0))
    disc2 = clasificar_discriminacion(item2.get('param_a', 0))
    if disc1 == disc2:
        resultado["similitudes"].append(f"Mismo nivel de discriminación: {disc1.value}")
    
    # Generar conclusión
    if len(resultado["similitudes"]) >= 2:
        resultado["conclusion"] = "Los ítems son psicométricamente similares"
    elif len(resultado["similitudes"]) == 1:
        resultado["conclusion"] = "Los ítems tienen algunas similitudes psicométricas"
    else:
        resultado["conclusion"] = "Los ítems son psicométricamente diferentes"
    
    return resultado


# ==============================================================================
# FUNCIONES DE CÁLCULO IRT
# ==============================================================================

def calcular_probabilidad_2pl(
    theta: float,
    param_a: float,
    param_b: float
) -> float:
    """
    Calcula la probabilidad de respuesta correcta usando el modelo 2PL.
    
    Fórmula: P(θ) = 1 / (1 + exp(-a(θ - b)))
    
    Args:
        theta: Nivel de habilidad del estudiante (logits)
        param_a: Parámetro de discriminación
        param_b: Parámetro de dificultad
    
    Returns:
        Probabilidad de respuesta correcta (0-1)
    
    Ejemplo:
        >>> calcular_probabilidad_2pl(0, 1.0, 0)
        0.5
        >>> calcular_probabilidad_2pl(1.0, 1.0, 0)
        0.731...
    """
    exponente = -param_a * (theta - param_b)
    # Prevenir overflow
    if exponente > 709:  # ln(max_double) aprox
        return 0.0
    if exponente < -709:
        return 1.0
    return 1.0 / (1.0 + math.exp(exponente))


def calcular_probabilidad_3pl(
    theta: float,
    param_a: float,
    param_b: float,
    param_c: float
) -> float:
    """
    Calcula la probabilidad de respuesta correcta usando el modelo 3PL.
    
    Fórmula: P(θ) = c + (1 - c) / (1 + exp(-a(θ - b)))
    
    Args:
        theta: Nivel de habilidad del estudiante (logits)
        param_a: Parámetro de discriminación
        param_b: Parámetro de dificultad
        param_c: Parámetro de adivinanza
    
    Returns:
        Probabilidad de respuesta correcta (0-1)
    """
    prob_2pl = calcular_probabilidad_2pl(theta, param_a, param_b)
    return param_c + (1 - param_c) * prob_2pl


def calcular_informacion_item(
    theta: float,
    param_a: float,
    param_b: float,
    param_c: Optional[float] = None
) -> float:
    """
    Calcula la función de información del ítem (IIF).
    
    La información indica qué tan precisamente el ítem mide
    el nivel de habilidad θ.
    
    Args:
        theta: Nivel de habilidad
        param_a: Parámetro de discriminación
        param_b: Parámetro de dificultad
        param_c: Parámetro de adivinanza (None para modelo 2PL)
    
    Returns:
        Valor de información del ítem
    """
    if param_c is None:
        # Modelo 2PL
        p = calcular_probabilidad_2pl(theta, param_a, param_b)
        return (param_a ** 2) * p * (1 - p)
    else:
        # Modelo 3PL
        p = calcular_probabilidad_3pl(theta, param_a, param_b, param_c)
        q = 1 - p
        return (param_a ** 2) * ((p - param_c) ** 2) * q / ((1 - param_c) ** 2 * p)


# ==============================================================================
# EXPORTAR FUNCIONES PRINCIPALES
# ==============================================================================

__all__ = [
    # Constantes y umbrales
    'UmbralesPsicometricos',
    
    # Enumeraciones
    'NivelDificultad',
    'NivelDiscriminacion',
    'CalidadItem',
    'TipoProblema',
    
    # Estructuras de datos
    'ParametrosIRT',
    'IndicesAjuste',
    'ResultadoEvaluacion',
    
    # Funciones de validación
    'validar_parametros_irt',
    'validar_rango_parametro_a',
    'validar_rango_parametro_b',
    
    # Funciones de clasificación
    'clasificar_dificultad',
    'clasificar_discriminacion',
    'obtener_descripcion_dificultad',
    'obtener_descripcion_discriminacion',
    
    # Funciones de detección de problemas
    'detectar_problemas_ajuste',
    'detectar_todos_problemas',
    
    # Funciones de evaluación
    'evaluar_calidad_item',
    
    # Funciones de recomendaciones
    'generar_recomendacion',
    'generar_recomendacion_mejora',
    
    # Funciones comparativas
    'comparar_items',
    
    # Funciones de cálculo IRT
    'calcular_probabilidad_2pl',
    'calcular_probabilidad_3pl',
    'calcular_informacion_item',
]
