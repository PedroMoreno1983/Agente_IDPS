"""
================================================================================
UTILIDADES PSICOMÉTRICAS
================================================================================

Módulo de utilidades y funciones auxiliares para análisis psicométrico.
Proporciona cálculos estadísticos, transformaciones, validaciones y constantes
utilizadas en todo el sistema de psicometría.

Autor: Agente de IA Psicométrico
Versión: 1.0.0
================================================================================
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import statistics


# ==============================================================================
# CONSTANTES PSICOMÉTRICAS
# ==============================================================================

class ConstantesPsicometricas:
    """Constantes matemáticas y psicométricas utilizadas en el sistema."""
    
    # Constantes matemáticas
    E = math.e
    PI = math.pi
    LOGIT_MAX = 4.0
    LOGIT_MIN = -4.0
    
    # Constantes IRT
    DISCRIMINACION_MINIMA = 0.0
    DISCRIMINACION_MAXIMA = 3.0
    DIFICULTAD_MINIMA = -4.0
    DIFICULTAD_MAXIMA = 4.0
    ADVINANZA_MINIMA = 0.0
    ADVINANZA_MAXIMA = 0.5
    
    # Constantes estadísticas
    NIVEL_CONFIANZA_95 = 1.96
    NIVEL_CONFIANZA_99 = 2.576
    
    # Formatos de respuesta soportados
    FORMATOS_RESPUESTA = {
        "likert_4": {
            "nombre": "Escala Likert de 4 puntos",
            "opciones": ["Totalmente en desacuerdo", "En desacuerdo", "De acuerdo", "Totalmente de acuerdo"],
            "valores": [1, 2, 3, 4]
        },
        "likert_5": {
            "nombre": "Escala Likert de 5 puntos",
            "opciones": ["Totalmente en desacuerdo", "En desacuerdo", "Ni de acuerdo ni en desacuerdo", 
                        "De acuerdo", "Totalmente de acuerdo"],
            "valores": [1, 2, 3, 4, 5]
        },
        "dicotomico": {
            "nombre": "Formato dicotómico (Sí/No)",
            "opciones": ["No", "Sí"],
            "valores": [0, 1]
        },
        "frecuencia_4": {
            "nombre": "Escala de frecuencia de 4 puntos",
            "opciones": ["Nunca", "A veces", "Casi siempre", "Siempre"],
            "valores": [1, 2, 3, 4]
        },
        "frecuencia_5": {
            "nombre": "Escala de frecuencia de 5 puntos",
            "opciones": ["Nunca", "Rara vez", "A veces", "Casi siempre", "Siempre"],
            "valores": [1, 2, 3, 4, 5]
        }
    }
    
    # Dimensiones psicométricas comunes en educación
    DIMENSIONES_EDUCATIVAS = [
        "Autoestima Académica",
        "Autoeficacia",
        "Motivación Intrínseca",
        "Motivación Extrínseca",
        "Clima de Aula",
        "Convivencia Escolar",
        "Relación con Pares",
        "Apoyo Docente",
        "Compromiso Escolar",
        "Bienestar Emocional",
        "Regulación Emocional",
        "Hábitos de Estudio",
        "Gestión del Tiempo",
        "Pertenencia Institucional",
        "Expectativas Académicas"
    ]


# ==============================================================================
# FUNCIONES DE TRANSFORMACIÓN
# ==============================================================================

def z_a_logit(z_score: float, escala: float = 1.0) -> float:
    """
    Transforma un z-score a escala logit.
    
    Args:
        z_score: Puntuación z estándar
        escala: Factor de escala (default: 1.0)
    
    Returns:
        Valor en escala logit
    
    Ejemplo:
        >>> z_a_logit(0)
        0.0
        >>> z_a_logit(1.96)
        1.96
    """
    return z_score * escala


def logit_a_z(logit: float, escala: float = 1.0) -> float:
    """
    Transforma un valor logit a z-score.
    
    Args:
        logit: Valor en escala logit
        escala: Factor de escala (default: 1.0)
    
    Returns:
        Puntuación z
    """
    if escala == 0:
        raise ValueError("La escala no puede ser cero")
    return logit / escala


def p_a_logit(proporcion: float) -> float:
    """
    Transforma una proporción (0-1) a escala logit.
    
    Fórmula: logit(p) = ln(p / (1 - p))
    
    Args:
        proporcion: Proporción entre 0 y 1
    
    Returns:
        Valor en escala logit
    
    Ejemplo:
        >>> p_a_logit(0.5)
        0.0
        >>> p_a_logit(0.7)
        0.847...
    """
    if proporcion <= 0 or proporcion >= 1:
        raise ValueError("La proporción debe estar entre 0 y 1 (exclusivo)")
    return math.log(proporcion / (1 - proporcion))


def logit_a_p(logit: float) -> float:
    """
    Transforma un valor logit a proporción (0-1).
    
    Fórmula: p = 1 / (1 + exp(-logit))
    
    Args:
        logit: Valor en escala logit
    
    Returns:
        Proporción entre 0 y 1
    
    Ejemplo:
        >>> logit_a_p(0)
        0.5
        >>> logit_a_p(1.0)
        0.731...
    """
    return 1.0 / (1.0 + math.exp(-logit))


def normalizar_puntuacion(
    puntuacion: float,
    media_original: float,
    desv_est_original: float,
    media_nueva: float = 50,
    desv_est_nueva: float = 10
) -> float:
    """
    Normaliza una puntuación a una nueva escala.
    
    Args:
        puntuacion: Puntuación original
        media_original: Media de la escala original
        desv_est_original: Desviación estándar de la escala original
        media_nueva: Media de la nueva escala (default: 50)
        desv_est_nueva: Desviación estándar de la nueva escala (default: 10)
    
    Returns:
        Puntuación normalizada
    
    Ejemplo:
        >>> normalizar_puntuacion(75, 70, 10, 500, 100)
        550.0
    """
    if desv_est_original == 0:
        raise ValueError("La desviación estándar original no puede ser cero")
    z = (puntuacion - media_original) / desv_est_original
    return media_nueva + (z * desv_est_nueva)


def truncar_logit(logit: float, minimo: float = -4.0, maximo: float = 4.0) -> float:
    """
    Trunca un valor logit a un rango específico.
    
    Args:
        logit: Valor logit a truncar
        minimo: Valor mínimo permitido (default: -4.0)
        maximo: Valor máximo permitido (default: 4.0)
    
    Returns:
        Valor logit truncado
    """
    return max(minimo, min(maximo, logit))


# ==============================================================================
# FUNCIONES ESTADÍSTICAS BÁSICAS
# ==============================================================================

def calcular_media(datos: List[float]) -> float:
    """
    Calcula la media aritmética de una lista de valores.
    
    Args:
        datos: Lista de valores numéricos
    
    Returns:
        Media aritmética
    
    Ejemplo:
        >>> calcular_media([1, 2, 3, 4, 5])
        3.0
    """
    if not datos:
        raise ValueError("La lista de datos no puede estar vacía")
    return statistics.mean(datos)


def calcular_desviacion_estandar(datos: List[float], muestral: bool = True) -> float:
    """
    Calcula la desviación estándar de una lista de valores.
    
    Args:
        datos: Lista de valores numéricos
        muestral: Si True, calcula la desviación muestral (n-1)
    
    Returns:
        Desviación estándar
    
    Ejemplo:
        >>> calcular_desviacion_estandar([1, 2, 3, 4, 5])
        1.581...
    """
    if len(datos) < 2:
        raise ValueError("Se necesitan al menos 2 valores")
    if muestral:
        return statistics.stdev(datos)
    return statistics.pstdev(datos)


def calcular_correlacion_pearson(x: List[float], y: List[float]) -> float:
    """
    Calcula el coeficiente de correlación de Pearson.
    
    Args:
        x: Primera lista de valores
        y: Segunda lista de valores
    
    Returns:
        Coeficiente de correlación (-1 a 1)
    
    Ejemplo:
        >>> calcular_correlacion_pearson([1, 2, 3], [2, 4, 6])
        1.0
    """
    if len(x) != len(y):
        raise ValueError("Las listas deben tener el mismo tamaño")
    if len(x) < 2:
        raise ValueError("Se necesitan al menos 2 pares de valores")
    
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi ** 2 for xi in x)
    sum_y2 = sum(yi ** 2 for yi in y)
    
    numerador = n * sum_xy - sum_x * sum_y
    denominador = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
    
    if denominador == 0:
        return 0.0
    
    return numerador / denominador


def calcular_error_estandar(media: float, desv_est: float, n: int) -> float:
    """
    Calcula el error estándar de la media.
    
    Args:
        media: Media de la muestra
        desv_est: Desviación estándar
        n: Tamaño de la muestra
    
    Returns:
        Error estándar
    """
    if n <= 0:
        raise ValueError("El tamaño de muestra debe ser positivo")
    return desv_est / math.sqrt(n)


def calcular_intervalo_confianza(
    media: float,
    desv_est: float,
    n: int,
    nivel_confianza: float = 0.95
) -> Tuple[float, float]:
    """
    Calcula el intervalo de confianza para la media.
    
    Args:
        media: Media de la muestra
        desv_est: Desviación estándar
        n: Tamaño de la muestra
        nivel_confianza: Nivel de confianza (default: 0.95)
    
    Returns:
        Tupla con (límite_inferior, límite_superior)
    
    Ejemplo:
        >>> calcular_intervalo_confianza(50, 10, 100)
        (48.04, 51.96)
    """
    if nivel_confianza == 0.95:
        z = ConstantesPsicometricas.NIVEL_CONFIANZA_95
    elif nivel_confianza == 0.99:
        z = ConstantesPsicometricas.NIVEL_CONFIANZA_99
    else:
        # Aproximación para otros niveles
        z = 1.96  # Simplificación
    
    error_est = calcular_error_estandar(media, desv_est, n)
    margen = z * error_est
    
    return (media - margen, media + margen)


def calcular_coeficiente_alfa_cronbach(
    items: List[List[float]]
) -> float:
    """
    Calcula el coeficiente alfa de Cronbach para medir consistencia interna.
    
    Args:
        items: Lista de listas, donde cada lista interna contiene las
               respuestas a un ítem para todos los sujetos
    
    Returns:
        Coeficiente alfa de Cronbach (0-1)
    
    Ejemplo:
        >>> items = [[1, 2, 3], [2, 3, 4], [1, 2, 4]]
        >>> calcular_coeficiente_alfa_cronbach(items)
        0.9375
    """
    if not items or len(items) < 2:
        raise ValueError("Se necesitan al menos 2 ítems")
    
    n_items = len(items)
    n_sujetos = len(items[0])
    
    # Varianza de cada ítem
    varianzas_items = [statistics.variance(item) for item in items]
    suma_varianzas = sum(varianzas_items)
    
    # Varianza del puntaje total
    puntajes_totales = [sum(items[i][j] for i in range(n_items)) for j in range(n_sujetos)]
    varianza_total = statistics.variance(puntajes_totales)
    
    # Fórmula de Cronbach
    alfa = (n_items / (n_items - 1)) * (1 - suma_varianzas / varianza_total)
    
    return max(0, min(1, alfa))  # Asegurar rango 0-1


def calcular_estadisticas_descriptivas(datos: List[float]) -> Dict[str, float]:
    """
    Calcula estadísticas descriptivas completas.
    
    Args:
        datos: Lista de valores numéricos
    
    Returns:
        Diccionario con estadísticas descriptivas
    
    Ejemplo:
        >>> calcular_estadisticas_descriptivas([1, 2, 3, 4, 5])
        {'media': 3.0, 'mediana': 3, 'desv_est': 1.58, ...}
    """
    if not datos:
        raise ValueError("La lista de datos no puede estar vacía")
    
    n = len(datos)
    datos_ordenados = sorted(datos)
    
    # Media
    media = statistics.mean(datos)
    
    # Mediana
    mediana = statistics.median(datos)
    
    # Moda (puede haber múltiples)
    try:
        moda = statistics.mode(datos)
    except statistics.StatisticsError:
        moda = None
    
    # Desviación estándar
    desv_est = statistics.stdev(datos) if n > 1 else 0
    
    # Varianza
    varianza = statistics.variance(datos) if n > 1 else 0
    
    # Mínimo y máximo
    minimo = min(datos)
    maximo = max(datos)
    
    # Rango
    rango = maximo - minimo
    
    # Percentiles
    p25 = datos_ordenados[int(n * 0.25)] if n > 1 else datos[0]
    p75 = datos_ordenados[int(n * 0.75)] if n > 1 else datos[0]
    
    # Asimetría (skewness) - aproximación
    if n > 2 and desv_est > 0:
        asimetria = sum((x - media) ** 3 for x in datos) / (n * desv_est ** 3)
    else:
        asimetria = 0
    
    # Curtosis - aproximación
    if n > 3 and desv_est > 0:
        curtosis = sum((x - media) ** 4 for x in datos) / (n * desv_est ** 4) - 3
    else:
        curtosis = 0
    
    return {
        "n": n,
        "media": round(media, 4),
        "mediana": round(mediana, 4),
        "moda": moda,
        "desv_est": round(desv_est, 4),
        "varianza": round(varianza, 4),
        "minimo": round(minimo, 4),
        "maximo": round(maximo, 4),
        "rango": round(rango, 4),
        "p25": round(p25, 4),
        "p75": round(p75, 4),
        "iqr": round(p75 - p25, 4),
        "asimetria": round(asimetria, 4),
        "curtosis": round(curtosis, 4)
    }


# ==============================================================================
# FUNCIONES DE VALIDACIÓN DE ÍTEMS
# ==============================================================================

def validar_texto_item(texto: str, min_caracteres: int = 10, max_caracteres: int = 500) -> Tuple[bool, List[str]]:
    """
    Valida el texto de un ítem psicométrico.
    
    Args:
        texto: Texto del ítem a validar
        min_caracteres: Longitud mínima permitida
        max_caracteres: Longitud máxima permitida
    
    Returns:
        Tupla con (es_valido, lista_de_errores)
    """
    errores = []
    
    if not texto or not texto.strip():
        errores.append("El texto del ítem no puede estar vacío")
        return False, errores
    
    texto_limpio = texto.strip()
    
    # Validar longitud
    if len(texto_limpio) < min_caracteres:
        errores.append(f"El texto es demasiado corto ({len(texto_limpio)} caracteres). "
                      f"Mínimo requerido: {min_caracteres}")
    
    if len(texto_limpio) > max_caracteres:
        errores.append(f"El texto es demasiado largo ({len(texto_limpio)} caracteres). "
                      f"Máximo permitido: {max_caracteres}")
    
    # Validar que termine con signo de interrogación si es pregunta
    if "?" in texto_limpio and not texto_limpio.endswith("?"):
        errores.append("El signo de interrogación debe estar al final del ítem")
    
    # Validar dobles espacios
    if "  " in texto_limpio:
        errores.append("El texto contiene dobles espacios")
    
    # Validar mayúsculas iniciales
    if texto_limpio and not texto_limpio[0].isupper():
        errores.append("El texto debe comenzar con mayúscula")
    
    # Validar negaciones
    negaciones = ["no", "nunca", "jamás", "tampoco", "ni"]
    palabras = texto_limpio.lower().split()
    num_negaciones = sum(1 for p in palabras if p.strip(".,;:!?") in negaciones)
    
    if num_negaciones > 1:
        errores.append(f"El ítem contiene múltiples negaciones ({num_negaciones}). "
                      f"Esto puede causar confusiones. Considere reformular.")
    
    return len(errores) == 0, errores


def validar_opciones_respuesta(
    opciones: List[str],
    formato: str = "likert_4"
) -> Tuple[bool, List[str]]:
    """
    Valida las opciones de respuesta de un ítem.
    
    Args:
        opciones: Lista de textos de opciones
        formato: Formato de respuesta esperado
    
    Returns:
        Tupla con (es_valido, lista_de_errores)
    """
    errores = []
    
    if not opciones:
        errores.append("Debe proporcionar al menos una opción de respuesta")
        return False, errores
    
    # Validar número de opciones según formato
    formatos_esperados = {
        "likert_4": 4,
        "likert_5": 5,
        "dicotomico": 2,
        "frecuencia_4": 4,
        "frecuencia_5": 5
    }
    
    num_esperado = formatos_esperados.get(formato)
    if num_esperado and len(opciones) != num_esperado:
        errores.append(f"El formato '{formato}' requiere {num_esperado} opciones, "
                      f"pero se proporcionaron {len(opciones)}")
    
    # Validar que no haya opciones vacías
    for i, opcion in enumerate(opciones):
        if not opcion or not opcion.strip():
            errores.append(f"La opción {i+1} está vacía")
    
    # Validar duplicados
    opciones_limpias = [op.strip().lower() for op in opciones if op]
    if len(opciones_limpias) != len(set(opciones_limpias)):
        errores.append("Hay opciones de respuesta duplicadas")
    
    # Validar longitud de opciones
    for i, opcion in enumerate(opciones):
        if opcion and len(opcion) > 100:
            errores.append(f"La opción {i+1} es demasiado larga ({len(opcion)} caracteres)")
    
    return len(errores) == 0, errores


def validar_dimension(dimension: str) -> Tuple[bool, str]:
    """
    Valida que una dimensión psicométrica sea reconocida.
    
    Args:
        dimension: Nombre de la dimensión
    
    Returns:
        Tupla con (es_valida, mensaje)
    """
    dimensiones = [d.lower() for d in ConstantesPsicometricas.DIMENSIONES_EDUCATIVAS]
    
    if not dimension or not dimension.strip():
        return False, "La dimensión no puede estar vacía"
    
    if dimension.lower() in dimensiones:
        return True, f"Dimensión '{dimension}' reconocida"
    
    # Buscar similitudes
    sugerencias = [d for d in ConstantesPsicometricas.DIMENSIONES_EDUCATIVAS 
                   if dimension.lower() in d.lower() or d.lower() in dimension.lower()]
    
    if sugerencias:
        return False, f"Dimensión no reconocida. ¿Quizás quiso decir: {', '.join(sugerencias)}?"
    
    return False, f"Dimensión no reconocida. Dimensiones válidas: {', '.join(ConstantesPsicometricas.DIMENSIONES_EDUCATIVAS[:5])}..."


def detectar_lenguaje_inclusivo(texto: str) -> Tuple[bool, List[str]]:
    """
    Detecta si el texto usa lenguaje inclusivo apropiado.
    
    Args:
        texto: Texto a analizar
    
    Returns:
        Tupla con (es_inclusivo, sugerencias)
    """
    sugerencias = []
    texto_lower = texto.lower()
    
    # Palabras a evitar
    terminos_no_inclusivos = {
        r"\b(el|los)\s+alumnos?\b": "Considere usar 'estudiantes' en lugar de 'alumnos'",
        r"\b(el|los)\s+profesores?\b": "Considere usar 'docentes' en lugar de 'profesores'",
        r"\bdiscapacitado\b": "Use 'persona con discapacidad'",
        r"\binvalido\b": "Use 'persona con discapacidad'",
        r"\bnormal\b": "Evite el término 'normal' para referirse a personas",
        r"\b(anormales?|anormal)\b": "Evite términos peyorativos",
    }
    
    for patron, sugerencia in terminos_no_inclusivos.items():
        if re.search(patron, texto_lower):
            sugerencias.append(sugerencia)
    
    return len(sugerencias) == 0, sugerencias


# ==============================================================================
# FUNCIONES DE ANÁLISIS DE TEXTO
# ==============================================================================

def contar_palabras(texto: str) -> int:
    """Cuenta el número de palabras en un texto."""
    if not texto:
        return 0
    return len(texto.split())


def contar_oraciones(texto: str) -> int:
    """Cuenta el número de oraciones en un texto."""
    if not texto:
        return 0
    oraciones = re.split(r'[.!?]+', texto)
    return len([o for o in oraciones if o.strip()])


def calcular_legibilidad(texto: str) -> Dict[str, float]:
    """
    Calcula métricas de legibilidad del texto.
    
    Args:
        texto: Texto a analizar
    
    Returns:
        Diccionario con métricas de legibilidad
    """
    if not texto:
        return {"palabras": 0, "oraciones": 0, "promedio_palabras_oracion": 0}
    
    palabras = texto.split()
    num_palabras = len(palabras)
    num_oraciones = contar_oraciones(texto)
    
    promedio = num_palabras / num_oraciones if num_oraciones > 0 else 0
    
    return {
        "palabras": num_palabras,
        "oraciones": num_oraciones,
        "promedio_palabras_oracion": round(promedio, 2)
    }


def extraer_palabras_clave(texto: str, n: int = 5) -> List[str]:
    """
    Extrae las palabras clave más relevantes de un texto.
    
    Args:
        texto: Texto a analizar
        n: Número de palabras clave a extraer
    
    Returns:
        Lista de palabras clave
    """
    if not texto:
        return []
    
    # Palabras vacías en español
    stopwords = {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
        "al", "y", "o", "pero", "por", "para", "con", "sin", "sobre", "entre",
        "a", "ante", "bajo", "desde", "hasta", "hacia", "en", "durante", "mediante",
        "según", "sin", "so", "tras", "que", "como", "cuando", "donde", "quien",
        "cuyo", "cuya", "este", "esta", "estos", "estas", "ese", "esa", "esos",
        "esas", "aquel", "aquella", "aquellos", "aquellas", "mi", "tu", "su",
        "nuestro", "vuestro", "suyo", "me", "te", "se", "nos", "os", "lo", "le",
        "les", "yo", "tú", "él", "ella", "nosotros", "nosotras", "vosotros",
        "vosotras", "ellos", "ellas", "es", "son", "está", "están", "fue", "fueron",
        "ser", "estar", "tener", "hay", "muy", "más", "menos", "también", "ya",
        "sí", "no", "si", "sino"
    }
    
    # Limpiar y tokenizar
    palabras = re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑ]+\b', texto.lower())
    
    # Filtrar stopwords y contar frecuencias
    frecuencias = {}
    for palabra in palabras:
        if palabra not in stopwords and len(palabra) > 2:
            frecuencias[palabra] = frecuencias.get(palabra, 0) + 1
    
    # Ordenar por frecuencia
    palabras_ordenadas = sorted(frecuencias.items(), key=lambda x: x[1], reverse=True)
    
    return [palabra for palabra, _ in palabras_ordenadas[:n]]


# ==============================================================================
# FUNCIONES DE FORMATO Y SERIALIZACIÓN
# ==============================================================================

def formatear_parametro_psicometrico(
    valor: float,
    tipo: str = "general",
    decimales: int = 3
) -> str:
    """
    Formatea un parámetro psicométrico para visualización.
    
    Args:
        valor: Valor numérico a formatear
        tipo: Tipo de parámetro ("discriminacion", "dificultad", "p_valor")
        decimales: Número de decimales
    
    Returns:
        Cadena formateada
    """
    if valor is None:
        return "N/A"
    
    if tipo == "p_valor":
        if valor < 0.001:
            return "< 0.001"
        return f"{valor:.{decimales}f}"
    
    return f"{valor:.{decimales}f}"


def generar_codigo_item(
    dimension: str,
    año: int,
    secuencial: int
) -> str:
    """
    Genera un código único para un ítem.
    
    Args:
        dimension: Nombre de la dimensión
        año: Año de creación
        secuencial: Número secuencial
    
    Returns:
        Código del ítem
    
    Ejemplo:
        >>> generar_codigo_item("Autoestima", 2024, 15)
        "AUT-2024-015"
    """
    # Abreviar dimensión (primeras 3 letras mayúsculas)
    abrev = dimension[:3].upper() if dimension else "GEN"
    
    # Formatear secuencial con ceros
    sec_formateado = str(secuencial).zfill(3)
    
    return f"{abrev}-{año}-{sec_formateado}"


def parsear_codigo_item(codigo: str) -> Dict[str, Any]:
    """
    Parsea un código de ítem en sus componentes.
    
    Args:
        codigo: Código del ítem
    
    Returns:
        Diccionario con los componentes
    """
    partes = codigo.split("-")
    
    if len(partes) != 3:
        return {"error": "Formato de código inválido"}
    
    return {
        "abreviatura": partes[0],
        "año": int(partes[1]) if partes[1].isdigit() else None,
        "secuencial": int(partes[2]) if partes[2].isdigit() else None
    }


def serializar_resultado_psicometrico(
    resultado: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Serializa un resultado psicométrico para almacenamiento o transmisión.
    
    Args:
        resultado: Diccionario con el resultado
    
    Returns:
        Diccionario serializado
    """
    serializado = {}
    
    for clave, valor in resultado.items():
        if isinstance(valor, float):
            # Redondear floats
            serializado[clave] = round(valor, 6)
        elif isinstance(valor, datetime):
            # Convertir datetime a ISO
            serializado[clave] = valor.isoformat()
        elif isinstance(valor, dict):
            # Recursión para diccionarios anidados
            serializado[clave] = serializar_resultado_psicometrico(valor)
        elif isinstance(valor, list):
            # Procesar listas
            serializado[clave] = [
                round(v, 6) if isinstance(v, float) else v 
                for v in valor
            ]
        else:
            serializado[clave] = valor
    
    return serializado


# ==============================================================================
# FUNCIONES DE CONVERSIÓN DE MODELOS
# ==============================================================================

def convertir_2pl_a_1pl(param_a: float, param_b: float) -> float:
    """
    Convierte parámetros 2PL a escala 1PL (Rasch).
    
    En el modelo 1PL, la discriminación es constante (generalmente 1),
    por lo que la dificultad se ajusta.
    
    Args:
        param_a: Discriminación en modelo 2PL
        param_b: Dificultad en modelo 2PL
    
    Returns:
        Dificultad ajustada para modelo 1PL
    """
    # Ajuste aproximado
    return param_a * param_b


def estandarizar_parametros(
    param_a: float,
    param_b: float,
    media_habilidad: float = 0,
    desv_est_habilidad: float = 1
) -> Tuple[float, float]:
    """
    Estandariza parámetros a una escala común.
    
    Args:
        param_a: Discriminación original
        param_b: Dificultad original
        media_habilidad: Media de la escala de habilidad
        desv_est_habilidad: Desviación estándar de la escala de habilidad
    
    Returns:
        Tupla con (param_a_estandarizado, param_b_estandarizado)
    """
    if desv_est_habilidad == 0:
        raise ValueError("La desviación estándar no puede ser cero")
    
    param_b_est = (param_b - media_habilidad) / desv_est_habilidad
    param_a_est = param_a * desv_est_habilidad
    
    return param_a_est, param_b_est


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def es_valor_valido(valor: Any) -> bool:
    """Verifica si un valor es válido (no None, no NaN, no infinito)."""
    if valor is None:
        return False
    if isinstance(valor, float):
        if math.isnan(valor) or math.isinf(valor):
            return False
    return True


def limpiar_valor_psicometrico(
    valor: Optional[float],
    valor_default: float = 0.0
) -> float:
    """
    Limpia un valor psicométrico, reemplazando valores inválidos.
    
    Args:
        valor: Valor a limpiar
        valor_default: Valor por defecto
    
    Returns:
        Valor limpio
    """
    if not es_valor_valido(valor):
        return valor_default
    return valor


def calcular_severidad_problema(
    tipo_problema: str,
    magnitud: float
) -> str:
    """
    Calcula la severidad de un problema psicométrico.
    
    Args:
        tipo_problema: Tipo de problema
        magnitud: Magnitud del problema
    
    Returns:
        Nivel de severidad ("baja", "media", "alta", "crítica")
    """
    umbrales = {
        "discriminacion_negativa": (0, "crítica"),
        "discriminacion_baja": (0.5, "alta", 0.3, "media"),
        "dificultad_extrema": (3.5, "alta", 2.5, "media"),
        "mal_ajuste": (0.001, "alta", 0.05, "media"),
        "adivinanza_alta": (0.4, "alta", 0.3, "media"),
    }
    
    config = umbrales.get(tipo_problema)
    if not config:
        return "media"
    
    if tipo_problema == "discriminacion_negativa":
        return "crítica" if magnitud < 0 else "baja"
    
    # Evaluar según umbrales
    for i in range(0, len(config), 2):
        if magnitud >= config[i]:
            return config[i + 1]
    
    return "baja"


# ==============================================================================
# EXPORTAR FUNCIONES
# ==============================================================================

__all__ = [
    # Constantes
    'ConstantesPsicometricas',
    
    # Transformaciones
    'z_a_logit',
    'logit_a_z',
    'p_a_logit',
    'logit_a_p',
    'normalizar_puntuacion',
    'truncar_logit',
    
    # Estadísticas
    'calcular_media',
    'calcular_desviacion_estandar',
    'calcular_correlacion_pearson',
    'calcular_error_estandar',
    'calcular_intervalo_confianza',
    'calcular_coeficiente_alfa_cronbach',
    'calcular_estadisticas_descriptivas',
    
    # Validación de ítems
    'validar_texto_item',
    'validar_opciones_respuesta',
    'validar_dimension',
    'detectar_lenguaje_inclusivo',
    
    # Análisis de texto
    'contar_palabras',
    'contar_oraciones',
    'calcular_legibilidad',
    'extraer_palabras_clave',
    
    # Formato y serialización
    'formatear_parametro_psicometrico',
    'generar_codigo_item',
    'parsear_codigo_item',
    'serializar_resultado_psicometrico',
    
    # Conversión de modelos
    'convertir_2pl_a_1pl',
    'estandarizar_parametros',
    
    # Auxiliares
    'es_valor_valido',
    'limpiar_valor_psicometrico',
    'calcular_severidad_problema',
]
