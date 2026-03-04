"""
================================================================================
PROMPTS ESPECIALIZADOS PARA LLM
================================================================================

Módulo que contiene templates de prompts especializados para interacciones
con Modelos de Lenguaje Grandes (LLM) en el contexto de psicometría educativa.

Los prompts están diseñados para:
- Generación de ítems psicométricos de alta calidad
- Análisis y evaluación de ítems existentes
- Explicación de parámetros psicométricos
- Sugerencias de mejora basadas en evidencia

Autor: Agente de IA Psicométrico
Versión: 1.0.0
================================================================================
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from string import Template


# ==============================================================================
# ESTRUCTURAS DE DATOS PARA PROMPTS
# ==============================================================================

@dataclass
class ContextoGeneracion:
    """Contexto para la generación de ítems."""
    dimension: str
    descripcion_dimension: str
    nivel_educativo: str
    poblacion_objetivo: str
    formato_respuesta: str
    dificultad_objetivo: Optional[str] = None
    items_referencia: Optional[List[Dict]] = None
    restricciones: Optional[List[str]] = None


@dataclass
class ContextoAnalisis:
    """Contexto para el análisis de ítems."""
    texto_item: str
    opciones_respuesta: List[str]
    parametros_irt: Dict[str, float]
    indices_ajuste: Optional[Dict[str, float]] = None
    estadisticas_respuesta: Optional[Dict[str, Any]] = None


# ==============================================================================
# PROMPTS PARA GENERACIÓN DE ÍTEMS
# ==============================================================================

PROMPT_GENERACION_ITEM = Template("""
Eres un experto psicometrista especializado en la creación de ítems para evaluaciones 
educativas de alta calidad. Tu tarea es generar un nuevo ítem que cumpla con estándares 
psicométricos rigurosos.

## CONTEXTO DE LA DIMENSIÓN
- **Dimensión**: ${dimension}
- **Descripción**: ${descripcion_dimension}
- **Nivel educativo**: ${nivel_educativo}
- **Población objetivo**: ${poblacion_objetivo}
- **Formato de respuesta**: ${formato_respuesta}
${dificultad_linea}

## ÍTEMS DE REFERENCIA (Ejemplos de alta calidad)
${items_referencia_texto}

## DIRECTRICES PARA LA GENERACIÓN

### 1. Alineación con el Constructo
- El ítem debe medir EXACTAMENTE la dimensión indicada
- Evitar ítems que midan constructos relacionados pero diferentes
- Asegurar que el contenido sea relevante para el nivel educativo

### 2. Claridad y Comprensibilidad
- Usar lenguaje claro y directo
- Evitar dobles negaciones
- Evite jerga técnica innecesaria
- Una sola idea por ítem
- Longitud apropiada (15-50 palabras)

### 3. Lenguaje Inclusivo
- Usar lenguaje no sexista
- Evitar estereotipos de género, etnia o condición socioeconómica
- Usar "estudiantes" en lugar de "alumnos"
- Considerar la diversidad cultural

### 4. Opciones de Respuesta
- Todas las opciones deben ser plausibles
- Evitar opciones obviamente incorrectas
- Mantener consistencia en la dirección de la escala
- El número de opciones debe coincidir con el formato indicado

### 5. Propiedades Psicométricas Deseadas
- Discriminación esperada: 0.8 - 1.5
- Dificultad objetivo: ${dificultad_objetivo_valor}
- Evitar extremos de dificultad (< -2.5 o > 2.5)

${restricciones_texto}

## FORMATO DE RESPUESTA
Genera el ítem en el siguiente formato JSON:

```json
{
    "texto_item": "Texto completo del ítem",
    "opciones_respuesta": ["Opción 1", "Opción 2", "Opción 3", "Opción 4"],
    "valores_opciones": [1, 2, 3, 4],
    "direccion": "positiva|negativa",
    "justificacion": "Explicación de por qué este ítem mide la dimensión",
    "nivel_dificultad_estimado": "muy_facil|facil|moderada|dificil|muy_dificil",
    "consideraciones": ["Lista de consideraciones especiales"]
}
```

Genera el ítem ahora:
""")


PROMPT_GENERACION_ITEM_SIN_REFERENCIA = Template("""
Eres un experto psicometrista especializado en evaluaciones educativas.
Genera un ítem de alta calidad para la siguiente dimensión:

**Dimensión**: ${dimension}
**Descripción**: ${descripcion_dimension}
**Nivel educativo**: ${nivel_educativo}
**Formato**: ${formato_respuesta}

El ítem debe:
1. Medir directamente el constructo indicado
2. Usar lenguaje claro y apropiado para el nivel educativo
3. Ser culturalmente sensible e inclusivo
4. Tener opciones de respuesta balanceadas

Responde en formato JSON con: texto_item, opciones_respuesta, direccion, justificacion
""")


# ==============================================================================
# PROMPTS PARA ANÁLISIS DE ÍTEMS
# ==============================================================================

PROMPT_ANALISIS_CALIDAD_ITEM = Template("""
Eres un experto psicometrista. Realiza un análisis completo del siguiente ítem:

## ÍTEM A ANALIZAR
**Texto**: ${texto_item}

**Opciones de respuesta**:
${opciones_texto}

## PARÁMETROS PSICOMÉTRICOS
- Discriminación (a): ${param_a}
- Dificultad (b): ${param_b}
${param_c_linea}
- P-valor de ajuste: ${p_valor}
${chi_cuadrado_linea}
${confiabilidad_linea}

## ANÁLISIS REQUERIDO

### 1. Evaluación de la Redacción
- Claridad del enunciado
- Ausencia de ambigüedades
- Adecuación al nivel educativo
- Uso de lenguaje inclusivo

### 2. Evaluación de Opciones de Respuesta
- Plausibilidad de todas las opciones
- Consistencia en la dirección
- Ausencia de claves visuales

### 3. Interpretación de Parámetros IRT
${interpretacion_parametros}

### 4. Identificación de Problemas
- Problemas psicométricos detectados
- Posibles causas
- Severidad de cada problema

### 5. Recomendaciones de Mejora
- Acciones específicas para mejorar el ítem
- Priorización de cambios
- Consideraciones para la reformulación

## FORMATO DE RESPUESTA
```json
{
    "evaluacion_redaccion": {
        "claridad": "excelente|buena|regular|deficiente",
        "comentario": "..."
    },
    "evaluacion_opciones": {
        "calidad": "excelente|buena|regular|deficiente",
        "comentario": "..."
    },
    "interpretacion_psicometrica": "...",
    "problemas_detectados": [
        {
            "tipo": "...",
            "descripcion": "...",
            "severidad": "alta|media|baja"
        }
    ],
    "recomendaciones": ["..."],
    "calidad_general": "excelente|buena|aceptable|cuestionable|deficiente",
    "accion_recomendada": "mantener|revisar|reformular|eliminar"
}
```
""")


PROMPT_ANALISIS_COMPARATIVO = Template("""
Compara los siguientes ítems psicométricamente:

## ÍTEM 1
${item1_texto}
Parámetros: a=${item1_a}, b=${item1_b}

## ÍTEM 2
${item2_texto}
Parámetros: a=${item2_a}, b=${item2_b}

Analiza:
1. Similitudes y diferencias en contenido
2. Similitudes y diferencias psicométricas
3. Cuál ítem es más efectivo y por qué
4. Recomendaciones para seleccionar entre ellos

Responde en formato JSON.
""")


# ==============================================================================
# PROMPTS PARA EXPLICACIÓN PSICOMÉTRICA
# ==============================================================================

PROMPT_EXPLICACION_PARAMETROS = Template("""
Eres un psicometrista experto. Explica los siguientes parámetros de manera clara 
para un usuario que no es experto en psicometría:

## PARÁMETROS DEL ÍTEM
- **Discriminación (a)**: ${param_a}
- **Dificultad (b)**: ${param_b}
${param_c_linea}
- **P-valor de ajuste**: ${p_valor}

## INSTRUCCIONES
Proporciona una explicación en lenguaje sencillo que incluya:

1. **¿Qué significa cada parámetro?**
   - Explicación conceptual sin fórmulas complejas
   - Analogías si son útiles

2. **¿Cómo interpretar estos valores?**
   - ¿Son buenos o malos valores?
   - Comparación con rangos típicos

3. **¿Qué implicaciones tiene para el ítem?**
   - ¿El ítem funciona bien?
   - ¿Para qué tipo de estudiantes es más informativo?

4. **Analogía o ejemplo práctico**
   - Usar una analogía del mundo real

## FORMATO DE RESPUESTA
Estructura tu respuesta en párrafos claros, evitando jerga técnica innecesaria.
""")


PROMPT_EXPLICACION_MODELO_IRT = Template("""
Explica el modelo ${modelo_irt} (Teoría de Respuesta al Ítem) de manera accesible:

## CONTEXTO
- Modelo: ${modelo_irt}
- Audiencia: Educadores sin formación psicométrica
- Objetivo: Entender cómo funciona el modelo

## CONTENIDO REQUERIDO
1. Concepto básico del modelo
2. Parámetros que utiliza y qué representan
3. Ventajas sobre la teoría clásica de tests
4. Cuándo es apropiado usarlo
5. Limitaciones a considerar

Usa analogías del mundo real para facilitar la comprensión.
""")


# ==============================================================================
# PROMPTS PARA SUGERENCIAS DE MEJORA
# ==============================================================================

PROMPT_SUGERIR_MEJORAS = Template("""
Eres un consultor psicométrico experto. Proporciona sugerencias específicas 
para mejorar el siguiente ítem:

## ÍTEM ACTUAL
**Texto**: ${texto_item}

**Opciones**: ${opciones_texto}

## PROBLEMAS IDENTIFICADOS
${problemas_texto}

## CONTEXTO ADICIONAL
- Dimensión: ${dimension}
- Nivel educativo: ${nivel_educativo}
- Objetivo de dificultad: ${dificultad_objetivo}

## INSTRUCCIONES
Proporciona sugerencias específicas y accionables:

1. **Cambios en la redacción**
   - Frases específicas a modificar
   - Alternativas propuestas
   - Justificación de cada cambio

2. **Cambios en opciones de respuesta**
   - Opciones problemáticas
   - Sugerencias de mejora
   - Cómo mantener el balance

3. **Versiones alternativas**
   - Proporciona 2-3 versiones reformuladas
   - Explica las diferencias entre versiones

4. **Consideraciones de implementación**
   - Cómo probar la nueva versión
   - Qué monitorear en la próxima aplicación

## FORMATO DE RESPUESTA
```json
{
    "cambios_redaccion": [
        {
            "original": "...",
            "sugerencia": "...",
            "justificacion": "..."
        }
    ],
    "cambios_opciones": [...],
    "versiones_alternativas": [
        {
            "texto": "...",
            "opciones": [...],
            "diferencias_clave": "..."
        }
    ],
    "consideraciones_implementacion": [...]
}
```
""")


PROMPT_VALIDACION_NUEVO_ITEM = Template("""
Valida el siguiente ítem propuesto según criterios psicométricos:

## ÍTEM PROPUESTO
**Texto**: ${texto_item}

**Dimensión**: ${dimension}

**Descripción de la dimensión**: ${descripcion_dimension}

## CRITERIOS DE VALIDACIÓN

### 1. Alineación con el Constructo
- ¿El ítem mide realmente la dimensión indicada?
- ¿Hay elementos que midan constructos diferentes?

### 2. Propiedades Lingüísticas
- Claridad y comprensibilidad
- Ausencia de ambigüedades
- Nivel de vocabulario apropiado
- Uso de lenguaje inclusivo

### 3. Estructura del Ítem
- Una sola idea central
- Sin dobles negaciones
- Sin preguntas capciosas

### 4. Potencial Psicométrico
- ¿Es probable que discrimine bien?
- ¿La dificultad parece apropiada?

## FORMATO DE RESPUESTA
```json
{
    "es_valido": true|false,
    "puntuacion": 0-100,
    "criterios": {
        "alineacion": {"puntos": 0-25, "comentario": "..."},
        "lenguaje": {"puntos": 0-25, "comentario": "..."},
        "estructura": {"puntos": 0-25, "comentario": "..."},
        "potencial": {"puntos": 0-25, "comentario": "..."}
    },
    "fortalezas": [...],
    "debilidades": [...],
    "sugerencias": [...],
    "recomendacion_final": "aceptar|revisar|rechazar"
}
```
""")


# ==============================================================================
# PROMPTS PARA BÚSQUEDA Y RECUPERACIÓN
# ==============================================================================

PROMPT_BUSQUEDA_SIMILITUD = Template("""
Analiza la similitud semántica entre el ítem de consulta y los ítems candidatos.

## ÍTEM DE CONSULTA
${texto_consulta}

## ÍTEMS CANDIDATOS
${items_candidatos_texto}

## INSTRUCCIONES
Para cada ítem candidato, evalúa:
1. Similitud temática (¿miden lo mismo?)
2. Similitud en redacción
3. Similitud en nivel de complejidad
4. Posibilidad de redundancia

Clasifica cada ítem como: "muy_similar", "similar", "relacionado", "diferente"

Responde en formato JSON con el análisis de cada candidato.
""")


PROMPT_GENERACION_CONSULTA = Template("""
Genera una consulta optimizada para buscar ítems similares basándote en:

**Texto del ítem**: ${texto_item}
**Dimensión**: ${dimension}
**Palabras clave identificadas**: ${palabras_clave}

Genera:
1. Una versión expandida del texto para búsqueda semántica
2. Términos relacionados y sinónimos
3. Conceptos asociados que podrían estar en ítems similares
4. Una consulta optimizada para vector search

Responde en formato JSON.
""")


# ==============================================================================
# PROMPTS PARA REPORTES Y DOCUMENTACIÓN
# ==============================================================================

PROMPT_GENERAR_RESUMEN_ANALISIS = Template("""
Genera un resumen ejecutivo del análisis psicométrico de un conjunto de ítems.

## DATOS DEL ANÁLISIS
- Total de ítems analizados: ${total_items}
- Ítems con problemas: ${items_problemas}
- Discriminación promedio: ${discriminacion_promedio}
- Dificultad promedio: ${dificultad_promedio}
- Ítems de excelente calidad: ${items_excelentes}

## DISTRIBUCIÓN DE CALIDAD
${distribucion_calidad}

## PROBLEMAS MÁS FRECUENTES
${problemas_frecuentes}

## INSTRUCCIONES
Genera un resumen ejecutivo que incluya:
1. Estado general del banco de ítems
2. Hallazgos principales
3. Recomendaciones prioritarias
4. Próximos pasos sugeridos

El tono debe ser profesional pero accesible.
""")


PROMPT_GENERAR_INFORME_ITEM = Template("""
Genera un informe técnico completo para el ítem:

**ID**: ${item_id}
**Texto**: ${texto_item}
**Dimensión**: ${dimension}

## PARÁMETROS PSICOMÉTRICOS
${parametros_texto}

## HISTORIAL DE APLICACIONES
${historial_texto}

Genera un informe que incluya:
1. Resumen ejecutivo
2. Análisis detallado de parámetros
3. Comparación con ítems similares
4. Tendencias temporales (si aplica)
5. Recomendaciones
6. Conclusión

El informe debe ser técnico pero comprensible.
""")


# ==============================================================================
# FUNCIONES DE CONSTRUCCIÓN DE PROMPTS
# ==============================================================================

def construir_prompt_generacion(
    contexto: ContextoGeneracion
) -> str:
    """
    Construye el prompt completo para generación de ítems.
    
    Args:
        contexto: Contexto de generación
    
    Returns:
        Prompt formateado
    """
    # Construir línea de dificultad
    dificultad_linea = ""
    dificultad_valor = "moderada (-1 a 1)"
    if contexto.dificultad_objetivo:
        dificultad_linea = f"- **Dificultad objetivo**: {contexto.dificultad_objetivo}"
        dificultad_map = {
            "muy_facil": "muy fácil (< -2.0)",
            "facil": "fácil (-2.0 a -1.0)",
            "moderada": "moderada (-1.0 a 1.0)",
            "dificil": "difícil (1.0 a 2.0)",
            "muy_dificil": "muy difícil (> 2.0)"
        }
        dificultad_valor = dificultad_map.get(contexto.dificultad_objetivo, "moderada (-1 a 1)")
    
    # Construir texto de ítems de referencia
    items_texto = "No se proporcionaron ítems de referencia."
    if contexto.items_referencia:
        items_refs = []
        for i, item in enumerate(contexto.items_referencia[:3], 1):
            items_refs.append(f"\n{i}. {item.get('texto', 'N/A')}")
            items_refs.append(f"   Parámetros: a={item.get('param_a', 'N/A')}, b={item.get('param_b', 'N/A')}")
        items_texto = "\n".join(items_refs)
    
    # Construir restricciones
    restricciones_texto = ""
    if contexto.restricciones:
        restrs = "\n".join(f"- {r}" for r in contexto.restricciones)
        restricciones_texto = f"\n## RESTRICCIONES ADICIONALES\n{restrs}"
    
    return PROMPT_GENERACION_ITEM.substitute(
        dimension=contexto.dimension,
        descripcion_dimension=contexto.descripcion_dimension,
        nivel_educativo=contexto.nivel_educativo,
        poblacion_objetivo=contexto.poblacion_objetivo,
        formato_respuesta=contexto.formato_respuesta,
        dificultad_linea=dificultad_linea,
        dificultad_objetivo_valor=dificultad_valor,
        items_referencia_texto=items_texto,
        restricciones_texto=restricciones_texto
    )


def construir_prompt_analisis(
    contexto: ContextoAnalisis
) -> str:
    """
    Construye el prompt para análisis de calidad de ítem.
    
    Args:
        contexto: Contexto de análisis
    
    Returns:
        Prompt formateado
    """
    # Formatear opciones
    opciones_texto = "\n".join(f"- {op}" for op in contexto.opciones_respuesta) if contexto.opciones_respuesta else "No especificadas"
    
    # Parámetros adicionales
    param_c_linea = ""
    if 'param_c' in contexto.parametros_irt and contexto.parametros_irt['param_c'] is not None:
        param_c_linea = f"- Parámetro de adivinanza (c): {contexto.parametros_irt['param_c']}"
    
    chi_cuadrado_linea = ""
    if contexto.indices_ajuste and 'chi_cuadrado' in contexto.indices_ajuste:
        chi_cuadrado_linea = f"- Chi-cuadrado: {contexto.indices_ajuste['chi_cuadrado']}"
    
    confiabilidad_linea = ""
    if contexto.indices_ajuste and 'confiabilidad_test_retest' in contexto.indices_ajuste:
        confiabilidad_linea = f"- Confiabilidad test-retest: {contexto.indices_ajuste['confiabilidad_test_retest']}"
    
    # Interpretación de parámetros
    param_a = contexto.parametros_irt.get('param_a', 0)
    param_b = contexto.parametros_irt.get('param_b', 0)
    p_valor = contexto.indices_ajuste.get('p_valor', None) if contexto.indices_ajuste else None
    
    interpretacion = []
    if param_a < 0.5:
        interpretacion.append("- Discriminación baja: el ítem diferencia poco entre niveles de habilidad")
    elif param_a > 1.5:
        interpretacion.append("- Discriminación alta: el ítem diferencia muy bien entre niveles")
    
    if abs(param_b) > 2:
        interpretacion.append(f"- Dificultad extrema ({param_b}): el ítem está en un extremo de la distribución")
    
    if p_valor is not None and p_valor < 0.05:
        interpretacion.append(f"- Mal ajuste (p={p_valor}): el ítem no se ajusta bien al modelo")
    
    interpretacion_texto = "\n".join(interpretacion) if interpretacion else "- Parámetros dentro de rangos normales"
    
    return PROMPT_ANALISIS_CALIDAD_ITEM.substitute(
        texto_item=contexto.texto_item,
        opciones_texto=opciones_texto,
        param_a=contexto.parametros_irt.get('param_a', 'N/A'),
        param_b=contexto.parametros_irt.get('param_b', 'N/A'),
        param_c_linea=param_c_linea,
        p_valor=p_valor if p_valor is not None else 'N/A',
        chi_cuadrado_linea=chi_cuadrado_linea,
        confiabilidad_linea=confiabilidad_linea,
        interpretacion_parametros=interpretacion_texto
    )


def construir_prompt_explicacion(
    param_a: float,
    param_b: float,
    p_valor: Optional[float] = None,
    param_c: Optional[float] = None
) -> str:
    """
    Construye el prompt para explicación de parámetros.
    
    Args:
        param_a: Discriminación
        param_b: Dificultad
        p_valor: P-valor de ajuste
        param_c: Adivinanza
    
    Returns:
        Prompt formateado
    """
    param_c_linea = ""
    if param_c is not None:
        param_c_linea = f"- **Parámetro de adivinanza (c)**: {param_c}"
    
    return PROMPT_EXPLICACION_PARAMETROS.substitute(
        param_a=param_a,
        param_b=param_b,
        param_c_linea=param_c_linea,
        p_valor=p_valor if p_valor is not None else 'N/A'
    )


def construir_prompt_sugerencias(
    texto_item: str,
    opciones: List[str],
    problemas: List[Dict[str, Any]],
    dimension: str,
    nivel_educativo: str = "general",
    dificultad_objetivo: str = "moderada"
) -> str:
    """
    Construye el prompt para sugerencias de mejora.
    
    Args:
        texto_item: Texto del ítem
        opciones: Opciones de respuesta
        problemas: Lista de problemas detectados
        dimension: Dimensión del ítem
        nivel_educativo: Nivel educativo
        dificultad_objetivo: Dificultad objetivo
    
    Returns:
        Prompt formateado
    """
    opciones_texto = ", ".join(opciones) if opciones else "No especificadas"
    
    problemas_texto = "\n".join(
        f"- {p.get('tipo', 'Problema')}: {p.get('descripcion', 'Sin descripción')}"
        for p in problemas
    ) if problemas else "No se detectaron problemas específicos."
    
    return PROMPT_SUGERIR_MEJORAS.substitute(
        texto_item=texto_item,
        opciones_texto=opciones_texto,
        problemas_texto=problemas_texto,
        dimension=dimension,
        nivel_educativo=nivel_educativo,
        dificultad_objetivo=dificultad_objetivo
    )


def construir_prompt_validacion(
    texto_item: str,
    dimension: str,
    descripcion_dimension: str
) -> str:
    """
    Construye el prompt para validación de nuevo ítem.
    
    Args:
        texto_item: Texto del ítem
        dimension: Dimensión
        descripcion_dimension: Descripción de la dimensión
    
    Returns:
        Prompt formateado
    """
    return PROMPT_VALIDACION_NUEVO_ITEM.substitute(
        texto_item=texto_item,
        dimension=dimension,
        descripcion_dimension=descripcion_dimension
    )


# ==============================================================================
# EXPORTAR
# ==============================================================================

__all__ = [
    # Estructuras de datos
    'ContextoGeneracion',
    'ContextoAnalisis',
    
    # Templates de prompts
    'PROMPT_GENERACION_ITEM',
    'PROMPT_GENERACION_ITEM_SIN_REFERENCIA',
    'PROMPT_ANALISIS_CALIDAD_ITEM',
    'PROMPT_ANALISIS_COMPARATIVO',
    'PROMPT_EXPLICACION_PARAMETROS',
    'PROMPT_EXPLICACION_MODELO_IRT',
    'PROMPT_SUGERIR_MEJORAS',
    'PROMPT_VALIDACION_NUEVO_ITEM',
    'PROMPT_BUSQUEDA_SIMILITUD',
    'PROMPT_GENERACION_CONSULTA',
    'PROMPT_GENERAR_RESUMEN_ANALISIS',
    'PROMPT_GENERAR_INFORME_ITEM',
    
    # Funciones de construcción
    'construir_prompt_generacion',
    'construir_prompt_analisis',
    'construir_prompt_explicacion',
    'construir_prompt_sugerencias',
    'construir_prompt_validacion',
]
