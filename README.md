# Agente de IA Psicométrico

Sistema de agente de inteligencia artificial especializado en psicometría educativa para la generación, análisis y validación de ítems de evaluación.

## 📋 Descripción

Este sistema implementa las capacidades psicométricas para un Agente de IA que asiste en el ciclo de vida completo de ítems psicométricos, desde la generación basada en datos históricos hasta el análisis de su efectividad.

## 🏗️ Arquitectura

```
agente_psicometrico/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── reglas_psicometricas.py    # Motor de reglas IRT
│   │   ├── utils_psicometricos.py     # Utilidades y cálculos
│   │   └── prompts.py                  # Prompts para LLM
│   ├── services/
│   │   ├── __init__.py
│   │   ├── analisis_service.py        # Análisis de ítems
│   │   └── generacion_service.py      # Generación de ítems
│   └── __init__.py
├── tests/
│   └── test_reglas_psicometricas.py
└── README.md
```

## 🚀 Componentes Principales

### 1. Motor de Reglas Psicométricas (`reglas_psicometricas.py`)

Implementa las reglas y algoritmos psicométricos:

- **Validación de parámetros IRT**: `validar_parametros_irt(param_a, param_b, param_c)`
- **Evaluación de calidad**: `evaluar_calidad_item(param_a, param_b, p_valor)`
- **Clasificación de dificultad**: `clasificar_dificultad(param_b)`
- **Clasificación de discriminación**: `clasificar_discriminacion(param_a)`
- **Detección de problemas**: `detectar_problemas_ajuste(p_valor)`
- **Generación de recomendaciones**: `generar_recomendacion(item_data)`
- **Cálculos IRT**: `calcular_probabilidad_2pl()`, `calcular_informacion_item()`

### 2. Servicio de Análisis Psicométrico (`analisis_service.py`)

Proporciona análisis completo de ítems:

```python
from app.services.analisis_service import ServicioAnalisisPsicometrico

servicio = ServicioAnalisisPsicometrico()

# Analizar un ítem
analisis = servicio.analizar_item(
    item_id=1,
    parametros={'param_a': 1.2, 'param_b': 0.5, 'p_valor': 0.25}
)

# Comparar ítems
comparacion = servicio.comparar_items([1, 2, 3])

# Identificar problemas
problemas = servicio.identificar_items_problematicos(dimension_id=1)

# Estadísticas por dimensión
stats = servicio.calcular_estadisticas_dimension(dimension_id=1)
```

### 3. Servicio de Generación de Ítems (`generacion_service.py`)

Genera ítems usando LLM + contexto histórico:

```python
from app.services.generacion_service import ServicioGeneracionItems

servicio = ServicioGeneracionItems(llm_client=cliente_openai)

# Generar ítem sugerido
item = servicio.generar_item_sugerido(
    dimension_id=1,
    dificultad_objetivo="moderada",
    formato_respuesta="likert_4"
)

# Sugerir mejoras
mejoras = servicio.sugerir_mejoras(item_id=1)

# Validar nuevo ítem
validacion = servicio.validar_nuevo_item(
    texto_item="Me siento seguro de mis habilidades",
    dimension_id=1
)

# Buscar ítems similares
similares = servicio.buscar_items_similares(
    texto_item="Consulta de ejemplo",
    top_k=5
)
```

### 4. Prompts Especializados (`prompts.py`)

Templates de prompts para interacción con LLM:

- `construir_prompt_generacion(contexto)`: Generación de ítems
- `construir_prompt_analisis(contexto)`: Análisis de calidad
- `construir_prompt_explicacion(param_a, param_b)`: Explicación de parámetros
- `construir_prompt_sugerencias(...)`: Sugerencias de mejora
- `construir_prompt_validacion(...)`: Validación de ítems

### 5. Utilidades Psicométricas (`utils_psicometricos.py`)

Funciones auxiliares:

- **Transformaciones**: `z_a_logit()`, `p_a_logit()`, `normalizar_puntuacion()`
- **Estadísticas**: `calcular_media()`, `calcular_correlacion_pearson()`, `calcular_coeficiente_alfa_cronbach()`
- **Validación**: `validar_texto_item()`, `validar_opciones_respuesta()`, `detectar_lenguaje_inclusivo()`
- **Análisis de texto**: `extraer_palabras_clave()`, `calcular_legibilidad()`

## 📊 Modelos IRT Soportados

### Modelo 2PL (Two Parameter Logistic)
- **Parámetro a**: Discriminación (capacidad para diferenciar niveles)
- **Parámetro b**: Dificultad (nivel del constructo donde es más informativo)

### Modelo 3PL (Three Parameter Logistic)
- Incluye parámetro **c**: Adivinanza (probabilidad de acertar por azar)

### Modelo GRM (Graded Response Model)
- Para ítems politómicos (más de 2 categorías de respuesta)

## 🎯 Umbrales Psicométricos

### Discriminación (param_a)
| Nivel | Rango | Interpretación |
|-------|-------|----------------|
| Baja | < 0.65 | Diferencia poco entre niveles |
| Moderada | 0.65 - 1.34 | Diferencia adecuadamente |
| Alta | 1.35 - 1.69 | Diferencia muy bien |
| Muy Alta | ≥ 1.70 | Excelente capacidad diferenciadora |

### Dificultad (param_b)
| Nivel | Rango (logits) | Interpretación |
|-------|----------------|----------------|
| Muy Fácil | ≤ -2.0 | Mayoría responde correctamente |
| Fácil | -2.0 a -1.0 | Estudiantes bajos responden bien |
| Moderada | -1.0 a 1.0 | Diferencia bien entre niveles |
| Difícil | 1.0 a 2.0 | Solo altos responden bien |
| Muy Difícil | ≥ 2.0 | Muy pocos responden correctamente |

### Ajuste del Modelo
- **p-valor ≥ 0.05**: Ajuste aceptable
- **p-valor < 0.05**: Mal ajuste (revisar ítem)
- **p-valor < 0.01**: Mal ajuste significativo

## 📝 Ejemplos de Uso

### Ejemplo 1: Validar parámetros IRT

```python
from app.core.reglas_psicometricas import validar_parametros_irt

es_valido, errores = validar_parametros_irt(
    param_a=1.2,
    param_b=0.5,
    param_c=None,
    modelo="2PL"
)

if es_valido:
    print("✅ Parámetros válidos")
else:
    print("❌ Errores encontrados:")
    for error in errores:
        print(f"  - {error}")
```

### Ejemplo 2: Clasificar un ítem

```python
from app.core.reglas_psicometricas import (
    clasificar_dificultad,
    clasificar_discriminacion,
    obtener_descripcion_dificultad
)

nivel_dif = clasificar_dificultad(0.8)
nivel_disc = clasificar_discriminacion(1.3)

print(f"Dificultad: {nivel_dif.value}")
print(f"Discriminación: {nivel_disc.value}")
print(obtener_descripcion_dificultad(nivel_dif))
```

### Ejemplo 3: Generar recomendaciones

```python
from app.core.reglas_psicometricas import generar_recomendacion

item_data = {
    'param_a': 0.4,
    'param_b': 2.5,
    'p_valor': 0.001
}

recomendaciones = generar_recomendacion(item_data)
for rec in recomendaciones:
    print(rec)
```

### Ejemplo 4: Calcular probabilidades IRT

```python
from app.core.reglas_psicometricas import calcular_probabilidad_2pl

# Probabilidad de respuesta correcta para diferentes niveles de habilidad
theta_values = [-2, -1, 0, 1, 2]
for theta in theta_values:
    prob = calcular_probabilidad_2pl(theta, param_a=1.0, param_b=0)
    print(f"θ = {theta}: P = {prob:.3f}")
```

## 🧪 Pruebas

Ejecutar las pruebas unitarias:

```bash
# Todas las pruebas
python -m pytest tests/ -v

# Pruebas específicas
python -m pytest tests/test_reglas_psicometricas.py -v

# Con cobertura
python -m pytest tests/ --cov=app --cov-report=html
```

## 📚 Dependencias

```
Python >= 3.8
numpy (opcional, para cálculos avanzados)
scipy (opcional, para estadísticas)
pandas (opcional, para manipulación de datos)
```

## 🔧 Instalación

```bash
# Clonar repositorio
git clone <repositorio>
cd agente_psicometrico

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "from app.core.reglas_psicometricas import *; print('✅ OK')"
```

## 📖 Referencias

- **IRT**: Hambleton, R. K., & Swaminathan, H. (1985). Item Response Theory: Principles and Applications.
- **Estándares**: AERA, APA, & NCME (2014). Standards for Educational and Psychological Testing.
- **Psicometría en Educación**: Muñiz, J. (2010). Las teorías de los tests: Teoría Clásica y Teoría de Respuesta a los Ítems.

## 📄 Licencia

Este proyecto es parte del Sistema Integrado de Análisis Psicométrico para instituciones educativas públicas de Chile.

## 👥 Autor

Agente de IA Psicométrico - Versión 1.0.0
