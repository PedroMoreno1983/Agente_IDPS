#!/usr/bin/env python3
"""
================================================================================
EJEMPLO DE USO - AGENTE PSICOMÉTRICO
================================================================================

Este script demuestra las capacidades del Agente de IA Psicométrico
con ejemplos prácticos de uso.

Ejecutar: python ejemplo_uso.py
================================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.reglas_psicometricas import (
    validar_parametros_irt,
    evaluar_calidad_item,
    clasificar_dificultad,
    clasificar_discriminacion,
    detectar_problemas_ajuste,
    detectar_todos_problemas,
    generar_recomendacion,
    comparar_items,
    calcular_probabilidad_2pl,
    calcular_informacion_item,
    obtener_descripcion_dificultad,
    obtener_descripcion_discriminacion,
    NivelDificultad,
    NivelDiscriminacion,
    CalidadItem
)

from app.core.utils_psicometricos import (
    calcular_estadisticas_descriptivas,
    validar_texto_item,
    extraer_palabras_clave,
    formatear_parametro_psicometrico,
    ConstantesPsicometricas
)

from app.services.analisis_service import ServicioAnalisisPsicometrico
from app.services.generacion_service import ServicioGeneracionItems


def print_seccion(titulo):
    """Imprime una sección formateada."""
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print("=" * 70)


def ejemplo_1_validacion_parametros():
    """Ejemplo 1: Validación de parámetros IRT."""
    print_seccion("EJEMPLO 1: Validación de Parámetros IRT")
    
    # Caso 1: Parámetros válidos
    print("\n📊 Caso 1: Parámetros válidos (2PL)")
    es_valido, errores = validar_parametros_irt(param_a=1.2, param_b=0.5)
    print(f"   Parámetros: a=1.2, b=0.5")
    print(f"   ✅ Válido: {es_valido}")
    if errores:
        for e in errores:
            print(f"   ⚠️  {e}")
    
    # Caso 2: Discriminación negativa (problema grave)
    print("\n📊 Caso 2: Discriminación negativa (problema)")
    es_valido, errores = validar_parametros_irt(param_a=-0.3, param_b=0.5)
    print(f"   Parámetros: a=-0.3, b=0.5")
    print(f"   ❌ Válido: {es_valido}")
    for e in errores:
        print(f"   ⚠️  {e}")
    
    # Caso 3: Modelo 3PL con adivinanza
    print("\n📊 Caso 3: Modelo 3PL con adivinanza")
    es_valido, errores = validar_parametros_irt(
        param_a=1.0, param_b=0.5, param_c=0.25, modelo="3PL"
    )
    print(f"   Parámetros: a=1.0, b=0.5, c=0.25")
    print(f"   ✅ Válido: {es_valido}")


def ejemplo_2_clasificacion():
    """Ejemplo 2: Clasificación de ítems."""
    print_seccion("EJEMPLO 2: Clasificación de Ítems")
    
    items = [
        {"nombre": "Ítem Fácil", "a": 0.8, "b": -1.5},
        {"nombre": "Ítem Moderado", "a": 1.2, "b": 0.0},
        {"nombre": "Ítem Difícil", "a": 1.5, "b": 1.8},
        {"nombre": "Ítem Problemático", "a": 0.3, "b": 3.5},
    ]
    
    for item in items:
        print(f"\n📋 {item['nombre']}")
        print(f"   Parámetros: a={item['a']}, b={item['b']}")
        
        nivel_dif = clasificar_dificultad(item['b'])
        nivel_disc = clasificar_discriminacion(item['a'])
        
        print(f"   📈 Dificultad: {nivel_dif.value}")
        print(f"      {obtener_descripcion_dificultad(nivel_dif)}")
        print(f"   📊 Discriminación: {nivel_disc.value}")
        print(f"      {obtener_descripcion_discriminacion(nivel_disc)}")


def ejemplo_3_evaluacion_calidad():
    """Ejemplo 3: Evaluación de calidad de ítems."""
    print_seccion("EJEMPLO 3: Evaluación de Calidad")
    
    items = [
        {"nombre": "Excelente", "a": 1.8, "b": 0.3, "p_valor": 0.25},
        {"nombre": "Bueno", "a": 1.2, "b": 0.5, "p_valor": 0.10},
        {"nombre": "Cuestionable", "a": 0.5, "b": 2.5, "p_valor": 0.03},
        {"nombre": "Deficiente", "a": 0.2, "b": -3.0, "p_valor": 0.001},
    ]
    
    for item in items:
        calidad = evaluar_calidad_item(
            param_a=item['a'],
            param_b=item['b'],
            p_valor=item['p_valor']
        )
        
        print(f"\n📋 Ítem: {item['nombre']}")
        print(f"   Parámetros: a={item['a']}, b={item['b']}, p={item['p_valor']}")
        print(f"   🏆 Calidad: {calidad.value.upper()}")


def ejemplo_4_deteccion_problemas():
    """Ejemplo 4: Detección de problemas."""
    print_seccion("EJEMPLO 4: Detección de Problemas")
    
    # Problema de ajuste
    print("\n📊 Análisis de ajuste del modelo:")
    for p_valor in [0.25, 0.04, 0.001]:
        tiene_problema, msg = detectar_problemas_ajuste(p_valor)
        icono = "❌" if tiene_problema else "✅"
        print(f"   {icono} p-valor = {p_valor}: {msg}")
    
    # Detección completa de problemas
    print("\n📋 Detección completa de problemas:")
    item_problematico = {
        'param_a': 0.3,
        'param_b': 3.2,
        'p_valor': 0.001,
        'confiabilidad_test_retest': 0.55
    }
    
    problemas = detectar_todos_problemas(**item_problematico)
    print(f"   Parámetros: a={item_problematico['param_a']}, "
          f"b={item_problematico['param_b']}, "
          f"p={item_problematico['p_valor']}")
    print(f"   ⚠️  Problemas detectados ({len(problemas)}):")
    for p in problemas:
        print(f"      - {p.value}")


def ejemplo_5_recomendaciones():
    """Ejemplo 5: Generación de recomendaciones."""
    print_seccion("EJEMPLO 5: Generación de Recomendaciones")
    
    items = [
        {"nombre": "Ítem con discriminación baja", "a": 0.4, "b": 0},
        {"nombre": "Ítem con dificultad extrema", "a": 1.2, "b": 3.5},
        {"nombre": "Ítem con mal ajuste", "a": 1.0, "b": 0, "p_valor": 0.001},
        {"nombre": "Ítem de buena calidad", "a": 1.3, "b": 0.2},
    ]
    
    for item in items:
        print(f"\n📋 {item['nombre']}")
        print(f"   Parámetros: a={item['a']}, b={item['b']}", end="")
        if 'p_valor' in item:
            print(f", p={item['p_valor']}")
        else:
            print()
        
        recomendaciones = generar_recomendacion(item)
        print("   💡 Recomendaciones:")
        for i, rec in enumerate(recomendaciones[:2], 1):
            print(f"      {i}. {rec[:100]}...")


def ejemplo_6_calculos_irt():
    """Ejemplo 6: Cálculos IRT."""
    print_seccion("EJEMPLO 6: Cálculos IRT")
    
    # Curva característica del ítem
    print("\n📊 Curva Característica del Ítem (CCI)")
    print("   Parámetros: a=1.5, b=0.0")
    print("\n   θ (habilidad) | P(θ) - Probabilidad de acierto")
    print("   " + "-" * 45)
    
    for theta in [-3, -2, -1, 0, 1, 2, 3]:
        prob = calcular_probabilidad_2pl(theta, param_a=1.5, param_b=0)
        barra = "█" * int(prob * 30)
        print(f"   {theta:>3}           | {prob:.3f} {barra}")
    
    # Función de información
    print("\n📊 Función de Información del Ítem")
    print("   Parámetros: a=1.5, b=0.0")
    print("\n   θ (habilidad) | I(θ) - Información")
    print("   " + "-" * 45)
    
    for theta in [-3, -2, -1, 0, 1, 2, 3]:
        info = calcular_informacion_item(theta, param_a=1.5, param_b=0)
        barra = "█" * int(info * 10)
        print(f"   {theta:>3}           | {info:.3f} {barra}")
    
    print("\n   ℹ️  La información es máxima cerca de θ = b (donde b=0)")


def ejemplo_7_comparacion_items():
    """Ejemplo 7: Comparación de ítems."""
    print_seccion("EJEMPLO 7: Comparación de Ítems")
    
    item1 = {
        'param_a': 1.2,
        'param_b': -0.5,
        'texto': 'Me siento bien en la escuela'
    }
    item2 = {
        'param_a': 1.0,
        'param_b': 0.5,
        'texto': 'Me siento seguro de mis habilidades'
    }
    
    print("\n📋 Ítem 1:")
    print(f"   Texto: {item1['texto']}")
    print(f"   a={item1['param_a']}, b={item1['param_b']}")
    
    print("\n📋 Ítem 2:")
    print(f"   Texto: {item2['texto']}")
    print(f"   a={item2['param_a']}, b={item2['param_b']}")
    
    resultado = comparar_items(item1, item2)
    
    print("\n📊 Resultado de la comparación:")
    print(f"   Diferencia en discriminación: {resultado['diferencias']['discriminacion']['valor']:.3f}")
    print(f"   Diferencia en dificultad: {resultado['diferencias']['dificultad']['valor']:.3f}")
    print(f"   Similitudes: {', '.join(resultado['similitudes']) if resultado['similitudes'] else 'Ninguna'}")
    print(f"   Conclusión: {resultado['conclusion']}")


def ejemplo_8_utilidades():
    """Ejemplo 8: Utilidades psicométricas."""
    print_seccion("EJEMPLO 8: Utilidades Psicométricas")
    
    # Validación de texto
    print("\n📋 Validación de texto de ítem:")
    textos = [
        "Me siento seguro de mis habilidades académicas.",
        "no me siento bien",  # Problema: minúscula inicial
        "Me siento   bien   hoy",  # Problema: dobles espacios
    ]
    
    for texto in textos:
        es_valido, errores = validar_texto_item(texto)
        icono = "✅" if es_valido else "❌"
        print(f"\n   {icono} \"{texto}\"")
        if errores:
            for e in errores:
                print(f"      ⚠️  {e}")
    
    # Extracción de palabras clave
    print("\n📋 Extracción de palabras clave:")
    texto = "Me siento seguro de mis habilidades para resolver problemas matemáticos"
    palabras_clave = extraer_palabras_clave(texto, n=5)
    print(f"   Texto: {texto}")
    print(f"   Palabras clave: {', '.join(palabras_clave)}")
    
    # Formateo de parámetros
    print("\n📋 Formateo de parámetros:")
    print(f"   Discriminación: {formatear_parametro_psicometrico(1.23456, 'discriminacion')}")
    print(f"   Dificultad: {formatear_parametro_psicometrico(-0.98765, 'dificultad')}")
    print(f"   P-valor: {formatear_parametro_psicometrico(0.00123, 'p_valor')}")


def ejemplo_9_servicio_analisis():
    """Ejemplo 9: Uso del servicio de análisis."""
    print_seccion("EJEMPLO 9: Servicio de Análisis Psicométrico")
    
    servicio = ServicioAnalisisPsicometrico()
    
    # Análisis de ítem
    print("\n📋 Análisis de ítem (simulado):")
    
    # Simulamos datos del ítem
    from app.services.analisis_service import DatosItem
    
    datos = DatosItem(
        item_id=1,
        texto="Me siento seguro de mis habilidades académicas",
        dimension_id=1,
        dimension_nombre="Autoestima Académica",
        formato_respuesta="likert_4"
    )
    
    parametros = {
        'param_a': 1.3,
        'param_b': 0.2,
        'p_valor': 0.18,
        'modelo_irt': '2PL'
    }
    
    analisis = servicio.analizar_item(1, datos, parametros)
    
    print(f"   Ítem ID: {analisis.item_id}")
    print(f"   Texto: {analisis.texto_item}")
    print(f"   Dimensión: {analisis.dimension}")
    print(f"   📊 Parámetros IRT:")
    print(f"      a (discriminación): {analisis.param_a}")
    print(f"      b (dificultad): {analisis.param_b}")
    print(f"   📈 Clasificación:")
    print(f"      Dificultad: {analisis.nivel_dificultad}")
    print(f"      Discriminación: {analisis.nivel_discriminacion}")
    print(f"      Calidad general: {analisis.calidad_general}")
    
    if analisis.problemas:
        print(f"   ⚠️  Problemas detectados: {len(analisis.problemas)}")
    else:
        print(f"   ✅ Sin problemas detectados")


def ejemplo_10_servicio_generacion():
    """Ejemplo 10: Uso del servicio de generación."""
    print_seccion("EJEMPLO 10: Servicio de Generación de Ítems")
    
    servicio = ServicioGeneracionItems()
    
    # Generar ítem (modo demo)
    print("\n📋 Generación de ítem (modo demo):")
    
    item = servicio.generar_item_sugerido(
        dimension_id=1,
        dificultad_objetivo="moderada",
        formato_respuesta="likert_4",
        nivel_educativo="secundaria"
    )
    
    print(f"   Texto: {item.texto}")
    print(f"   Opciones: {', '.join(item.opciones_respuesta)}")
    print(f"   Dirección: {item.direccion}")
    print(f"   Dificultad estimada: {item.nivel_dificultad_estimado}")
    print(f"   Justificación: {item.justificacion}")
    
    # Validar ítem
    print("\n📋 Validación de ítem:")
    
    validacion = servicio.validar_nuevo_item(
        texto_item="Me siento seguro de mis habilidades",
        dimension_id=1,
        opciones_respuesta=["Nunca", "A veces", "Casi siempre", "Siempre"],
        usar_llm=False
    )
    
    icono = "✅" if validacion.es_valido else "❌"
    print(f"   {icono} Válido: {validacion.es_valido}")
    print(f"   📊 Puntuación: {validacion.puntuacion}/100")
    print(f"   📝 Recomendación: {validacion.recomendacion_final}")
    
    if validacion.fortalezas:
        print(f"   ✅ Fortalezas:")
        for f in validacion.fortalezas[:3]:
            print(f"      - {f}")
    
    if validacion.debilidades:
        print(f"   ⚠️  Debilidades:")
        for d in validacion.debilidades[:3]:
            print(f"      - {d}")


def main():
    """Función principal que ejecuta todos los ejemplos."""
    print("\n" + "=" * 70)
    print("  AGENTE DE IA PSICOMÉTRICO - DEMOSTRACIÓN DE CAPACIDADES")
    print("=" * 70)
    
    ejemplos = [
        ("Validación de Parámetros", ejemplo_1_validacion_parametros),
        ("Clasificación de Ítems", ejemplo_2_clasificacion),
        ("Evaluación de Calidad", ejemplo_3_evaluacion_calidad),
        ("Detección de Problemas", ejemplo_4_deteccion_problemas),
        ("Generación de Recomendaciones", ejemplo_5_recomendaciones),
        ("Cálculos IRT", ejemplo_6_calculos_irt),
        ("Comparación de Ítems", ejemplo_7_comparacion_items),
        ("Utilidades Psicométricas", ejemplo_8_utilidades),
        ("Servicio de Análisis", ejemplo_9_servicio_analisis),
        ("Servicio de Generación", ejemplo_10_servicio_generacion),
    ]
    
    print(f"\n  Total de ejemplos: {len(ejemplos)}")
    print("\n  Presiona Enter para comenzar...")
    input()
    
    for nombre, funcion in ejemplos:
        try:
            funcion()
            print("\n  ✅ Ejemplo completado")
        except Exception as e:
            print(f"\n  ❌ Error en ejemplo: {e}")
        
        print("\n  Presiona Enter para continuar...")
        input()
    
    print("\n" + "=" * 70)
    print("  DEMOSTRACIÓN COMPLETADA")
    print("=" * 70)
    print("\n  Para más información, consulta el README.md")
    print()


if __name__ == "__main__":
    main()
