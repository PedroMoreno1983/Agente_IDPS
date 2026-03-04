"""
================================================================================
TESTS - MOTOR DE REGLAS PSICOMÉTRICAS
================================================================================

Pruebas unitarias para el motor de reglas psicométricas.

Ejecutar con: pytest tests/test_reglas_psicometricas.py -v
================================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import math

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
    calcular_probabilidad_3pl,
    calcular_informacion_item,
    NivelDificultad,
    NivelDiscriminacion,
    CalidadItem,
    TipoProblema,
    UmbralesPsicometricos
)


class TestValidacionParametros(unittest.TestCase):
    """Pruebas para validación de parámetros IRT."""
    
    def test_parametros_validos_2pl(self):
        """Parámetros 2PL válidos deben pasar."""
        es_valido, errores = validar_parametros_irt(1.2, 0.5)
        self.assertTrue(es_valido)
        self.assertEqual(len(errores), 0)
    
    def test_discriminacion_negativa(self):
        """Discriminación negativa debe fallar."""
        es_valido, errores = validar_parametros_irt(-0.5, 0.5)
        self.assertFalse(es_valido)
        self.assertTrue(any("negativa" in e.lower() for e in errores))
    
    def test_discriminacion_cero(self):
        """Discriminación cero debe fallar."""
        es_valido, errores = validar_parametros_irt(0, 0.5)
        self.assertFalse(es_valido)
    
    def test_dificultad_extrema(self):
        """Dificultad extrema debe generar advertencia."""
        es_valido, errores = validar_parametros_irt(1.0, 4.5)
        self.assertFalse(es_valido)
        self.assertTrue(any("extrema" in e.lower() for e in errores))
    
    def test_modelo_3pl_sin_c(self):
        """Modelo 3PL sin parámetro c debe fallar."""
        es_valido, errores = validar_parametros_irt(1.0, 0.5, modelo="3PL")
        self.assertFalse(es_valido)
        self.assertTrue(any("adivinanza" in e.lower() for e in errores))
    
    def test_adivinanza_alta(self):
        """Adivinanza demasiado alta debe fallar."""
        es_valido, errores = validar_parametros_irt(1.0, 0.5, param_c=0.6, modelo="3PL")
        self.assertFalse(es_valido)


class TestClasificacionDificultad(unittest.TestCase):
    """Pruebas para clasificación de dificultad."""
    
    def test_muy_facil(self):
        self.assertEqual(clasificar_dificultad(-2.5), NivelDificultad.MUY_FACIL)
        self.assertEqual(clasificar_dificultad(-3.0), NivelDificultad.MUY_FACIL)
    
    def test_facil(self):
        self.assertEqual(clasificar_dificultad(-1.5), NivelDificultad.FACIL)
        self.assertEqual(clasificar_dificultad(-1.0), NivelDificultad.FACIL)
    
    def test_moderada(self):
        self.assertEqual(clasificar_dificultad(0), NivelDificultad.MODERADA)
        self.assertEqual(clasificar_dificultad(0.5), NivelDificultad.MODERADA)
        self.assertEqual(clasificar_dificultad(-0.5), NivelDificultad.MODERADA)
    
    def test_dificil(self):
        self.assertEqual(clasificar_dificultad(1.5), NivelDificultad.DIFICIL)
        self.assertEqual(clasificar_dificultad(2.0), NivelDificultad.DIFICIL)
    
    def test_muy_dificil(self):
        self.assertEqual(clasificar_dificultad(2.5), NivelDificultad.MUY_DIFICIL)
        self.assertEqual(clasificar_dificultad(3.0), NivelDificultad.MUY_DIFICIL)


class TestClasificacionDiscriminacion(unittest.TestCase):
    """Pruebas para clasificación de discriminación."""
    
    def test_baja(self):
        self.assertEqual(clasificar_discriminacion(0.3), NivelDiscriminacion.BAJA)
        self.assertEqual(clasificar_discriminacion(0.6), NivelDiscriminacion.BAJA)
    
    def test_moderada(self):
        self.assertEqual(clasificar_discriminacion(0.7), NivelDiscriminacion.MODERADA)
        self.assertEqual(clasificar_discriminacion(1.0), NivelDiscriminacion.MODERADA)
        self.assertEqual(clasificar_discriminacion(1.3), NivelDiscriminacion.MODERADA)
    
    def test_alta(self):
        self.assertEqual(clasificar_discriminacion(1.4), NivelDiscriminacion.ALTA)
        self.assertEqual(clasificar_discriminacion(1.6), NivelDiscriminacion.ALTA)
    
    def test_muy_alta(self):
        self.assertEqual(clasificar_discriminacion(1.7), NivelDiscriminacion.MUY_ALTA)
        self.assertEqual(clasificar_discriminacion(2.0), NivelDiscriminacion.MUY_ALTA)


class TestDeteccionProblemas(unittest.TestCase):
    """Pruebas para detección de problemas."""
    
    def test_ajuste_aceptable(self):
        tiene_problema, msg = detectar_problemas_ajuste(0.15)
        self.assertFalse(tiene_problema)
        self.assertIn("aceptable", msg.lower())
    
    def test_mal_ajuste(self):
        tiene_problema, msg = detectar_problemas_ajuste(0.001)
        self.assertTrue(tiene_problema)
        self.assertIn("mal ajuste", msg.lower())
    
    def test_p_valor_nulo(self):
        tiene_problema, msg = detectar_problemas_ajuste(None)
        self.assertFalse(tiene_problema)
    
    def test_discriminacion_negativa(self):
        problemas = detectar_todos_problemas(-0.5, 0)
        self.assertIn(TipoProblema.DISCRIMINACION_NEGATIVA, problemas)
    
    def test_discriminacion_baja(self):
        problemas = detectar_todos_problemas(0.4, 0)
        self.assertIn(TipoProblema.DISCRIMINACION_BAJA, problemas)
    
    def test_dificultad_extrema(self):
        problemas = detectar_todos_problemas(1.0, 3.5)
        self.assertIn(TipoProblema.DIFICULTAD_EXTREMA, problemas)
    
    def test_mal_ajuste_detectado(self):
        problemas = detectar_todos_problemas(1.0, 0, p_valor=0.001)
        self.assertIn(TipoProblema.MAL_AJUSTE, problemas)


class TestEvaluacionCalidad(unittest.TestCase):
    """Pruebas para evaluación de calidad."""
    
    def test_calidad_excelente(self):
        calidad = evaluar_calidad_item(1.8, 0.3, p_valor=0.25)
        self.assertEqual(calidad, CalidadItem.EXCELENTE)
    
    def test_calidad_buena(self):
        calidad = evaluar_calidad_item(1.2, 0.5, p_valor=0.10)
        self.assertIn(calidad, [CalidadItem.EXCELENTE, CalidadItem.BUENA])
    
    def test_calidad_deficiente_discriminacion(self):
        calidad = evaluar_calidad_item(0.2, 0, p_valor=0.001)
        self.assertIn(calidad, [CalidadItem.CUESTIONABLE, CalidadItem.DEFICIENTE])
    
    def test_calidad_mal_ajuste(self):
        calidad = evaluar_calidad_item(1.0, 0, p_valor=0.001)
        self.assertIn(calidad, [CalidadItem.ACEPTABLE, CalidadItem.CUESTIONABLE])


class TestGeneracionRecomendaciones(unittest.TestCase):
    """Pruebas para generación de recomendaciones."""
    
    def test_recomendacion_discriminacion_negativa(self):
        item_data = {'param_a': -0.5, 'param_b': 0}
        recomendaciones = generar_recomendacion(item_data)
        self.assertTrue(any("negativa" in r.lower() for r in recomendaciones))
    
    def test_recomendacion_discriminacion_baja(self):
        item_data = {'param_a': 0.4, 'param_b': 0}
        recomendaciones = generar_recomendacion(item_data)
        self.assertTrue(any("baja" in r.lower() for r in recomendaciones))
    
    def test_recomendacion_item_bueno(self):
        item_data = {'param_a': 1.2, 'param_b': 0.3}
        recomendaciones = generar_recomendacion(item_data)
        self.assertTrue(any("buena calidad" in r.lower() or "excelente" in r.lower() 
                           for r in recomendaciones))


class TestCalculosIRT(unittest.TestCase):
    """Pruebas para cálculos IRT."""
    
    def test_probabilidad_2pl_theta_igual_b(self):
        """Cuando θ = b, la probabilidad debe ser 0.5."""
        prob = calcular_probabilidad_2pl(0, 1.0, 0)
        self.assertAlmostEqual(prob, 0.5, places=5)
    
    def test_probabilidad_2pl_theta_mayor_b(self):
        """Cuando θ > b, la probabilidad debe ser > 0.5."""
        prob = calcular_probabilidad_2pl(1.0, 1.0, 0)
        self.assertGreater(prob, 0.5)
    
    def test_probabilidad_2pl_theta_menor_b(self):
        """Cuando θ < b, la probabilidad debe ser < 0.5."""
        prob = calcular_probabilidad_2pl(-1.0, 1.0, 0)
        self.assertLess(prob, 0.5)
    
    def test_probabilidad_3pl_cero(self):
        """Probabilidad 3PL debe ser >= c."""
        prob = calcular_probabilidad_3pl(-10, 1.0, 0, 0.2)
        self.assertGreaterEqual(prob, 0.2)
    
    def test_informacion_maxima_en_b(self):
        """La información es máxima cerca de θ = b."""
        info_en_b = calcular_informacion_item(0, 1.0, 0)
        info_lejos = calcular_informacion_item(3, 1.0, 0)
        self.assertGreater(info_en_b, info_lejos)


class TestComparacionItems(unittest.TestCase):
    """Pruebas para comparación de ítems."""
    
    def test_comparar_items_similares(self):
        item1 = {'param_a': 1.0, 'param_b': 0.5}
        item2 = {'param_a': 1.1, 'param_b': 0.6}
        resultado = comparar_items(item1, item2)
        self.assertIn("diferencias", resultado)
        self.assertIn("similitudes", resultado)
    
    def test_comparar_items_diferentes(self):
        item1 = {'param_a': 1.0, 'param_b': -2.0}
        item2 = {'param_a': 0.5, 'param_b': 2.0}
        resultado = comparar_items(item1, item2)
        self.assertIn("diferentes", resultado.get("conclusion", "").lower())


class TestUmbrales(unittest.TestCase):
    """Pruebas para constantes y umbrales."""
    
    def test_umbrales_definidos(self):
        self.assertIsNotNone(UmbralesPsicometricos.DISCRIMINACION_BAJA_MAX)
        self.assertIsNotNone(UmbralesPsicometricos.DIFICULTAD_MODERADA_MIN)
        self.assertIsNotNone(UmbralesPsicometricos.P_VALOR_UMBRAL_SIGNIFICATIVO)


if __name__ == '__main__':
    unittest.main()
