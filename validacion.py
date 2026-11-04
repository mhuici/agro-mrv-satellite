"""
validacion.py — Framework de validación para AgTech MRV

Compara los resultados del análisis satelital (gee_processor.procesar_lote)
contra datos de ground truth verificados en campo, y calcula métricas de
accuracy, precision, recall, F1-score y matriz de confusión.

Clasificación binaria:
  - Positivo: siembra_directa (diagnostico.consistente = True)
  - Negativo: convencional / quema (diagnostico.consistente = False)

Objetivo de accuracy: >85%
"""

import json
import sys
import datetime
from typing import List, Dict, Any, Optional

from gee_processor import procesar_lote


# =============================================================================
# Carga de ground truth
# =============================================================================

def cargar_ground_truth(path: str) -> List[Dict[str, Any]]:
    """Carga datos de ground truth desde un archivo JSON.

    Cada registro debe tener:
      - geojson: ruta al archivo GeoJSON del polígono
      - fecha_inicio: inicio del período de análisis (YYYY-MM-DD)
      - fecha_fin: fin del período de análisis (YYYY-MM-DD)
      - practica_real: 'siembra_directa', 'convencional' o 'quema'
      - fuente: quién validó en campo (ej: nombre del ingeniero agrónomo)

    Args:
        path: ruta al archivo JSON con los registros de ground truth.

    Returns:
        Lista de diccionarios con los registros de ground truth.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el formato es incorrecto o faltan campos obligatorios.
    """
    campos_requeridos = {
        'geojson', 'fecha_inicio', 'fecha_fin', 'practica_real', 'fuente'
    }
    practicas_validas = {'siembra_directa', 'convencional', 'quema'}

    with open(path, 'r', encoding='utf-8') as f:
        datos = json.load(f)

    if not isinstance(datos, list):
        raise ValueError(
            f"El archivo debe contener una lista JSON, "
            f"se encontró: {type(datos).__name__}"
        )

    if len(datos) == 0:
        raise ValueError("El archivo de ground truth está vacío")

    for i, registro in enumerate(datos):
        faltantes = campos_requeridos - set(registro.keys())
        if faltantes:
            raise ValueError(
                f"Registro #{i + 1}: faltan campos obligatorios: "
                f"{', '.join(sorted(faltantes))}"
            )

        if registro['practica_real'] not in practicas_validas:
            raise ValueError(
                f"Registro #{i + 1}: practica_real '{registro['practica_real']}' "
                f"no es válida. Opciones: {', '.join(sorted(practicas_validas))}"
            )

    print(f"Ground truth cargado: {len(datos)} registro(s) desde {path}")
    return datos


# =============================================================================
# Ejecución de validación
# =============================================================================

def _practica_a_consistente(practica: str) -> bool:
    """Convierte la práctica real a valor esperado de consistente.

    siembra_directa → True (positivo)
    convencional / quema → False (negativo)
    """
    return practica == 'siembra_directa'


def ejecutar_validacion(
    ground_truth: List[Dict[str, Any]],
    proyecto_gee: str = None
) -> List[Dict[str, Any]]:
    """Ejecuta el análisis satelital sobre cada registro de ground truth
    y compara la predicción con la práctica real.

    Para cada registro:
      1. Llama a procesar_lote() con el polígono y las fechas
      2. Extrae diagnostico.consistente como predicción
      3. Compara contra la práctica real

    Args:
        ground_truth: lista de registros de ground truth (ver cargar_ground_truth).
        proyecto_gee: ID del proyecto GEE (opcional).

    Returns:
        Lista de dicts con:
          - practica_real: práctica verificada en campo
          - prediccion: 'siembra_directa' o 'no_siembra_directa'
          - consistente_real: True si la práctica es siembra_directa
          - consistente_predicho: valor de diagnostico.consistente
          - correcto: True si la predicción coincide con la realidad
          - error: mensaje de error si el procesamiento falló (None si OK)
    """
    resultados = []
    total = len(ground_truth)

    print("\n" + "=" * 60)
    print("VALIDACIÓN — Comparación contra ground truth")
    print("=" * 60)

    for i, registro in enumerate(ground_truth):
        print(f"\n--- Registro {i + 1}/{total} ---")
        print(f"  Polígono: {registro['geojson']}")
        print(f"  Período: {registro['fecha_inicio']} a {registro['fecha_fin']}")
        print(f"  Práctica real: {registro['practica_real']}")
        print(f"  Fuente: {registro['fuente']}")

        consistente_real = _practica_a_consistente(registro['practica_real'])

        try:
            resultado_gee = procesar_lote(
                geojson_path=registro['geojson'],
                fecha_inicio=registro['fecha_inicio'],
                fecha_fin=registro['fecha_fin'],
                proyecto_gee=proyecto_gee,
            )
            consistente_predicho = resultado_gee['diagnostico']['consistente']
            error = None

        except Exception as e:
            print(f"  ERROR en procesamiento: {e}")
            # Si falla, marcamos como incorrecto para no inflar métricas
            consistente_predicho = not consistente_real
            error = str(e)

        correcto = (consistente_real == consistente_predicho)
        prediccion = (
            'siembra_directa' if consistente_predicho
            else 'no_siembra_directa'
        )

        estado = "CORRECTO" if correcto else "INCORRECTO"
        print(f"  Resultado: {estado} "
              f"(real={consistente_real}, predicho={consistente_predicho})")

        resultados.append({
            'practica_real': registro['practica_real'],
            'prediccion': prediccion,
            'consistente_real': consistente_real,
            'consistente_predicho': consistente_predicho,
            'correcto': correcto,
            'error': error,
        })

    return resultados


# =============================================================================
# Cálculo de métricas
# =============================================================================

def _contar_clasificaciones(resultados: List[Dict]) -> Dict[str, int]:
    """Cuenta verdaderos/falsos positivos/negativos.

    Positivo = siembra_directa (consistente=True)
    Negativo = convencional/quema (consistente=False)
    """
    tp = 0  # verdadero positivo: real=SD, predicho=consistente
    fp = 0  # falso positivo: real=conv/quema, predicho=consistente
    tn = 0  # verdadero negativo: real=conv/quema, predicho=no consistente
    fn = 0  # falso negativo: real=SD, predicho=no consistente

    for r in resultados:
        real = r['consistente_real']
        pred = r['consistente_predicho']

        if real and pred:
            tp += 1
        elif real and not pred:
            fn += 1
        elif not real and pred:
            fp += 1
        else:
            tn += 1

    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def calcular_metricas_validacion(
    resultados: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calcula métricas de clasificación binaria usando solo stdlib.

    Métricas calculadas:
      - accuracy: (TP + TN) / total
      - precision: TP / (TP + FP)
      - recall (sensibilidad): TP / (TP + FN)
      - f1_score: 2 * (precision * recall) / (precision + recall)
      - matriz_confusion: {tp, fp, tn, fn}

    Args:
        resultados: lista de dicts generada por ejecutar_validacion().

    Returns:
        Dict con accuracy, precision, recall, f1_score, matriz_confusion,
        total, correctos y errores_procesamiento.
    """
    if not resultados:
        raise ValueError("La lista de resultados está vacía")

    conteo = _contar_clasificaciones(resultados)
    tp, fp, tn, fn = conteo['tp'], conteo['fp'], conteo['tn'], conteo['fn']
    total = tp + fp + tn + fn

    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Precision (evitar división por cero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall / Sensibilidad
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1-Score
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # Contar errores de procesamiento
    errores = sum(1 for r in resultados if r.get('error') is not None)

    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4),
        'matriz_confusion': conteo,
        'total': total,
        'correctos': tp + tn,
        'errores_procesamiento': errores,
    }


# =============================================================================
# Reporte de validación
# =============================================================================

def _formatear_porcentaje(valor: float) -> str:
    """Formatea un valor decimal como porcentaje con 1 decimal."""
    return f"{valor * 100:.1f}%"


def _linea(char: str = '-', largo: int = 60) -> str:
    """Genera una línea separadora."""
    return char * largo


def generar_reporte_validacion(
    metricas: Dict[str, Any],
    resultados: List[Dict[str, Any]],
    output_path: str = None
) -> None:
    """Genera un reporte de validación formateado.

    Incluye:
      - Resumen de métricas (accuracy, precision, recall, F1)
      - Comparación contra objetivo de accuracy >85%
      - Matriz de confusión
      - Detalle por lote
      - Errores de procesamiento (si los hubo)

    Args:
        metricas: dict generado por calcular_metricas_validacion().
        resultados: lista de dicts generada por ejecutar_validacion().
        output_path: ruta para guardar el reporte como texto (opcional).
    """
    lineas = []

    def agregar(texto: str = ''):
        lineas.append(texto)

    # Encabezado
    agregar(_linea('='))
    agregar('REPORTE DE VALIDACIÓN — AgTech MRV')
    agregar(f'Fecha: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    agregar(_linea('='))

    # Resumen
    agregar('')
    agregar('RESUMEN')
    agregar(_linea())
    agregar(f'  Total de lotes evaluados: {metricas["total"]}')
    agregar(f'  Clasificaciones correctas: {metricas["correctos"]}')
    agregar(f'  Errores de procesamiento: {metricas["errores_procesamiento"]}')

    # Métricas principales
    agregar('')
    agregar('MÉTRICAS DE CLASIFICACIÓN')
    agregar(_linea())
    agregar(f'  Accuracy:   {_formatear_porcentaje(metricas["accuracy"])}')
    agregar(f'  Precision:  {_formatear_porcentaje(metricas["precision"])}')
    agregar(f'  Recall:     {_formatear_porcentaje(metricas["recall"])}')
    agregar(f'  F1-Score:   {_formatear_porcentaje(metricas["f1_score"])}')

    # Comparación contra objetivo
    objetivo_accuracy = 0.85
    cumple = metricas['accuracy'] >= objetivo_accuracy
    estado_obj = 'CUMPLE' if cumple else 'NO CUMPLE'
    agregar('')
    agregar(f'  Objetivo accuracy: '
            f'>{_formatear_porcentaje(objetivo_accuracy)}')
    agregar(f'  Estado: {estado_obj} '
            f'({_formatear_porcentaje(metricas["accuracy"])} actual)')

    # Matriz de confusión
    mc = metricas['matriz_confusion']
    agregar('')
    agregar('MATRIZ DE CONFUSIÓN')
    agregar(_linea())
    agregar('                          Predicho')
    agregar('                    Positivo   Negativo')
    agregar(f'  Real Positivo  |    {mc["tp"]:>4}   |   {mc["fn"]:>4}   |'
            f'  (siembra_directa)')
    agregar(f'  Real Negativo  |    {mc["fp"]:>4}   |   {mc["tn"]:>4}   |'
            f'  (convencional/quema)')
    agregar('')
    agregar(f'  TP={mc["tp"]}  FP={mc["fp"]}  '
            f'FN={mc["fn"]}  TN={mc["tn"]}')

    # Detalle por lote
    agregar('')
    agregar('DETALLE POR LOTE')
    agregar(_linea())

    for i, r in enumerate(resultados):
        estado = 'OK' if r['correcto'] else 'ERROR'
        agregar(f'  [{i + 1}] {estado} | '
                f'Real: {r["practica_real"]:<18} | '
                f'Predicción: {r["prediccion"]}')
        if r.get('error'):
            agregar(f'       Error de procesamiento: {r["error"]}')

    # Pie de reporte
    agregar('')
    agregar(_linea('='))
    agregar('Clasificación binaria: siembra_directa = positivo (consistente=True)')
    agregar('                       convencional/quema = negativo (consistente=False)')
    agregar(_linea('='))

    # Imprimir a consola
    reporte_texto = '\n'.join(lineas)
    print(reporte_texto)

    # Guardar a archivo si se pidió
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(reporte_texto)
        print(f'\nReporte guardado en: {output_path}')


# =============================================================================
# Ejecución principal
# =============================================================================

def main():
    """Punto de entrada: ejecuta validación completa desde la línea de comandos.

    Uso:
        python validacion.py <ruta_ground_truth.json> [proyecto_gee] [output.txt]

    Argumentos:
        1. Ruta al archivo JSON de ground truth (obligatorio)
        2. ID del proyecto GEE (opcional)
        3. Ruta para guardar el reporte como texto (opcional)
    """
    if len(sys.argv) < 2:
        print("Uso: python validacion.py <ground_truth.json> "
              "[proyecto_gee] [reporte_output.txt]")
        print("\nEjemplo:")
        print("  python validacion.py ground_truth/ejemplo.json")
        print("  python validacion.py ground_truth/ejemplo.json mi-proyecto-gee")
        print("  python validacion.py ground_truth/ejemplo.json "
              "mi-proyecto-gee reporte.txt")
        sys.exit(1)

    path_gt = sys.argv[1]
    proyecto_gee = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    # 1. Cargar ground truth
    ground_truth = cargar_ground_truth(path_gt)

    # 2. Ejecutar validación contra GEE
    resultados = ejecutar_validacion(ground_truth, proyecto_gee)

    # 3. Calcular métricas
    metricas = calcular_metricas_validacion(resultados)

    # 4. Generar reporte
    generar_reporte_validacion(metricas, resultados, output_path)


if __name__ == '__main__':
    main()
