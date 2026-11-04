"""
run_pipeline.py — Ejecuta el pipeline completo: GEE → PDF

Uso:
    python run_pipeline.py
    python run_pipeline.py lote.geojson 2023-10-01 2024-04-30
    python run_pipeline.py lote.geojson 2023-10-01 2024-04-30 --proyecto mi-proyecto-gee
    python run_pipeline.py lote.geojson --multi 2022-10-01,2023-04-30 2023-10-01,2024-04-30 --proyecto mi-proyecto-gee
"""

import sys
import json
import os

from gee_processor import procesar_lote
from pdf_generator import generar_pdf


def comparar_campanas(resultados: list) -> dict:
    """Compara métricas entre campañas para evaluar tendencia.

    Args:
        resultados: Lista de dicts con 'periodo' y 'datos' (output de procesar_lote).

    Returns:
        Dict con tendencias y resumen comparativo.
    """
    ndvi_por_campana = []
    bsi_por_campana = []
    consistente_todas = True

    for r in resultados:
        d = r['datos']
        ndvi_por_campana.append(d['metricas']['ndvi_medio'])
        bsi_por_campana.append(d['metricas']['bsi_medio'])
        if not d['diagnostico']['consistente']:
            consistente_todas = False

    # Tendencia NDVI
    if len(ndvi_por_campana) >= 2:
        diff = ndvi_por_campana[-1] - ndvi_por_campana[0]
        if abs(diff) < 0.03:
            ndvi_tendencia = 'estable'
        elif diff > 0:
            ndvi_tendencia = 'mejorando'
        else:
            ndvi_tendencia = 'declinando'
    else:
        ndvi_tendencia = 'sin datos suficientes'

    return {
        'ndvi_por_campana': ndvi_por_campana,
        'bsi_por_campana': bsi_por_campana,
        'ndvi_tendencia': ndvi_tendencia,
        'consistente_todas': consistente_todas,
        'total_campanas': len(resultados),
    }


def main():
    # Extraer proyecto
    proyecto = None
    if '--proyecto' in sys.argv:
        idx = sys.argv.index('--proyecto')
        proyecto = sys.argv[idx + 1]

    # Detectar modo multi-campaña
    modo_multi = '--multi' in sys.argv

    geojson = (
        sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-')
        else 'sample_polygons/lote_santa_fe.geojson'
    )
    nombre_base = os.path.splitext(os.path.basename(geojson))[0]
    os.makedirs('sample_outputs', exist_ok=True)

    if modo_multi:
        # Modo multi-campaña: extraer rangos de fechas
        idx_multi = sys.argv.index('--multi')
        rangos = []
        for arg in sys.argv[idx_multi + 1:]:
            if arg.startswith('-'):
                break
            if ',' in arg:
                inicio, fin = arg.split(',')
                rangos.append((inicio.strip(), fin.strip()))

        if not rangos:
            print("Error: --multi requiere rangos de fechas (ej: 2022-10-01,2023-04-30)")
            sys.exit(1)

        print(f"\nModo multi-campaña: {len(rangos)} campañas")
        resultados = []
        for i, (inicio, fin) in enumerate(rangos):
            print(f"\n{'='*60}")
            print(f"CAMPAÑA {i+1}/{len(rangos)}: {inicio} a {fin}")
            print(f"{'='*60}")

            datos = procesar_lote(geojson, inicio, fin, proyecto)
            periodo = f"{inicio} a {fin}"
            resultados.append({'periodo': periodo, 'datos': datos})

            # Guardar JSON individual
            output_json = f'sample_outputs/{nombre_base}_{inicio}_{fin}_datos.json'
            with open(output_json, 'w') as f:
                json.dump(datos, f, indent=2, ensure_ascii=False)

        # Comparar campañas
        comparacion = comparar_campanas(resultados)

        # Guardar datos multi-campaña
        datos_multi = {
            'tipo': 'multicampana',
            'campanas': resultados,
            'comparacion_campanas': comparacion,
        }
        output_json_multi = f'sample_outputs/{nombre_base}_multicampana.json'
        with open(output_json_multi, 'w') as f:
            json.dump(datos_multi, f, indent=2, ensure_ascii=False)

        # Generar PDF (usa la última campaña como principal + comparación)
        output_pdf = f'sample_outputs/{nombre_base}_multicampana.pdf'
        generar_pdf(
            resultados[-1]['datos'], output_pdf,
            datos_multi=datos_multi,
        )

        print(f"\n{'='*60}")
        print("MULTI-CAMPAÑA COMPLETO")
        print(f"  Lote: {geojson}")
        print(f"  Campañas: {len(rangos)}")
        print(f"  Tendencia NDVI: {comparacion['ndvi_tendencia']}")
        print(f"  Consistente en todas: {'SÍ' if comparacion['consistente_todas'] else 'NO'}")
        print(f"  PDF: {output_pdf}")
        print(f"{'='*60}")

    else:
        # Modo single campaign
        fecha_inicio = sys.argv[2] if len(sys.argv) > 2 else '2023-10-01'
        fecha_fin = sys.argv[3] if len(sys.argv) > 3 else '2024-04-30'

        datos = procesar_lote(geojson, fecha_inicio, fecha_fin, proyecto)

        output_json = f'sample_outputs/{nombre_base}_datos.json'
        with open(output_json, 'w') as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        print(f"\nDatos guardados en: {output_json}")

        output_pdf = f'sample_outputs/{nombre_base}_reporte.pdf'
        generar_pdf(datos, output_pdf)
        print(f"Reporte PDF en: {output_pdf}")

        diag = datos['diagnostico']
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETO")
        print(f"  Lote: {geojson}")
        print(f"  Período: {fecha_inicio} a {fecha_fin}")
        print(f"  Resultado: {'CONSISTENTE' if diag['consistente'] else 'REQUIERE VERIFICACIÓN'}")
        print(f"  Confianza: {diag['confianza']['score']}/100")
        print(f"  PDF: {output_pdf}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
