"""
gee_processor.py — Núcleo de procesamiento satelital con Google Earth Engine

Procesa un lote agrícola pampeano y genera evidencia satelital de prácticas
regenerativas (siembra directa, cobertura vegetal, no labranza) para
certificación ARC/ASC de AAPRESID.

Enfoque: métricas RELATIVAS (lote vs departamento), no umbrales absolutos.
Esto permite auto-calibración sin intervención de un agrónomo.

Datos: Sentinel-2 L2A (óptico) + Sentinel-1 SAR GRD (radar)
"""

import ee
import json
import datetime
from typing import Dict, List, Any


# =============================================================================
# Inicialización
# =============================================================================

def inicializar_gee(proyecto=None):
    """Inicializa la conexión con Google Earth Engine.

    Requiere autenticación previa con `earthengine authenticate`.

    Args:
        proyecto: ID del proyecto GEE (opcional, usa el default si no se pasa).
    """
    try:
        if proyecto:
            ee.Initialize(project=proyecto)
        else:
            ee.Initialize()
        print("GEE inicializado correctamente")
    except Exception as e:
        raise RuntimeError(
            "No se pudo inicializar GEE. "
            "Ejecutá 'earthengine authenticate' primero. "
            f"Error: {e}"
        )


def cargar_poligono(geojson_path: str) -> ee.Geometry:
    """Carga un polígono desde un archivo GeoJSON.

    Soporta Feature, FeatureCollection y Geometry directa.
    """
    with open(geojson_path, 'r') as f:
        geojson = json.load(f)

    if geojson['type'] == 'FeatureCollection':
        geometria = geojson['features'][0]['geometry']
    elif geojson['type'] == 'Feature':
        geometria = geojson['geometry']
    else:
        geometria = geojson

    return ee.Geometry(geometria)


# =============================================================================
# Máscara de nubes
# =============================================================================

def mascara_nubes_s2(imagen: ee.Image) -> ee.Image:
    """Aplica máscara de nubes a imagen Sentinel-2 usando banda SCL.

    Conserva píxeles clasificados como:
      4 = vegetación, 5 = suelo desnudo, 6 = agua.
    Descarta nubes, sombras, cirrus y nieve.
    """
    scl = imagen.select('SCL')
    mascara = scl.gte(4).And(scl.lte(6))
    return imagen.updateMask(mascara)


# =============================================================================
# Carga de colecciones satelitales
# =============================================================================

def cargar_sentinel2(
    poligono: ee.Geometry,
    fecha_inicio: str,
    fecha_fin: str
) -> ee.ImageCollection:
    """Carga Sentinel-2 L2A filtrada por polígono, fechas y nubosidad.

    Filtra escenas con más de 30% de nubes y aplica máscara pixel a pixel.
    """
    coleccion = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(poligono)
        .filterDate(fecha_inicio, fecha_fin)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        .map(mascara_nubes_s2)
    )

    cantidad = coleccion.size().getInfo()
    if cantidad == 0:
        raise ValueError(
            f"Sin imágenes Sentinel-2 para {fecha_inicio} / {fecha_fin}. "
            "Verificá las fechas y el polígono."
        )

    print(f"  Sentinel-2: {cantidad} imágenes")
    return coleccion


def cargar_sentinel1(
    poligono: ee.Geometry,
    fecha_inicio: str,
    fecha_fin: str
):
    """Carga Sentinel-1 SAR GRD en modo IW, polarización VV+VH.

    Returns:
        ee.ImageCollection o None si no hay datos disponibles.
        Sentinel-1B falló en 2021, la cobertura en Sudamérica es parcial.
    """
    coleccion = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(poligono)
        .filterDate(fecha_inicio, fecha_fin)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
    )

    cantidad = coleccion.size().getInfo()
    if cantidad == 0:
        print("  Sentinel-1: sin datos disponibles para esta zona/período")
        return None

    print(f"  Sentinel-1: {cantidad} imágenes")
    return coleccion


# =============================================================================
# Índices espectrales — Sentinel-2
# =============================================================================

def calcular_ndvi(imagen: ee.Image) -> ee.Image:
    """NDVI = (NIR - Red) / (NIR + Red). Sentinel-2: B8, B4."""
    ndvi = imagen.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return imagen.addBands(ndvi)


def calcular_evi(imagen: ee.Image) -> ee.Image:
    """EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 10000).

    El +10000 compensa que S2 L2A tiene reflectancias en escala 0-10000.
    """
    evi = imagen.expression(
        '2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 10000.0)',
        {
            'NIR': imagen.select('B8'),
            'RED': imagen.select('B4'),
            'BLUE': imagen.select('B2'),
        }
    ).rename('EVI')
    return imagen.addBands(evi)


def calcular_bsi(imagen: ee.Image) -> ee.Image:
    """BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue)).

    Valores altos = suelo desnudo expuesto (posible labranza).
    """
    bsi = imagen.expression(
        '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
        {
            'SWIR1': imagen.select('B11'),
            'RED': imagen.select('B4'),
            'NIR': imagen.select('B8'),
            'BLUE': imagen.select('B2'),
        }
    ).rename('BSI')
    return imagen.addBands(bsi)


def calcular_nbr(imagen: ee.Image) -> ee.Image:
    """NBR = (NIR - SWIR2) / (NIR + SWIR2). Sentinel-2: B8, B12.

    Valores bajos o negativos = posible quema de rastrojo.
    """
    nbr = imagen.normalizedDifference(['B8', 'B12']).rename('NBR')
    return imagen.addBands(nbr)


# =============================================================================
# Índice SAR — Sentinel-1
# =============================================================================

def calcular_rvi(imagen: ee.Image) -> ee.Image:
    """RVI = 4 * VH_lin / (VV_lin + VH_lin).

    Sentinel-1 GRD viene en dB → convertimos a escala lineal.
    RVI bajo (~0-0.3) = suelo liso/labrado.
    RVI alto (~0.6-1.0) = superficie rugosa/no labrada con vegetación.
    """
    vv_lin = ee.Image(10).pow(imagen.select('VV').divide(10))
    vh_lin = ee.Image(10).pow(imagen.select('VH').divide(10))
    rvi = vh_lin.multiply(4).divide(vv_lin.add(vh_lin)).rename('RVI')
    return imagen.addBands(rvi)


# =============================================================================
# Serie temporal NDVI
# =============================================================================

def serie_temporal_ndvi(
    coleccion: ee.ImageCollection,
    poligono: ee.Geometry
) -> List[Dict]:
    """Genera serie temporal de NDVI promedio del lote.

    Returns:
        Lista de dicts con 'fecha' y 'ndvi', ordenada cronológicamente.
    """
    col_ndvi = coleccion.map(calcular_ndvi)

    def extraer_ndvi(imagen):
        stats = imagen.select('NDVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=poligono,
            scale=10,
            maxPixels=1e9,
        )
        return ee.Feature(None, {
            'fecha': imagen.date().format('YYYY-MM-dd'),
            'ndvi': stats.get('NDVI'),
        })

    serie = col_ndvi.map(extraer_ndvi).getInfo()

    # Filtrar nulos y ordenar
    resultado = []
    for feature in serie['features']:
        props = feature['properties']
        if props.get('ndvi') is not None:
            resultado.append({
                'fecha': props['fecha'],
                'ndvi': round(props['ndvi'], 4),
            })

    resultado.sort(key=lambda x: x['fecha'])
    print(f"  Serie NDVI: {len(resultado)} puntos temporales")
    return resultado


# =============================================================================
# Detección de patrones temporales (auto-calibrado)
# =============================================================================

def detectar_caidas_abruptas(
    serie_ndvi: List[Dict],
    umbral_caida: float = 0.60
) -> List[Dict]:
    """Detecta caídas abruptas de NDVI que sugieren labranza convencional.

    En siembra directa las transiciones son graduales (rastrojo mantiene NDVI).
    Una caída relativa > 60% entre observaciones consecutivas es firma de
    remoción de cobertura.

    Este método es auto-calibrado: usa cambio relativo, no valores absolutos.

    Args:
        serie_ndvi: Serie temporal de NDVI del lote.
        umbral_caida: Fracción de caída relativa para considerar abrupta (0.60 = 60%).

    Returns:
        Lista de eventos de caída detectados.
    """
    caidas = []
    for i in range(1, len(serie_ndvi)):
        ndvi_prev = serie_ndvi[i - 1]['ndvi']
        ndvi_curr = serie_ndvi[i]['ndvi']

        # Solo evaluar caídas desde valores con vegetación significativa
        if ndvi_prev > 0.20:
            cambio_rel = (ndvi_prev - ndvi_curr) / ndvi_prev
            if cambio_rel > umbral_caida:
                caidas.append({
                    'fecha': serie_ndvi[i]['fecha'],
                    'ndvi_antes': ndvi_prev,
                    'ndvi_despues': ndvi_curr,
                    'cambio_pct': round(cambio_rel * 100, 1),
                })

    if caidas:
        print(f"  ⚠ {len(caidas)} caída(s) abrupta(s) detectada(s)")
    else:
        print(f"  Sin caídas abruptas — perfil temporal consistente con SD")

    return caidas


# =============================================================================
# Serie temporal NBR — detección de eventos de quema
# =============================================================================

def serie_temporal_nbr(
    coleccion: ee.ImageCollection,
    poligono: ee.Geometry
) -> List[Dict]:
    """Genera serie temporal de NBR promedio del lote.

    Mismo patrón que serie_temporal_ndvi(). Permite identificar la fecha
    exacta de eventos de quema (NBR negativo).

    Returns:
        Lista de dicts con 'fecha' y 'nbr', ordenada cronológicamente.
    """
    col_nbr = coleccion.map(calcular_nbr)

    def extraer_nbr(imagen):
        stats = imagen.select('NBR').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=poligono,
            scale=10,
            maxPixels=1e9,
        )
        return ee.Feature(None, {
            'fecha': imagen.date().format('YYYY-MM-dd'),
            'nbr': stats.get('NBR'),
        })

    serie = col_nbr.map(extraer_nbr).getInfo()

    resultado = []
    for feature in serie['features']:
        props = feature['properties']
        if props.get('nbr') is not None:
            resultado.append({
                'fecha': props['fecha'],
                'nbr': round(props['nbr'], 4),
            })

    resultado.sort(key=lambda x: x['fecha'])
    print(f"  Serie NBR: {len(resultado)} puntos temporales")
    return resultado


def _clasificar_severidad_nbr(nbr: float) -> str:
    """Clasifica severidad de quema por valor NBR puntual."""
    if nbr < -0.20:
        return 'severa'
    elif nbr < -0.10:
        return 'moderada'
    return 'leve'


def detectar_eventos_quema(serie_nbr: List[Dict]) -> List[Dict]:
    """Detecta eventos de quema en la serie temporal NBR.

    Agrupa observaciones consecutivas con NBR negativo como un solo evento,
    quedándose con la fecha y valor del NBR más bajo (pico del evento).
    Esto evita contar la "cola" de un evento como múltiples quemas.

    Returns:
        Lista de eventos únicos con fecha, valor NBR y severidad.
    """
    if not serie_nbr:
        return []

    # Identificar grupos consecutivos de NBR negativo
    eventos = []
    grupo_actual = []

    for punto in serie_nbr:
        if punto['nbr'] < 0:
            grupo_actual.append(punto)
        else:
            if grupo_actual:
                # Tomar el punto más bajo del grupo como el evento
                peor = min(grupo_actual, key=lambda p: p['nbr'])
                eventos.append({
                    'fecha': peor['fecha'],
                    'nbr': peor['nbr'],
                    'severidad': _clasificar_severidad_nbr(peor['nbr']),
                    'duracion_obs': len(grupo_actual),
                })
                grupo_actual = []

    # Cerrar último grupo si existe
    if grupo_actual:
        peor = min(grupo_actual, key=lambda p: p['nbr'])
        eventos.append({
            'fecha': peor['fecha'],
            'nbr': peor['nbr'],
            'severidad': _clasificar_severidad_nbr(peor['nbr']),
            'duracion_obs': len(grupo_actual),
        })

    if eventos:
        resumen = ', '.join(
            f'{e["fecha"]} ({e["severidad"]}, {e["duracion_obs"]} obs)'
            for e in eventos
        )
        print(f"  Eventos de quema: {resumen}")
    else:
        print(f"  Sin eventos de quema en serie NBR")

    return eventos


def calcular_dnbr(
    serie_nbr: List[Dict],
    eventos_quema: List[Dict]
) -> List[Dict]:
    """Calcula delta NBR (pre-quema − post-quema) para cada evento.

    Busca la última observación con NBR positivo antes del evento como
    referencia pre-quema, para evitar comparar contra observaciones que
    ya estaban afectadas por el mismo evento.

    Metodología USGS:
      - sin quema: dNBR < 0.10
      - baja: 0.10 - 0.27
      - moderada-baja: 0.27 - 0.44
      - moderada-alta: 0.44 - 0.66
      - alta: > 0.66

    Returns:
        Lista de eventos enriquecidos con dNBR y severidad USGS.
    """
    if not eventos_quema or len(serie_nbr) < 2:
        return []

    fechas_ordenadas = [p['fecha'] for p in serie_nbr]

    resultados = []
    for evento in eventos_quema:
        fecha_evento = evento['fecha']
        idx = fechas_ordenadas.index(fecha_evento) if fecha_evento in fechas_ordenadas else -1
        if idx < 1:
            continue

        # Buscar última observación con NBR >= 0 antes del evento
        nbr_pre = None
        for i in range(idx - 1, -1, -1):
            if serie_nbr[i]['nbr'] >= 0:
                nbr_pre = serie_nbr[i]['nbr']
                break

        if nbr_pre is None:
            continue

        nbr_post = evento['nbr']
        dnbr = nbr_pre - nbr_post

        # Clasificación USGS
        if dnbr >= 0.66:
            severidad_usgs = 'alta'
        elif dnbr >= 0.44:
            severidad_usgs = 'moderada-alta'
        elif dnbr >= 0.27:
            severidad_usgs = 'moderada-baja'
        elif dnbr >= 0.10:
            severidad_usgs = 'baja'
        else:
            severidad_usgs = 'sin quema'

        resultados.append({
            'fecha': fecha_evento,
            'nbr_pre': round(nbr_pre, 4),
            'nbr_post': round(nbr_post, 4),
            'dnbr': round(dnbr, 4),
            'severidad_usgs': severidad_usgs,
        })

    if resultados:
        for r in resultados:
            print(f"  dNBR {r['fecha']}: {r['dnbr']:.3f} ({r['severidad_usgs']})")

    return resultados


def evaluar_disponibilidad_datos(
    serie_ndvi: List[Dict],
    fecha_inicio: str,
    fecha_fin: str
) -> Dict[str, Any]:
    """Evalúa la disponibilidad de datos por mes.

    Reporta meses con datos insuficientes (< 2 observaciones claras),
    especialmente crítico en enero-febrero por nubosidad estival.

    Returns:
        Dict con conteo por mes y lista de meses con gaps.
    """
    conteo_mensual = {}
    for punto in serie_ndvi:
        mes = punto['fecha'][:7]  # 'YYYY-MM'
        conteo_mensual[mes] = conteo_mensual.get(mes, 0) + 1

    # Generar todos los meses del período
    inicio = datetime.date.fromisoformat(fecha_inicio)
    fin = datetime.date.fromisoformat(fecha_fin)
    meses_con_gaps = []
    mes_actual = inicio.replace(day=1)
    while mes_actual <= fin:
        clave = mes_actual.strftime('%Y-%m')
        obs = conteo_mensual.get(clave, 0)
        if obs < 2:
            meses_con_gaps.append(clave)
        # Avanzar al siguiente mes
        if mes_actual.month == 12:
            mes_actual = mes_actual.replace(year=mes_actual.year + 1, month=1)
        else:
            mes_actual = mes_actual.replace(month=mes_actual.month + 1)

    if meses_con_gaps:
        print(f"  Gaps de datos en: {', '.join(meses_con_gaps)}")

    return {
        'observaciones_por_mes': conteo_mensual,
        'meses_con_gaps': meses_con_gaps,
        'total_observaciones': len(serie_ndvi),
    }


# =============================================================================
# Métricas del lote
# =============================================================================

def calcular_metricas_lote(
    s2_col: ee.ImageCollection,
    s1_col: ee.ImageCollection,
    poligono: ee.Geometry
) -> Dict[str, float]:
    """Calcula todos los índices espectrales del lote para el período.

    Usa mediana temporal para obtener valores robustos (reduce ruido y nubes
    residuales). Para NDVI mínimo usa el reductor min (detecta el momento
    de menor cobertura).

    Returns:
        Dict con ndvi_medio, ndvi_min, evi_medio, bsi_medio, nbr_medio, rvi_medio.
    """
    # Índices ópticos sobre Sentinel-2
    s2_indices = (
        s2_col
        .map(calcular_ndvi)
        .map(calcular_evi)
        .map(calcular_bsi)
        .map(calcular_nbr)
    )

    mediana_s2 = s2_indices.median()
    stats_s2 = mediana_s2.select(['NDVI', 'EVI', 'BSI', 'NBR']).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poligono,
        scale=10,
        maxPixels=1e9,
    ).getInfo()

    # NDVI mínimo: mínimo por pixel → promedio del lote
    ndvi_min_img = s2_indices.select('NDVI').min()
    stats_ndvi_min = ndvi_min_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poligono,
        scale=10,
        maxPixels=1e9,
    ).getInfo()

    # NBR mínimo: detecta eventos puntuales de quema (NBR negativo = quema)
    nbr_min_img = s2_indices.select('NBR').min()
    stats_nbr_min = nbr_min_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poligono,
        scale=10,
        maxPixels=1e9,
    ).getInfo()

    # RVI desde Sentinel-1 SAR (puede no estar disponible)
    rvi_valor = None
    if s1_col is not None:
        s1_rvi = s1_col.map(calcular_rvi)
        mediana_s1 = s1_rvi.median()
        stats_s1 = mediana_s1.select('RVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=poligono,
            scale=10,
            maxPixels=1e9,
        ).getInfo()
        rvi_valor = stats_s1.get('RVI')

    def safe_round(val):
        return round(val, 4) if val is not None else 0.0

    metricas = {
        'ndvi_medio': safe_round(stats_s2.get('NDVI')),
        'evi_medio': safe_round(stats_s2.get('EVI')),
        'bsi_medio': safe_round(stats_s2.get('BSI')),
        'nbr_medio': safe_round(stats_s2.get('NBR')),
        'nbr_min': safe_round(stats_nbr_min.get('NBR')),
        'ndvi_min': safe_round(stats_ndvi_min.get('NDVI')),
        'rvi_medio': safe_round(rvi_valor),
        'rvi_disponible': rvi_valor is not None,
    }

    print(f"  NDVI medio: {metricas['ndvi_medio']}")
    print(f"  NDVI mín:   {metricas['ndvi_min']}")
    print(f"  RVI medio:  {metricas['rvi_medio']} {'' if metricas['rvi_disponible'] else '(sin datos SAR)'}")
    print(f"  BSI medio:  {metricas['bsi_medio']}")
    print(f"  NBR medio:  {metricas['nbr_medio']}")
    print(f"  NBR mín:    {metricas['nbr_min']}")
    return metricas


# =============================================================================
# Comparación zonal — departamento (auto-calibración)
# =============================================================================

def obtener_departamento(poligono: ee.Geometry) -> Dict[str, Any]:
    """Identifica el departamento (nivel 2 admin) que contiene al lote.

    Usa FAO GAUL para divisiones administrativas de Argentina.

    Returns:
        Dict con 'feature' (ee.Feature), 'nombre', 'provincia'.
    """
    departamentos = ee.FeatureCollection('FAO/GAUL/2015/level2')
    argentina = departamentos.filter(ee.Filter.eq('ADM0_NAME', 'Argentina'))
    centroide = poligono.centroid()
    depto = argentina.filterBounds(centroide).first()

    info = depto.getInfo()
    nombre = info['properties']['ADM2_NAME']
    provincia = info['properties']['ADM1_NAME']
    print(f"  Departamento: {nombre}, {provincia}")

    return {
        'feature': depto,
        'nombre': nombre,
        'provincia': provincia,
    }


def comparacion_departamental(
    poligono: ee.Geometry,
    fecha_inicio: str,
    fecha_fin: str,
    metricas_lote: Dict[str, float]
) -> Dict[str, Any]:
    """Calcula métricas del departamento y compara con el lote.

    Usa escala de 100m para eficiencia (el departamento es grande).
    Calcula el percentil del lote dentro del departamento.

    Returns:
        Dict con métricas departamentales, diferencias y percentil.
    """
    depto_info = obtener_departamento(poligono)
    depto_geom = depto_info['feature'].geometry()

    # NDVI departamental
    s2_depto = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(depto_geom)
        .filterDate(fecha_inicio, fecha_fin)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        .map(mascara_nubes_s2)
        .map(calcular_ndvi)
        .map(calcular_bsi)
        .map(calcular_nbr)
    )

    mediana_depto = s2_depto.median()

    # Media y desvío estándar del departamento
    stats_depto = mediana_depto.select(['NDVI', 'BSI', 'NBR']).reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True,
        ),
        geometry=depto_geom,
        scale=250,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()

    ndvi_depto_mean = stats_depto.get('NDVI_mean')
    ndvi_depto_std = stats_depto.get('NDVI_stdDev')
    bsi_depto_mean = stats_depto.get('BSI_mean')
    bsi_depto_std = stats_depto.get('BSI_stdDev')
    nbr_depto_mean = stats_depto.get('NBR_mean')

    # Calcular z-score del lote (cuántos desvíos sobre/bajo la media)
    if ndvi_depto_std and ndvi_depto_std > 0:
        z_score = (metricas_lote['ndvi_medio'] - ndvi_depto_mean) / ndvi_depto_std
    else:
        z_score = 0.0

    def safe_round(val):
        return round(val, 4) if val is not None else 0.0

    # z-score BSI (invertido: BSI alto es peor)
    if bsi_depto_std and bsi_depto_std > 0:
        bsi_z_score = (metricas_lote['bsi_medio'] - (bsi_depto_mean or 0)) / bsi_depto_std
    else:
        bsi_z_score = 0.0

    resultado = {
        'nombre_departamento': depto_info['nombre'],
        'provincia': depto_info['provincia'],
        'ndvi_departamental': safe_round(ndvi_depto_mean),
        'ndvi_depto_stddev': safe_round(ndvi_depto_std),
        'bsi_departamental': safe_round(bsi_depto_mean),
        'nbr_departamental': safe_round(nbr_depto_mean),
        'ndvi_z_score': round(z_score, 2),
        'bsi_z_score': round(bsi_z_score, 2),
        'ndvi_diferencia': round(
            metricas_lote['ndvi_medio'] - (ndvi_depto_mean or 0), 4
        ),
    }

    signo = '+' if resultado['ndvi_diferencia'] >= 0 else ''
    print(
        f"  NDVI depto: {resultado['ndvi_departamental']} "
        f"(lote: {signo}{resultado['ndvi_diferencia']}, "
        f"z-score: {resultado['ndvi_z_score']})"
    )
    print(
        f"  BSI depto:  {resultado['bsi_departamental']} "
        f"(lote: {metricas_lote['bsi_medio']}, z-score: {resultado['bsi_z_score']})"
    )
    print(f"  NBR depto:  {resultado['nbr_departamental']}")

    return resultado


# =============================================================================
# Imágenes para el PDF
# =============================================================================

def obtener_imagen_rgb(
    s2_col: ee.ImageCollection,
    poligono: ee.Geometry
) -> str:
    """URL de imagen RGB (color real) del lote. Mediana temporal."""
    rgb = s2_col.median().select(['B4', 'B3', 'B2'])
    url = rgb.getThumbURL({
        'region': poligono,
        'dimensions': 512,
        'format': 'png',
        'min': 0,
        'max': 3000,
    })
    print("  Imagen RGB generada")
    return url


def obtener_imagen_ndvi(
    s2_col: ee.ImageCollection,
    poligono: ee.Geometry
) -> str:
    """URL de imagen NDVI coloreada del lote. Paleta rojo-amarillo-verde."""
    ndvi = s2_col.map(calcular_ndvi).median().select('NDVI')
    url = ndvi.getThumbURL({
        'region': poligono,
        'dimensions': 512,
        'format': 'png',
        'min': -0.1,
        'max': 0.9,
        'palette': ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850'],
    })
    print("  Imagen NDVI generada")
    return url


def obtener_imagen_bsi(
    s2_col: ee.ImageCollection,
    poligono: ee.Geometry
) -> str:
    """URL de imagen BSI coloreada del lote. Paleta beige-marrón."""
    bsi = s2_col.map(calcular_bsi).median().select('BSI')
    url = bsi.getThumbURL({
        'region': poligono,
        'dimensions': 512,
        'format': 'png',
        'min': -0.3,
        'max': 0.3,
        'palette': ['1a9850', 'a6d96a', 'f5f5dc', 'd2b48c', 'a0522d', '8b4513'],
    })
    print("  Imagen BSI generada")
    return url


# =============================================================================
# Score de confianza (reemplaza revisión humana en casos claros)
# =============================================================================

def calcular_confianza(
    metricas: Dict[str, float],
    comparacion: Dict[str, Any],
    disponibilidad: Dict[str, Any],
    caidas: List[Dict]
) -> Dict[str, Any]:
    """Calcula score de confianza 0-100 para el diagnóstico.

    El score determina si el reporte puede salir sin revisión humana:
      > 80: alta confianza, no necesita revisión
      50-80: confianza media, revisión recomendada
      < 50: baja confianza, revisión obligatoria

    Componentes (25 pts cada uno):
      1. Disponibilidad de datos — ¿hay suficientes imágenes claras?
      2. Convergencia de indicadores — ¿SAR y óptico coinciden?
      3. Consistencia temporal — ¿hay caídas abruptas sospechosas?
      4. Posición zonal — ¿el lote está dentro del rango del departamento?
    """
    puntaje = 0
    detalle = []

    # 1. Disponibilidad de datos (0-25)
    # Puntaje base por cantidad de observaciones, penalización leve por gaps.
    # Con ≥15 obs la mediana es robusta; gaps solo importan con pocas obs.
    total_obs = disponibilidad['total_observaciones']
    gaps = len(disponibilidad['meses_con_gaps'])
    if total_obs >= 15:
        pts = 25 - gaps          # 15+ obs: robusto, -1 por gap
    elif total_obs >= 10:
        pts = 20 - (gaps * 2)    # 10-14 obs: bueno, -2 por gap
    elif total_obs >= 5:
        pts = 12 - (gaps * 2)    # 5-9 obs: aceptable
    else:
        pts = 5
    pts = max(0, min(25, pts))
    puntaje += pts
    detalle.append({
        'componente': 'Disponibilidad de datos',
        'puntaje': pts,
        'nota': f"{total_obs} obs, {gaps} meses con gaps",
    })

    # 2. Convergencia de indicadores (0-25)
    bsi_ok = metricas['bsi_medio'] < 0.15
    ndvi_ok = metricas['ndvi_medio'] > 0.3
    tiene_sar = metricas.get('rvi_disponible', False)

    if tiene_sar:
        # Con SAR: convergencia multi-sensor
        rvi_ok = metricas['rvi_medio'] > 0.4
        coinciden = sum([rvi_ok, bsi_ok, ndvi_ok])
        if coinciden == 3:
            pts = 25
        elif coinciden == 2:
            pts = 15
        elif coinciden == 1:
            pts = 8
        else:
            pts = 0

        # Penalizar contradicción directa SAR vs óptico
        if rvi_ok and not bsi_ok:
            pts = max(0, pts - 10)
            detalle.append({
                'componente': 'Convergencia SAR/óptico',
                'puntaje': pts,
                'nota': 'CONTRADICCIÓN: SAR indica no labranza pero BSI alto',
            })
        elif not rvi_ok and bsi_ok:
            pts = max(0, pts - 10)
            detalle.append({
                'componente': 'Convergencia SAR/óptico',
                'puntaje': pts,
                'nota': 'CONTRADICCIÓN: BSI bajo pero SAR indica posible labranza',
            })
        else:
            detalle.append({
                'componente': 'Convergencia SAR/óptico',
                'puntaje': pts,
                'nota': f"Indicadores {'convergentes' if coinciden >= 2 else 'débiles'}",
            })
    else:
        # Sin SAR: solo óptico, tope de 18/25 (penalización por falta de sensor)
        coinciden = sum([bsi_ok, ndvi_ok])
        if coinciden == 2:
            pts = 18
        elif coinciden == 1:
            pts = 10
        else:
            pts = 0
        detalle.append({
            'componente': 'Convergencia (solo óptico)',
            'puntaje': pts,
            'nota': f"Sin datos SAR — diagnóstico basado en NDVI+BSI (máx 18/25)",
        })

    puntaje += pts

    # 3. Consistencia temporal (0-25)
    if len(caidas) == 0:
        pts = 25
    elif len(caidas) == 1:
        pts = 10  # Una caída puede ser cosecha normal
    else:
        pts = 0   # Múltiples caídas abruptas = sospechoso
    detalle.append({
        'componente': 'Consistencia temporal',
        'puntaje': pts,
        'nota': f"{len(caidas)} caída(s) abrupta(s) detectada(s)",
    })
    puntaje += pts

    # 4. Posición zonal (0-25)
    z = comparacion.get('ndvi_z_score', 0)
    if z >= 0.5:
        pts = 25  # Muy por encima de la media
    elif z >= 0:
        pts = 20  # Sobre la media
    elif z >= -0.5:
        pts = 12  # Levemente bajo la media
    elif z >= -1.0:
        pts = 5   # Significativamente bajo
    else:
        pts = 0   # Muy por debajo
    detalle.append({
        'componente': 'Posición zonal',
        'puntaje': pts,
        'nota': f"z-score = {z:+.2f} vs departamento",
    })
    puntaje += pts

    # Categoría
    if puntaje > 80:
        categoria = 'ALTA'
        requiere_revision = False
    elif puntaje >= 50:
        categoria = 'MEDIA'
        requiere_revision = True
    else:
        categoria = 'BAJA'
        requiere_revision = True

    print(f"  Confianza: {puntaje}/100 ({categoria})")

    return {
        'score': puntaje,
        'categoria': categoria,
        'requiere_revision': requiere_revision,
        'detalle': detalle,
    }


# =============================================================================
# Diagnóstico de siembra directa
# =============================================================================

def diagnosticar_siembra_directa(
    metricas: Dict[str, float],
    comparacion: Dict[str, Any],
    serie_ndvi: List[Dict],
    caidas: List[Dict],
    eventos_quema: List[Dict],
    dnbr_eventos: List[Dict],
    confianza: Dict[str, Any],
    fecha_inicio: str,
    fecha_fin: str
) -> Dict[str, Any]:
    """Evalúa si las evidencias son consistentes con siembra directa.

    Usa tres preguntas convergentes (no criterios independientes):
      1. ¿Hubo labranza? → RVI + caídas abruptas + BSI
      2. ¿Hay cobertura permanente? → NDVI min relativo + posición zonal
      3. ¿Hubo quema? → NBR

    Returns:
        Dict con diagnóstico, criterios evaluados y conclusión textual.
    """
    criterios = []

    # --- Pregunta 1: ¿Hubo labranza? ---
    evidencia_labranza = []
    sin_labranza = True

    # RVI SAR (puede no estar disponible)
    if metricas.get('rvi_disponible', False):
        if metricas['rvi_medio'] > 0.4:
            evidencia_labranza.append('RVI indica superficie no labrada')
        else:
            evidencia_labranza.append('RVI sugiere posible labranza')
            sin_labranza = False
    else:
        evidencia_labranza.append('Sin datos SAR disponibles para esta zona')

    # Caídas abruptas de NDVI
    if len(caidas) == 0:
        evidencia_labranza.append('Sin caídas abruptas de cobertura')
    else:
        evidencia_labranza.append(
            f'{len(caidas)} caída(s) abrupta(s) de NDVI detectada(s)'
        )
        sin_labranza = False

    # BSI — comparación relativa contra departamento
    bsi_z = comparacion.get('bsi_z_score', 0)
    if bsi_z <= 1.0:
        evidencia_labranza.append(
            f'BSI dentro del rango departamental (z={bsi_z:+.2f})'
        )
    else:
        evidencia_labranza.append(
            f'BSI significativamente sobre la media departamental (z={bsi_z:+.2f})'
        )
        sin_labranza = False

    criterios.append({
        'pregunta': '¿Hubo labranza?',
        'resultado': 'NO' if sin_labranza else 'POSIBLE',
        'evidencia': evidencia_labranza,
    })

    # --- Pregunta 2: ¿Hay cobertura permanente? ---
    evidencia_cobertura = []
    buena_cobertura = True

    # NDVI mínimo vs departamento
    z = comparacion.get('ndvi_z_score', 0)
    if z >= 0:
        evidencia_cobertura.append(
            f'NDVI sobre la media departamental (z={z:+.2f})'
        )
    else:
        evidencia_cobertura.append(
            f'NDVI bajo la media departamental (z={z:+.2f})'
        )
        if z < -1.0:
            buena_cobertura = False

    # NDVI mínimo absoluto (único umbral absoluto, muy conservador)
    if metricas['ndvi_min'] > 0.10:
        evidencia_cobertura.append(
            f'NDVI mínimo {metricas["ndvi_min"]:.2f} — rastrojo presente'
        )
    else:
        evidencia_cobertura.append(
            f'NDVI mínimo {metricas["ndvi_min"]:.2f} — suelo posiblemente expuesto'
        )
        buena_cobertura = False

    criterios.append({
        'pregunta': '¿Hay cobertura permanente?',
        'resultado': 'SÍ' if buena_cobertura else 'INSUFICIENTE',
        'evidencia': evidencia_cobertura,
    })

    # --- Pregunta 3: ¿Hubo quema? ---
    # Usar eventos_quema (serie temporal NBR) para fechas exactas.
    nbr_depto = comparacion.get('nbr_departamental', 0)
    evidencia_quema = []

    # Crear lookup de dNBR por fecha
    dnbr_por_fecha = {d['fecha']: d for d in dnbr_eventos} if dnbr_eventos else {}

    if not eventos_quema:
        sin_quema = True
        evidencia_quema.append('Sin eventos de quema en serie temporal NBR')
    else:
        # Filtrar: solo considerar eventos con dNBR real (>= "baja" USGS)
        eventos_reales = []
        eventos_descartados = []
        for ev in eventos_quema:
            dnbr = dnbr_por_fecha.get(ev['fecha'])
            if dnbr and dnbr['severidad_usgs'] != 'sin quema':
                eventos_reales.append((ev, dnbr))
            elif dnbr:
                eventos_descartados.append((ev, dnbr))
            else:
                # Sin dNBR calculado, mantener como evento real
                eventos_reales.append((ev, None))

        # Evaluar si los eventos reales son significativos
        # Si todos son "baja" USGS y el NBR mediano está sobre el depto → no quema
        nbr_sobre_depto = (nbr_depto and metricas['nbr_medio'] >= nbr_depto - 0.05)
        todos_baja = all(
            d and d['severidad_usgs'] == 'baja' for _, d in eventos_reales
        ) if eventos_reales else False

        if not eventos_reales or (todos_baja and nbr_sobre_depto):
            sin_quema = True
            if not eventos_reales:
                evidencia_quema.append(
                    'NBR negativo puntual detectado pero dNBR indica sin quema (USGS)'
                )
            else:
                evidencia_quema.append(
                    'Eventos de dNBR baja con NBR mediano sobre el departamento — sin quema significativa'
                )
        else:
            sin_quema = False
            for ev, dnbr in eventos_reales:
                if dnbr:
                    evidencia_quema.append(
                        f'{ev["fecha"]}: NBR={ev["nbr"]:.2f}, '
                        f'dNBR={dnbr["dnbr"]:.2f} ({dnbr["severidad_usgs"]} USGS)'
                    )
                else:
                    evidencia_quema.append(
                        f'{ev["fecha"]}: NBR={ev["nbr"]:.2f} '
                        f'(severidad {ev["severidad"]})'
                    )

    # Contexto: comparar NBR mediano contra departamento
    if nbr_depto and nbr_depto > 0:
        diff_nbr = metricas['nbr_medio'] - nbr_depto
        if diff_nbr >= -0.05:
            evidencia_quema.append(
                f'NBR mediano ({metricas["nbr_medio"]:.2f}) en rango departamental ({nbr_depto:.2f})'
            )
            # Si NBR mediano sobre el depto y no hay eventos reales, confirmar NO
            if sin_quema is False and not eventos_reales:
                sin_quema = True
        else:
            evidencia_quema.append(
                f'NBR mediano ({metricas["nbr_medio"]:.2f}) por debajo del depto ({nbr_depto:.2f})'
            )

    criterios.append({
        'pregunta': '¿Hubo quema de rastrojo?',
        'resultado': 'NO' if sin_quema else 'POSIBLE',
        'evidencia': evidencia_quema,
    })

    # --- Conclusión global ---
    consistente = sin_labranza and buena_cobertura and sin_quema
    depto = comparacion['nombre_departamento']
    prov = comparacion['provincia']

    if consistente:
        conclusion = (
            f"Las imágenes satelitales del período {fecha_inicio} a {fecha_fin} "
            f"son consistentes con prácticas de siembra directa y cobertura "
            f"vegetal permanente en el lote analizado ({depto}, {prov}), "
            f"sin evidencia de labranza convencional ni quema de rastrojo."
        )
    else:
        problemas = []
        if not sin_labranza:
            problemas.append("posible labranza")
        if not buena_cobertura:
            problemas.append("cobertura insuficiente")
        if not sin_quema:
            problemas.append("posible quema")
        conclusion = (
            f"Las imágenes satelitales del período {fecha_inicio} a {fecha_fin} "
            f"presentan indicadores que requieren verificación adicional: "
            f"{', '.join(problemas)}. "
            f"Se recomienda revisión por profesional agronómico."
        )

    if confianza['requiere_revision']:
        conclusion += (
            f" Nota: confianza del análisis {confianza['categoria']} "
            f"({confianza['score']}/100)."
        )

    return {
        'consistente': consistente,
        'criterios': criterios,
        'conclusion': conclusion,
        'confianza': confianza,
    }


# =============================================================================
# Función principal
# =============================================================================

def procesar_lote(
    geojson_path: str,
    fecha_inicio: str,
    fecha_fin: str,
    proyecto_gee: str = None
) -> Dict[str, Any]:
    """Procesa un lote agrícola y devuelve todos los datos para el PDF.

    Args:
        geojson_path: Ruta al archivo GeoJSON del polígono.
        fecha_inicio: Inicio del análisis ('YYYY-MM-DD').
        fecha_fin: Fin del análisis ('YYYY-MM-DD').
        proyecto_gee: ID del proyecto GEE (opcional).

    Returns:
        Dict completo con métricas, serie temporal, comparación zonal,
        diagnóstico, URLs de imágenes y metadata del procesamiento.
    """
    print("=" * 60)
    print("AgTech MRV — Procesamiento de lote")
    print("=" * 60)

    # [1/8] Inicializar
    print("\n[1/8] Inicializando Google Earth Engine...")
    inicializar_gee(proyecto_gee)

    # [2/8] Cargar polígono
    print("\n[2/8] Cargando polígono...")
    poligono = cargar_poligono(geojson_path)
    superficie_m2 = poligono.area().getInfo()
    superficie_ha = round(superficie_m2 / 10000, 2)
    centroide = poligono.centroid().coordinates().getInfo()
    bbox_coords = poligono.bounds().coordinates().getInfo()[0]
    bbox = {
        'west': min(c[0] for c in bbox_coords),
        'east': max(c[0] for c in bbox_coords),
        'south': min(c[1] for c in bbox_coords),
        'north': max(c[1] for c in bbox_coords),
    }
    # Coordenadas del polígono original para overlay
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    if geojson_data['type'] == 'FeatureCollection':
        poligono_coords = geojson_data['features'][0]['geometry']['coordinates'][0]
    elif geojson_data['type'] == 'Feature':
        poligono_coords = geojson_data['geometry']['coordinates'][0]
    else:
        poligono_coords = geojson_data['coordinates'][0]

    print(f"  Superficie: {superficie_ha} ha")
    print(f"  Centroide: {centroide[1]:.4f}, {centroide[0]:.4f}")

    # [3/8] Cargar Sentinel-2
    print("\n[3/8] Cargando Sentinel-2...")
    s2_col = cargar_sentinel2(poligono, fecha_inicio, fecha_fin)

    # [4/8] Cargar Sentinel-1
    print("\n[4/8] Cargando Sentinel-1 SAR...")
    s1_col = cargar_sentinel1(poligono, fecha_inicio, fecha_fin)

    # [5/8] Calcular métricas
    print("\n[5/8] Calculando índices espectrales...")
    metricas = calcular_metricas_lote(s2_col, s1_col, poligono)

    # [6/8] Serie temporal + patrones
    print("\n[6/8] Analizando serie temporal...")
    serie_ndvi = serie_temporal_ndvi(s2_col, poligono)
    caidas = detectar_caidas_abruptas(serie_ndvi)
    disponibilidad = evaluar_disponibilidad_datos(
        serie_ndvi, fecha_inicio, fecha_fin
    )

    # Serie NBR + detección de eventos de quema
    print("  Analizando serie NBR...")
    serie_nbr = serie_temporal_nbr(s2_col, poligono)
    eventos_quema = detectar_eventos_quema(serie_nbr)
    dnbr_eventos = calcular_dnbr(serie_nbr, eventos_quema)

    # [7/8] Comparación departamental
    print("\n[7/8] Calculando referencia departamental...")
    comparacion = comparacion_departamental(
        poligono, fecha_inicio, fecha_fin, metricas
    )

    # [8/8] Imágenes + diagnóstico
    print("\n[8/8] Generando imágenes y diagnóstico...")
    url_rgb = obtener_imagen_rgb(s2_col, poligono)
    url_ndvi = obtener_imagen_ndvi(s2_col, poligono)
    url_bsi = obtener_imagen_bsi(s2_col, poligono)

    confianza = calcular_confianza(
        metricas, comparacion, disponibilidad, caidas
    )
    diagnostico = diagnosticar_siembra_directa(
        metricas, comparacion, serie_ndvi, caidas, eventos_quema,
        dnbr_eventos, confianza, fecha_inicio, fecha_fin
    )

    # Resumen final
    estado = 'CONSISTENTE' if diagnostico['consistente'] else 'REQUIERE VERIFICACIÓN'
    print("\n" + "=" * 60)
    print(f"RESULTADO: {estado}")
    print(f"Confianza: {confianza['score']}/100 ({confianza['categoria']})")
    print("=" * 60)

    return {
        'metricas': metricas,
        'serie_ndvi': serie_ndvi,
        'serie_nbr': serie_nbr,
        'caidas_abruptas': caidas,
        'eventos_quema': eventos_quema,
        'dnbr_eventos': dnbr_eventos,
        'disponibilidad_datos': disponibilidad,
        'comparacion_zonal': comparacion,
        'diagnostico': diagnostico,
        'imagenes': {
            'rgb_url': url_rgb,
            'ndvi_url': url_ndvi,
            'bsi_url': url_bsi,
            'bbox': bbox,
        },
        'metadata': {
            'superficie_ha': superficie_ha,
            'centroide_lat': round(centroide[1], 4),
            'centroide_lon': round(centroide[0], 4),
            'poligono_coords': poligono_coords,
            'fecha_inicio': fecha_inicio,
            'fecha_fin': fecha_fin,
            'fecha_procesamiento': datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M'
            ),
        },
    }


# =============================================================================
# Ejecución directa
# =============================================================================

if __name__ == '__main__':
    import sys

    geojson = (
        sys.argv[1] if len(sys.argv) > 1
        else 'sample_polygons/lote_santa_fe.geojson'
    )
    inicio = sys.argv[2] if len(sys.argv) > 2 else '2023-10-01'
    fin = sys.argv[3] if len(sys.argv) > 3 else '2024-04-30'

    resultado = procesar_lote(geojson, inicio, fin)

    print("\n--- Métricas ---")
    for k, v in resultado['metricas'].items():
        print(f"  {k}: {v}")

    print(f"\n--- Diagnóstico ---")
    print(f"  {resultado['diagnostico']['conclusion']}")

    print(f"\n--- Imágenes ---")
    print(f"  RGB:  {resultado['imagenes']['rgb_url'][:80]}...")
    print(f"  NDVI: {resultado['imagenes']['ndvi_url'][:80]}...")
