"""
pdf_generator.py — Genera reporte PDF de evidencia satelital para ARC/AAPRESID

Toma el output de gee_processor.procesar_lote() y genera un PDF profesional
de 2 páginas apto para adjuntar al expediente de certificación.

Página 1: Mapa RGB + mapa NDVI del lote + encabezado
Página 2: Gráfico temporal NDVI + tabla de métricas + diagnóstico
"""

import io
import os
import datetime
import urllib.request
from typing import Dict, Any

import matplotlib
matplotlib.use('Agg')  # Backend sin display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.lib import colors


# =============================================================================
# Colores del reporte
# =============================================================================

COLOR_PRIMARIO = HexColor('#1a5632')    # Verde oscuro
COLOR_SECUNDARIO = HexColor('#2d8a4e')  # Verde medio
COLOR_FONDO = HexColor('#f5f9f6')       # Verde muy claro
COLOR_TEXTO = HexColor('#1a1a1a')       # Casi negro
COLOR_GRIS = HexColor('#666666')
COLOR_CUMPLE = HexColor('#2d8a4e')
COLOR_NO_CUMPLE = HexColor('#c0392b')


# =============================================================================
# Estilos
# =============================================================================

def crear_estilos():
    """Crea los estilos de texto para el PDF."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'Titulo',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=COLOR_PRIMARIO,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        'Subtitulo',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        textColor=COLOR_GRIS,
        spaceAfter=12,
    ))

    styles.add(ParagraphStyle(
        'SeccionTitulo',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=COLOR_PRIMARIO,
        spaceBefore=12,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        'CuerpoTexto',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=COLOR_TEXTO,
        leading=13,
        alignment=TA_JUSTIFY,
    ))

    styles.add(ParagraphStyle(
        'Conclusion',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=COLOR_TEXTO,
        leading=12,
        alignment=TA_JUSTIFY,
        spaceBefore=4,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        'Nota',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        textColor=COLOR_GRIS,
        leading=9,
    ))

    styles.add(ParagraphStyle(
        'MetricaLabel',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=COLOR_GRIS,
    ))

    styles.add(ParagraphStyle(
        'MetricaValor',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=COLOR_PRIMARIO,
    ))

    return styles


# =============================================================================
# Descarga de imágenes
# =============================================================================

def descargar_imagen(url: str, nombre: str = 'imagen') -> bytes:
    """Descarga imagen desde URL de GEE y retorna bytes."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'AgTech-MRV/1.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            datos = response.read()
        print(f"  Descargada: {nombre} ({len(datos)} bytes)")
        return datos
    except Exception as e:
        print(f"  Error descargando {nombre}: {e}")
        return None


# =============================================================================
# Mapa de ubicación
# =============================================================================

# Silueta simplificada de Argentina (lon, lat)
_ARGENTINA_CONTORNO = [
    (-69.3, -22.1), (-66.2, -22.1), (-65.0, -22.8), (-64.3, -24.0),
    (-62.8, -25.5), (-60.5, -27.0), (-58.6, -27.3), (-57.6, -25.4),
    (-57.9, -22.1), (-61.0, -19.8), (-69.3, -22.1),  # norte
    (-69.3, -22.1), (-67.2, -24.5), (-66.0, -28.0), (-68.5, -29.5),
    (-69.9, -30.3), (-70.0, -32.0), (-70.0, -34.5), (-69.8, -36.2),
    (-71.0, -38.0), (-71.5, -40.0), (-71.8, -42.0), (-71.5, -44.0),
    (-72.0, -46.0), (-71.5, -48.0), (-70.5, -50.0), (-69.3, -51.0),
    (-68.5, -52.5), (-68.3, -54.8), (-65.2, -55.2), (-65.0, -52.0),
    (-66.5, -51.0), (-65.8, -47.0), (-65.5, -45.0), (-65.0, -42.0),
    (-63.6, -39.8), (-62.0, -38.7), (-60.0, -38.5), (-58.5, -37.5),
    (-57.5, -36.0), (-56.7, -35.0), (-58.3, -34.2), (-58.5, -33.5),
    (-59.0, -32.5), (-59.7, -31.0), (-60.0, -29.5), (-59.0, -28.0),
    (-58.6, -27.3),
]


def generar_mapa_ubicacion(lat: float, lon: float) -> bytes:
    """Genera mapa de Argentina con punto del lote marcado.

    Usa matplotlib puro con silueta hardcoded. Sin llamadas a GEE.

    Returns:
        Bytes PNG del mapa.
    """
    fig, ax = plt.subplots(figsize=(3, 4), dpi=120)

    # Dibujar silueta
    lons = [p[0] for p in _ARGENTINA_CONTORNO]
    lats = [p[1] for p in _ARGENTINA_CONTORNO]
    ax.fill(lons, lats, color='#e8e8e8', edgecolor='#999999', linewidth=0.8)

    # Marcar lote
    ax.plot(lon, lat, 'o', color='#c0392b', markersize=8, zorder=5)
    ax.plot(lon, lat, 'o', color='#e74c3c', markersize=5, zorder=6)

    ax.set_xlim(-76, -53)
    ax.set_ylim(-56, -20)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120,
                transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def coordenadas_a_pixeles(coords, bbox, img_width, img_height):
    """Convierte coordenadas lon/lat a píxeles de imagen.

    Args:
        coords: Lista de [lon, lat] del polígono.
        bbox: Dict con west, east, south, north.
        img_width: Ancho de la imagen en píxeles.
        img_height: Alto de la imagen en píxeles.

    Returns:
        Lista de tuplas (x, y) en píxeles.
    """
    lon_range = bbox['east'] - bbox['west']
    lat_range = bbox['north'] - bbox['south']
    pixeles = []
    for lon, lat in coords:
        x = int((lon - bbox['west']) / lon_range * img_width)
        y = int((bbox['north'] - lat) / lat_range * img_height)
        pixeles.append((x, y))
    return pixeles


def superponer_poligono(img_bytes, coordenadas, bbox, color='yellow', grosor=2):
    """Superpone el contorno del polígono sobre una imagen satelital.

    Args:
        img_bytes: Bytes PNG de la imagen.
        coordenadas: Lista de [lon, lat] del polígono.
        bbox: Dict con west, east, south, north.
        color: Color del contorno.
        grosor: Grosor de la línea en píxeles.

    Returns:
        Bytes PNG con el polígono superpuesto, o img_bytes original si falla.
    """
    try:
        from PIL import Image as PILImage, ImageDraw

        img = PILImage.open(io.BytesIO(img_bytes))
        pixeles = coordenadas_a_pixeles(
            coordenadas, bbox, img.width, img.height
        )
        draw = ImageDraw.Draw(img)
        draw.polygon(pixeles, outline=color, width=grosor)

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"  Error superponiedo polígono: {e}")
        return img_bytes


# =============================================================================
# Gráfico NDVI temporal
# =============================================================================

def generar_grafico_confianza(confianza: Dict[str, Any]) -> bytes:
    """Genera gráfico de barras horizontales con los 4 componentes del score.

    Colores: verde >18pts, amarillo >10pts, rojo <=10pts.
    Returns:
        Bytes PNG del gráfico.
    """
    detalle = confianza['detalle']
    # Nombres cortos para las barras
    nombres = [d['componente'].split('(')[0].strip() for d in detalle]
    puntajes = [d['puntaje'] for d in detalle]
    colores = [
        '#2d8a4e' if p > 18 else '#e6a817' if p > 10 else '#c0392b'
        for p in puntajes
    ]

    fig, ax = plt.subplots(figsize=(8, 2.2), dpi=120)

    y_pos = range(len(nombres))
    bars = ax.barh(y_pos, puntajes, color=colores, height=0.6, edgecolor='white')

    # Etiquetas de puntaje dentro de las barras
    for bar, pts in zip(bars, puntajes):
        ax.text(
            bar.get_width() - 0.5, bar.get_y() + bar.get_height() / 2,
            f'{pts}/25', ha='right', va='center', fontsize=8,
            color='white', fontweight='bold',
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(nombres, fontsize=8)
    ax.set_xlim(0, 25)
    ax.set_xlabel('')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax.set_facecolor('#fafafa')

    # Score total
    total = confianza['score']
    cat = confianza['categoria']
    ax.set_title(
        f'Score de confianza: {total}/100 ({cat})',
        fontsize=10, fontweight='bold', color='#1a5632', loc='left',
    )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generar_grafico_ndvi(
    serie_ndvi: list,
    caidas: list,
    comparacion: Dict[str, Any],
    output_path: str = None
) -> bytes:
    """Genera gráfico de serie temporal NDVI con matplotlib.

    Incluye: curva NDVI, media departamental, y marcadores de caídas abruptas.
    Retorna bytes PNG del gráfico.
    """
    fechas = [
        datetime.datetime.strptime(p['fecha'], '%Y-%m-%d')
        for p in serie_ndvi
    ]
    valores = [p['ndvi'] for p in serie_ndvi]

    fig, ax = plt.subplots(figsize=(17, 7), dpi=150)

    # Curva NDVI del lote
    ax.plot(
        fechas, valores,
        color='#2d8a4e', linewidth=1.8, marker='o', markersize=3,
        label='NDVI del lote', zorder=3,
    )
    ax.fill_between(fechas, valores, alpha=0.15, color='#2d8a4e')

    # Línea media departamental
    ndvi_depto = comparacion.get('ndvi_departamental', 0)
    if ndvi_depto:
        ax.axhline(
            y=ndvi_depto, color='#e67e22', linestyle='--', linewidth=1.2,
            label=f'Media depto ({ndvi_depto:.2f})', zorder=2,
        )

    # Marcar caídas abruptas
    for caida in caidas:
        fecha_caida = datetime.datetime.strptime(caida['fecha'], '%Y-%m-%d')
        ax.axvline(
            x=fecha_caida, color='#c0392b', linestyle=':', linewidth=1,
            alpha=0.7, zorder=1,
        )
        ax.annotate(
            f'-{caida["cambio_pct"]:.0f}%',
            xy=(fecha_caida, caida['ndvi_despues']),
            xytext=(10, 15), textcoords='offset points',
            fontsize=7, color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8),
        )

    ax.set_ylabel('NDVI', fontsize=9, color='#333333')
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlim(min(fechas), max(fechas))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, fontsize=7)
    plt.yticks(fontsize=7)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_facecolor('#fafafa')
    fig.tight_layout()

    # Guardar a bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)

    if output_path:
        with open(output_path, 'wb') as f:
            f.write(buf.getvalue())
        buf.seek(0)

    return buf.getvalue()


# =============================================================================
# Tabla de métricas
# =============================================================================

def crear_tabla_metricas(
    metricas: Dict[str, float],
    comparacion: Dict[str, Any]
) -> Table:
    """Crea tabla con métricas del lote y comparación departamental."""
    datos = [
        ['Indicador', 'Valor del lote', 'Ref. departamental', 'Interpretación'],
        [
            'NDVI medio',
            f'{metricas["ndvi_medio"]:.3f}',
            f'{comparacion["ndvi_departamental"]:.3f}',
            'Mayor = más biomasa verde',
        ],
        [
            'NDVI mínimo (barbecho)',
            f'{metricas["ndvi_min"]:.3f}',
            '—',
            '> 0.10 indica rastrojo presente',
        ],
        [
            'EVI medio',
            f'{metricas["evi_medio"]:.3f}',
            '—',
            'Complementa NDVI en alta densidad',
        ],
        [
            'BSI (suelo desnudo)',
            f'{metricas["bsi_medio"]:.3f}',
            f'{comparacion.get("bsi_departamental", 0):.3f}',
            'Menor = menos suelo expuesto',
        ],
        [
            'NBR mediano',
            f'{metricas["nbr_medio"]:.3f}',
            f'{comparacion.get("nbr_departamental", 0):.3f}',
            'Similar al depto = normal',
        ],
        [
            'NBR mínimo (quema)',
            f'{metricas.get("nbr_min", 0):.3f}',
            '—',
            'Positivo = sin quema; negativo = posible evento' if metricas.get('nbr_min', 0) >= 0
            else 'Negativo indica posible evento de quema',
        ],
        [
            'RVI SAR (labranza)',
            f'{metricas["rvi_medio"]:.3f}' if metricas.get('rvi_disponible', True) else 'Sin datos',
            '—',
            '> 0.40 superficie no labrada' if metricas.get('rvi_disponible', True) else 'Sin cobertura SAR en esta zona',
        ],
        [
            'NDVI z-score vs depto',
            f'{comparacion["ndvi_z_score"]:+.2f}',
            '0.00',
            'Positivo = sobre la media zonal',
        ],
    ]

    tabla = Table(datos, colWidths=[4.2 * cm, 3 * cm, 3.2 * cm, 5.5 * cm])

    estilo = TableStyle([
        # Encabezado
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARIO),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        # Cuerpo
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (1, 1), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # Bordes
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_GRIS),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_FONDO]),
        # Padding
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ])

    tabla.setStyle(estilo)
    return tabla


# =============================================================================
# Tabla de diagnóstico por criterio
# =============================================================================

def crear_tabla_diagnostico(criterios: list) -> Table:
    """Crea tabla con los criterios de evaluación de siembra directa."""
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        'CeldaDiag', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8, leading=10,
    )
    cell_center = ParagraphStyle(
        'CeldaDiagCenter', parent=cell_style, alignment=TA_CENTER,
    )

    datos = [['Pregunta', 'Resultado', 'Evidencia']]

    for c in criterios:
        resultado = c['resultado']
        color_res = '#2d8a4e' if resultado in ('NO', 'SÍ') else '#c0392b'
        evidencia_texto = '; '.join(c['evidencia'])
        datos.append([
            Paragraph(c['pregunta'], cell_style),
            Paragraph(
                f'<font color="{color_res}"><b>{resultado}</b></font>',
                cell_center,
            ),
            Paragraph(evidencia_texto, cell_style),
        ])

    tabla = Table(datos, colWidths=[4 * cm, 2.5 * cm, 9.4 * cm])

    estilo = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARIO),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_GRIS),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_FONDO]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ])

    tabla.setStyle(estilo)
    return tabla


# =============================================================================
# Multi-campaña
# =============================================================================

def generar_grafico_ndvi_multicampana(campanas: list) -> bytes:
    """Genera gráfico NDVI con curvas superpuestas de múltiples campañas.

    Args:
        campanas: Lista de dicts con 'periodo' y 'datos'.

    Returns:
        Bytes PNG del gráfico.
    """
    colores = ['#2d8a4e', '#e67e22', '#3498db', '#9b59b6', '#e74c3c']
    fig, ax = plt.subplots(figsize=(17, 6), dpi=150)

    for i, camp in enumerate(campanas):
        serie = camp['datos']['serie_ndvi']
        fechas = [
            datetime.datetime.strptime(p['fecha'], '%Y-%m-%d')
            for p in serie
        ]
        valores = [p['ndvi'] for p in serie]
        color = colores[i % len(colores)]
        ax.plot(
            fechas, valores, color=color, linewidth=1.5,
            marker='o', markersize=3, label=camp['periodo'],
        )

    ax.set_ylabel('NDVI', fontsize=9, color='#333333')
    ax.set_ylim(-0.05, 1.0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, fontsize=7)
    plt.yticks(fontsize=7)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_facecolor('#fafafa')
    ax.set_title('Evolución NDVI — Comparación entre campañas',
                 fontsize=10, fontweight='bold', color='#1a5632', loc='left')
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def crear_tabla_comparacion_campanas(campanas: list, comparacion: dict) -> Table:
    """Crea tabla comparativa de métricas entre campañas."""
    datos_tabla = [['Métrica'] + [c['periodo'].split(' a ')[0][:7] for c in campanas]]

    # NDVI medio
    fila = ['NDVI medio']
    for c in campanas:
        fila.append(f'{c["datos"]["metricas"]["ndvi_medio"]:.3f}')
    datos_tabla.append(fila)

    # BSI medio
    fila = ['BSI medio']
    for c in campanas:
        fila.append(f'{c["datos"]["metricas"]["bsi_medio"]:.3f}')
    datos_tabla.append(fila)

    # Score de confianza
    fila = ['Confianza']
    for c in campanas:
        score = c['datos']['diagnostico']['confianza']['score']
        fila.append(f'{score}/100')
    datos_tabla.append(fila)

    # Resultado
    fila = ['Resultado']
    for c in campanas:
        ok = c['datos']['diagnostico']['consistente']
        fila.append('Consistente' if ok else 'Verificar')
    datos_tabla.append(fila)

    # Tendencia
    datos_tabla.append(['Tendencia NDVI', comparacion['ndvi_tendencia'], '', ''])

    col_w = 4 * cm
    n_cols = 1 + len(campanas)
    tabla = Table(datos_tabla, colWidths=[col_w] * n_cols)

    estilo = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a5632')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#666666')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f5f9f6')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ])
    tabla.setStyle(estilo)
    return tabla


# =============================================================================
# Generador principal del PDF
# =============================================================================

def generar_pdf(
    datos: Dict[str, Any],
    output_path: str = 'reporte_mrv.pdf',
    datos_multi: Dict[str, Any] = None
) -> str:
    """Genera el PDF completo de evidencia satelital.

    Args:
        datos: Output de gee_processor.procesar_lote() (campaña principal).
        output_path: Ruta donde guardar el PDF.
        datos_multi: Output multi-campaña (opcional). Si se pasa, agrega
            una página de comparación entre campañas.

    Returns:
        Ruta del PDF generado.
    """
    print("\n" + "=" * 60)
    print("Generando reporte PDF...")
    print("=" * 60)

    styles = crear_estilos()
    meta = datos['metadata']
    diag = datos['diagnostico']
    metricas = datos['metricas']
    comparacion = datos['comparacion_zonal']
    confianza = diag['confianza']

    # Descargar imágenes satelitales
    print("\n  Descargando imágenes...")
    img_rgb_bytes = descargar_imagen(datos['imagenes']['rgb_url'], 'RGB')
    img_ndvi_bytes = descargar_imagen(datos['imagenes']['ndvi_url'], 'NDVI')
    img_bsi_bytes = descargar_imagen(
        datos['imagenes'].get('bsi_url', ''), 'BSI'
    ) if datos['imagenes'].get('bsi_url') else None

    # Superponer polígono del lote
    bbox = datos['imagenes'].get('bbox')
    poligono_coords = meta.get('poligono_coords')
    if bbox and poligono_coords:
        if img_rgb_bytes:
            img_rgb_bytes = superponer_poligono(
                img_rgb_bytes, poligono_coords, bbox,
            )
        if img_ndvi_bytes:
            img_ndvi_bytes = superponer_poligono(
                img_ndvi_bytes, poligono_coords, bbox,
            )
        if img_bsi_bytes:
            img_bsi_bytes = superponer_poligono(
                img_bsi_bytes, poligono_coords, bbox,
            )

    # Generar gráfico NDVI
    print("  Generando gráfico temporal...")
    grafico_bytes = generar_grafico_ndvi(
        datos['serie_ndvi'],
        datos['caidas_abruptas'],
        comparacion,
    )

    # Construir documento
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    contenido = []

    # =========================================================================
    # PÁGINA 1
    # =========================================================================

    # Encabezado
    contenido.append(Paragraph('AgTech MRV', styles['Titulo']))
    contenido.append(Paragraph(
        'Reporte de Evidencia Satelital — Certificación ARC/AAPRESID',
        styles['Subtitulo'],
    ))

    contenido.append(HRFlowable(
        width='100%', thickness=1, color=COLOR_PRIMARIO,
        spaceAfter=8, spaceBefore=4,
    ))

    # Datos del lote + mapa de ubicación
    depto = comparacion['nombre_departamento']
    prov = comparacion['provincia']

    info_texto = Paragraph(
        f'<b>Ubicación:</b> {depto}, {prov}<br/>'
        f'<b>Superficie:</b> {meta["superficie_ha"]} ha<br/>'
        f'<b>Coordenadas:</b> {meta["centroide_lat"]}, {meta["centroide_lon"]}<br/>'
        f'<b>Período:</b> {meta["fecha_inicio"]} a {meta["fecha_fin"]}<br/>'
        f'<b>Generado:</b> {meta["fecha_procesamiento"]}',
        styles['CuerpoTexto'],
    )

    # Mapa de Argentina con punto del lote
    mapa_bytes = generar_mapa_ubicacion(
        meta['centroide_lat'], meta['centroide_lon']
    )
    mapa_img = Image(io.BytesIO(mapa_bytes), width=3.2 * cm, height=4.2 * cm)

    header_table = Table(
        [[info_texto, mapa_img]],
        colWidths=[12.5 * cm, 4 * cm],
    )
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
    ]))
    contenido.append(header_table)
    contenido.append(Spacer(1, 4))

    # Imágenes satelitales lado a lado
    contenido.append(Paragraph('Imágenes Satelitales del Lote', styles['SeccionTitulo']))

    img_width = 5.2 * cm
    img_height = 5.2 * cm

    imgs = []
    labels = []

    if img_rgb_bytes:
        imgs.append(Image(io.BytesIO(img_rgb_bytes), width=img_width, height=img_height))
        labels.append(Paragraph('<b>Color real (RGB)</b><br/>Mediana Sentinel-2', styles['Nota']))
    if img_ndvi_bytes:
        imgs.append(Image(io.BytesIO(img_ndvi_bytes), width=img_width, height=img_height))
        labels.append(Paragraph('<b>NDVI</b><br/>Rojo=bajo, Verde=alto', styles['Nota']))
    if img_bsi_bytes:
        imgs.append(Image(io.BytesIO(img_bsi_bytes), width=img_width, height=img_height))
        labels.append(Paragraph('<b>BSI (suelo desnudo)</b><br/>Verde=cubierto, Marrón=expuesto', styles['Nota']))

    if imgs:
        col_w = 5.5 * cm
        img_table = Table(
            [imgs, labels],
            colWidths=[col_w] * len(imgs),
        )
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        contenido.append(img_table)

    contenido.append(Spacer(1, 4))

    # Resultado rápido
    if diag['consistente']:
        color_resultado = '#2d8a4e'
        texto_resultado = 'CONSISTENTE CON SIEMBRA DIRECTA'
    else:
        color_resultado = '#c0392b'
        texto_resultado = 'REQUIERE VERIFICACIÓN ADICIONAL'

    contenido.append(Paragraph(
        f'<font color="{color_resultado}" size="14"><b>{texto_resultado}</b></font>',
        ParagraphStyle('ResultadoInline', parent=styles['Normal'], alignment=TA_CENTER),
    ))
    contenido.append(Spacer(1, 2))

    # Gráfico de desglose de confianza
    confianza_bytes = generar_grafico_confianza(confianza)
    confianza_img = Image(
        io.BytesIO(confianza_bytes), width=14 * cm, height=3.5 * cm,
    )
    contenido.append(confianza_img)

    # =========================================================================
    # PÁGINA 2
    # =========================================================================
    contenido.append(PageBreak())

    # Gráfico temporal
    contenido.append(Paragraph(
        'Evolución Temporal del NDVI', styles['SeccionTitulo'],
    ))

    if grafico_bytes:
        grafico_img = Image(
            io.BytesIO(grafico_bytes),
            width=16.5 * cm,
            height=5.0 * cm,
        )
        contenido.append(grafico_img)

    contenido.append(Spacer(1, 2))

    # Tabla de métricas
    contenido.append(Paragraph('Métricas del Lote', styles['SeccionTitulo']))
    contenido.append(crear_tabla_metricas(metricas, comparacion))

    contenido.append(Spacer(1, 2))

    # Tabla de diagnóstico
    contenido.append(Paragraph(
        'Evaluación de Prácticas', styles['SeccionTitulo'],
    ))
    contenido.append(crear_tabla_diagnostico(diag['criterios']))

    contenido.append(Spacer(1, 2))

    # Conclusión
    contenido.append(Paragraph('Conclusión', styles['SeccionTitulo']))
    contenido.append(Paragraph(diag['conclusion'], styles['Conclusion']))

    contenido.append(Spacer(1, 2))

    # Nota metodológica
    contenido.append(HRFlowable(
        width='100%', thickness=0.5, color=COLOR_GRIS,
        spaceAfter=2, spaceBefore=2,
    ))
    contenido.append(Paragraph('Nota Metodológica', styles['SeccionTitulo']))

    disp = datos['disponibilidad_datos']
    gaps_texto = (
        f"Meses con datos limitados: {', '.join(disp['meses_con_gaps'])}."
        if disp['meses_con_gaps']
        else "Cobertura temporal completa."
    )
    contenido.append(Paragraph(
        '<b>Fuentes:</b> Sentinel-2 L2A (10m, índices ópticos) y Sentinel-1 SAR GRD '
        '(10m, RVI) vía Google Earth Engine. '
        '<b>Metodología:</b> Mediana temporal, comparación zonal contra media departamental '
        '(100m), detección de labranza por convergencia multi-sensor. '
        f'<b>Datos:</b> {disp["total_observaciones"]} observaciones Sentinel-2. {gaps_texto}',
        styles['Nota'],
    ))
    contenido.append(Paragraph(
        '<b>Limitaciones:</b> Reporte automático, evidencia complementaria. '
        'Interpretar con profesional agronómico en contexto de zona y sistema productivo.',
        styles['Nota'],
    ))

    # Footer
    contenido.append(Spacer(1, 3))
    contenido.append(Paragraph(
        'AgTech MRV v0.1 — Generado automáticamente',
        ParagraphStyle(
            'Footer', parent=styles['Nota'],
            alignment=TA_CENTER, textColor=COLOR_GRIS,
        ),
    ))

    # =========================================================================
    # PÁGINA 3 (opcional): Comparación multi-campaña
    # =========================================================================
    if datos_multi and datos_multi.get('campanas'):
        campanas = datos_multi['campanas']
        comp = datos_multi['comparacion_campanas']

        contenido.append(PageBreak())
        contenido.append(Paragraph(
            'Comparación entre Campañas', styles['Titulo'],
        ))
        contenido.append(Spacer(1, 4))

        # Gráfico NDVI superpuesto
        multi_chart_bytes = generar_grafico_ndvi_multicampana(campanas)
        contenido.append(Image(
            io.BytesIO(multi_chart_bytes), width=16.5 * cm, height=5.5 * cm,
        ))
        contenido.append(Spacer(1, 4))

        # Tabla comparativa
        contenido.append(Paragraph(
            'Métricas por Campaña', styles['SeccionTitulo'],
        ))
        contenido.append(crear_tabla_comparacion_campanas(campanas, comp))

        contenido.append(Spacer(1, 6))

        # Conclusión multi-campaña
        if comp['consistente_todas']:
            texto_multi = (
                f'Las {comp["total_campanas"]} campañas analizadas muestran '
                f'resultados consistentes con siembra directa. '
                f'Tendencia NDVI: {comp["ndvi_tendencia"]}.'
            )
        else:
            texto_multi = (
                f'De las {comp["total_campanas"]} campañas analizadas, '
                f'al menos una requiere verificación adicional. '
                f'Tendencia NDVI: {comp["ndvi_tendencia"]}.'
            )
        contenido.append(Paragraph(texto_multi, styles['Conclusion']))

    # Construir PDF
    print("\n  Escribiendo PDF...")
    doc.build(contenido)
    print(f"  PDF generado: {output_path}")
    return output_path


# =============================================================================
# Ejecución directa
# =============================================================================

if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) > 1:
        # Cargar datos desde JSON (output de gee_processor)
        with open(sys.argv[1], 'r') as f:
            datos = json.load(f)
        output = sys.argv[2] if len(sys.argv) > 2 else 'reporte_mrv.pdf'
        generar_pdf(datos, output)
    else:
        print("Uso: python pdf_generator.py <datos.json> [output.pdf]")
        print()
        print("Normalmente se ejecuta desde el pipeline completo:")
        print("  from gee_processor import procesar_lote")
        print("  from pdf_generator import generar_pdf")
        print("  datos = procesar_lote('lote.geojson', '2023-10-01', '2024-04-30')")
        print("  generar_pdf(datos, 'reporte.pdf')")
