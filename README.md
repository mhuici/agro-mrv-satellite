# AgTech MRV — Satellite evidence reports for regenerative agriculture

Given the boundary of an Argentine farm plot, this pipeline pulls Sentinel-1 and Sentinel-2 imagery through Google Earth Engine, computes spectral and radar indices across a growing season, and produces a 2-page PDF report with evidence of no-till farming, cover crops and stubble burning.

It was built as an MVP for the **ARC certification** (Aapresid's Regenerative Environment Certification) and **EUDR** compliance, and validated with an agronomist on real cadastral plots in Córdoba.

> Code comments and report text are in Spanish, the language of the end users.

## What it does

- **Multi-sensor time series** — Sentinel-2 optical (NDVI, EVI, BSI, NBR) with cloud masking, plus Sentinel-1 SAR (RVI), which sees the soil through clouds.
- **Burn detection** — NBR time series and dNBR, with the event date and USGS severity class.
- **Zonal benchmark** — every plot metric is compared against the mean of its department, as the certification requires.
- **Confidence score** — a 0–100 score broken into four components, plus a data-availability check before any conclusion.
- **No-till diagnosis** — a rules-based classifier that states whether the season is consistent with direct seeding.
- **Validation harness** — runs the classifier against ground-truth field records and reports accuracy and a confusion matrix.
- **Multi-season reports** — compares two or more seasons on the same plot.

## Example output

| Plot (Córdoba) | Area | Score | Finding |
|---|---|---|---|
| General San Martín | 496 ha | 99/100 | Consistent with no-till |
| General Roca | 201 ha | 98/100 | Possible burn event |
| Unión | 261 ha | 86/100 | Moderate–high burn |
| Marcos Juárez | 191 ha | 67/100 | Low NBR |
| Unión | 251 ha | 49/100 | Low NDVI, severe burn |

The PDFs are in [`portfolio/`](portfolio/) and [`sample_outputs/`](sample_outputs/).

## Architecture

```
GeoJSON plot + season dates
        │
        ▼
gee_processor.py ── Earth Engine: S2 + S1 collections → indices → time series
        │            burn events · department benchmark · confidence score
        ▼
   metrics (JSON) ──► pdf_generator.py ── ReportLab + matplotlib → 2-page PDF
        │
        ▼
validacion.py ── accuracy vs. ground-truth field records
```

| File | Role |
|---|---|
| `gee_processor.py` | Earth Engine processing: collections, indices, time series, burn detection, benchmark, scoring, diagnosis |
| `pdf_generator.py` | PDF report with satellite imagery, index maps, NDVI curve and metrics table |
| `run_pipeline.py` | CLI for single-season and multi-season runs |
| `validacion.py` | Validation against ground truth |
| `demo_notebook.ipynb` | Interactive walkthrough |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
earthengine authenticate   # needs a Google Cloud project with the Earth Engine API enabled

# One season
python run_pipeline.py sample_polygons/lote_santa_fe.geojson 2023-10-01 2024-04-30 --proyecto YOUR_GEE_PROJECT

# Compare two seasons
python run_pipeline.py sample_polygons/lote_santa_fe.geojson \
  --multi 2022-10-01,2023-04-30 2023-10-01,2024-04-30 --proyecto YOUR_GEE_PROJECT
```

Earth Engine's free Community tier is enough to run the examples. Credentials stay on your machine and are never stored in this repository.

## Stack

Python · Google Earth Engine API · Sentinel-1 SAR · Sentinel-2 L2A · ReportLab · matplotlib · Jupyter

---

Built by [Mariano Huici](https://github.com/mhuici).
