# U3 Saber Hacer · Análisis supervisado con dataset UFO

Corrección del caso práctico de la Unidad 3 usando el dataset de avistamientos del repositorio [Dataset_UFO](https://github.com/EdwinQ12/Dataset_UFO). El problema es una clasificación binaria: predecir si un reporte tendrá cobertura mediática (`Media = Y`).

## Contenido

- `ufo_supervised.py`: limpieza, ingeniería de variables, modelos, métricas y figuras.
- `ufo_data.csv`: dataset original del repositorio Dataset_UFO.
- `ufo_clean.csv`: dataset limpio utilizado en el modelado.
- `resultados/`: figuras PNG y `resultados.json`.
- `docs/`: documento final en PDF y versión editable en DOCX.
- `requirements.txt`: dependencias.

## Requisitos

- Python 3.10 o superior.
- NumPy, pandas y Matplotlib.

## Ejecución

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate.ps1
pip install -r requirements.txt
python ufo_supervised.py
```

## Caso

- Dataset original: 1,179 reportes.
- Dataset limpio: 1,164 reportes.
- Variable objetivo: `media` (1 = `Media = Y`, 0 = sin media).
- Distribución: 749 sin media (64.3 %) y 415 con media (35.7 %).
- Partición: 815 entrenamiento / 349 prueba, estratificada.
- Variables: fecha, ubicación, forma agrupada, longitud y palabras del resumen, banderas de palabras clave.

## Resultados principales

| Modelo | Exactitud | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|
| Regresión logística | 0.6275 | 0.4559 | 0.2500 | 0.3229 |
| k-NN (k=15) | 0.6132 | 0.3962 | 0.1694 | 0.2373 |
| Árbol de decisión | 0.6332 | 0.4500 | 0.1452 | 0.2195 |
| Random Forest | 0.6533 | 0.6667 | 0.0484 | 0.0902 |
| Random Forest optimizado (umbral 0.35) | 0.5702 | 0.4235 | 0.5806 | 0.4898 |

- ROC-AUC del bosque: 0.628.
- Mejor profundidad: 7.
- Mejor umbral: 0.35.

## Interpretación

El rendimiento es modesto porque la cobertura mediática depende de factores externos que no están en el dataset (agenda del medio, novedad del caso, calidad del testimonio). El modelo conservador (umbral 0.5) tiene alta precisión y recall casi nulo; el umbral 0.35 equilibra mejor la detección. El ejercicio demuestra la importancia de elegir la métrica y el punto de operación según el costo del error.

## Nota

Las implementaciones son didácticas y no sustituyen bibliotecas optimizadas. El dataset es público y se conserva en su versión original y limpia para reproducibilidad.

## Repositorio

https://github.com/EdwinQ12/U3_SH_AnalisisSupervisado_UFO


## U4 · Análisis no supervisado

La carpeta `U4/` contiene el análisis de agrupación y reducción de dimensionalidad con K-Means y PCA, usando el mismo dataset UFO. Incluye código, datasets original/limpio/transformado, figuras y README propio.

- Código: `U4/ufo_kmeans_pca.py`.
- Documento: `docs/U4_SABER_HACER_AnalisisNoSupervisado_CortesJaraQuiroz_IDGS-9C.pdf`.
- Resultados: `U4/resultados/`.

